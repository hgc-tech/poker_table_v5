// Linear Monte Carlo CFR with external sampling (as in Pluribus), v5 storage.
//
// Each iteration deals one set of cards and traverses the tree once for every
// player. At the traverser's nodes every action is explored; at other
// players' nodes one action is sampled from the current strategy. During the
// linear phase regrets (and the preflop average) are discounted by t/(t+1)
// every `discount_interval` iterations (Linear CFR). After `prune_after`
// iterations, actions with very negative regret are skipped 95% of the time
// (negative-regret pruning), except on the river.
//
// Memory (the v5 changes):
//  * Regrets are 2-byte integers. The 16 buckets of one node that share a
//    chunk also share a power-of-two scale: when a regret would overflow, the
//    whole chunk is halved and its scale doubles, so large regrets never
//    saturate. Updates are rounded stochastically, so small updates are not
//    lost (they round up with the right probability).
//  * Regrets have a floor (regret_floor); the floor also bounds how long a
//    pruned action stays pruned.
//  * Chunks are allocated only when the traverser first reaches them, from
//    one arena with a hard cap (memory_gb). Past the cap, unreached infosets
//    keep playing uniformly and are not trained; the trainer says so.
//  * The average strategy is kept for preflop only. After the flop the bots
//    play the current strategy (from regrets), refined by search.
#pragma once

#include <atomic>
#include <csignal>
#include <sys/mman.h>

#include "pb_abstraction.h"
#include "pb_game.h"

namespace pb {

struct TrainOptions {
    int threads = 4;
    uint64_t discount_interval = 10000000;
    uint64_t linear_until = 2000000000;
    uint64_t prune_after = 200000000;
    double prune_threshold = -0.5;   // in units of the chips at the table (the utility scale)
    double regret_floor = -2.0;      // same units
    double checkpoint_minutes = 30;
    double memory_gb = 22;           // hard cap for regrets (GiB)
    void from_kv(const KV& kv) {
        discount_interval = (uint64_t)kv.num("discount_interval", (double)discount_interval);
        linear_until = (uint64_t)kv.num("linear_until", (double)linear_until);
        prune_after = (uint64_t)kv.num("prune_after", (double)prune_after);
        prune_threshold = kv.num("prune_threshold", prune_threshold);
        regret_floor = kv.num("regret_floor", regret_floor);
        checkpoint_minutes = kv.num("checkpoint_minutes", checkpoint_minutes);
        memory_gb = kv.num("memory_gb", memory_gb);
    }
};

static std::atomic<bool> g_stop(false);
inline void on_signal(int) { g_stop = true; }

struct Dealt {
    int hole[MAX_PLAYERS][2];
    int board[5];
    int16_t bucket[MAX_PLAYERS][4];
    uint32_t value[MAX_PLAYERS];
    bool values_done = false;
};

// Chunked 2-byte regret store with lazy allocation.
struct RegretStore {
    std::vector<uint32_t> cptr;      // per chunk: arena offset in 16-byte units (0 = not allocated)
    std::vector<uint8_t> cexp;       // per chunk: scale exponent
    int16_t* arena = nullptr;
    uint64_t cap = 0;                // bytes reserved
    std::atomic<uint64_t> used{16};  // bytes used; the first 16 bytes are unused (offset 0 = none)
    std::atomic<bool> full{false};

    ~RegretStore() { release(); }
    void release() {
        if (arena) munmap(arena, cap);
        arena = nullptr;
    }
    void init(uint64_t chunks, uint64_t cap_bytes) {
        release();
        cptr.assign(chunks, 0);
        cexp.assign(chunks, 0);
        cap = std::max<uint64_t>(cap_bytes, 1 << 20);
        cap = (cap + 4095) & ~4095ull;
        if (cap / 16 >= 0xFFFFFFFFull) cap = 0xFFFFFFF0ull * 16 / 16 * 16;
        void* p = mmap(nullptr, cap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) throw std::runtime_error("cannot reserve memory for regrets");
        arena = (int16_t*)p;
        used = 16;
        full = false;
    }
    inline int16_t* get(uint64_t c) const {
        uint32_t o = __atomic_load_n(&cptr[c], __ATOMIC_ACQUIRE);
        return o ? arena + (size_t)o * 8 : nullptr;
    }
    inline int16_t* get_or_alloc(uint64_t c, int nact) {
        int16_t* p = get(c);
        if (p || full.load(std::memory_order_relaxed)) return p;
        uint64_t bytes = (uint64_t)CHUNK * nact * 2;   // a multiple of 32
        uint64_t at = used.fetch_add(bytes);
        if (at + bytes > cap) { full = true; return nullptr; }
        uint32_t want = (uint32_t)(at / 16), expect = 0;
        if (__atomic_compare_exchange_n(&cptr[c], &expect, want, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            return arena + (size_t)want * 8;
        return arena + (size_t)expect * 8;   // another thread won; our block is wasted (rare)
    }
    uint64_t bytes_used() const { return std::min<uint64_t>(used.load(), cap); }
};

struct Trainer {
    Tree tree;
    const CardAbstraction* abs = nullptr;
    RegretStore store;
    std::vector<float> avg;          // preflop average strategy
    uint64_t iterations = 0;
    double seconds = 0;
    TrainOptions opt;
    double unit = 1;                 // utility per regret unit at exponent 0
    float floor_units = -30000;
    double prune_true = -1;

    static constexpr uint32_t MAGIC = 0x35424350;   // "PCB5"

    std::string tree_text() const {
        std::ostringstream o;
        o << tree.hash_text() << "abstraction=" << abs->hash << "\n";
        return o.str();
    }

    void init(int players, const Stacks& stack_bb, const GameConfig& g, const CardAbstraction& a, int level,
              bool allocate = true) {
        abs = &a;
        const auto& L = a.levels[level];
        int K[4] = {169, L.K[1], L.K[2], L.K[3]};
        tree.build(players, stack_bb, g, K, level);
        unit = tree.util_scale / 2000.0;
        prune_true = opt.prune_threshold * tree.util_scale;
        floor_units = (float)std::max(-32000.0, opt.regret_floor * tree.util_scale / unit);
        if (allocate) {
            store.init(tree.total_chunks, (uint64_t)(opt.memory_gb * 1073741824.0));
            avg.assign(tree.avg_slots, 0.f);
        }
        iterations = 0;
        seconds = 0;
    }

    // Strategy at a node for a bucket (what the bots play): preflop average,
    // postflop regret matching; with no positive regret, the action with the
    // highest regret. Unreached infosets: uniform. Returns false if unreached.
    bool play_strategy(const Node& nd, int b, float* p) const {
        int na = nd.nact;
        if (nd.street == 0) {
            const float* S = &avg[nd.avg + (size_t)b * na];
            double s = 0;
            for (int a = 0; a < na; ++a) s += S[a];
            if (s > 0) { for (int a = 0; a < na; ++a) p[a] = (float)(S[a] / s); return true; }
        }
        const int16_t* c = store.get(nd.offset + b / CHUNK);
        if (!c) { for (int a = 0; a < na; ++a) p[a] = 1.f / na; return false; }
        const int16_t* R = c + (size_t)(b % CHUNK) * na;
        double s = 0;
        for (int a = 0; a < na; ++a) s += R[a] > 0 ? R[a] : 0;
        if (s > 0) { for (int a = 0; a < na; ++a) p[a] = (float)((R[a] > 0 ? R[a] : 0) / s); return true; }
        int best = 0;
        for (int a = 1; a < na; ++a) if (R[a] > R[best]) best = a;
        bool all_same = true;
        for (int a = 1; a < na; ++a) all_same &= R[a] == R[0];
        for (int a = 0; a < na; ++a) p[a] = all_same ? 1.f / na : (a == best ? 1.f : 0.f);
        return true;
    }

    // Checkpoints ----------------------------------------------------------------
    static void wr(FILE* f, const void* p, size_t bytes) {
        const char* c = (const char*)p;
        const size_t CH = 1 << 26;
        for (size_t i = 0; i < bytes; i += CH)
            if (std::fwrite(c + i, 1, std::min(CH, bytes - i), f) != std::min(CH, bytes - i))
                throw std::runtime_error("error writing checkpoint (disk full?)");
    }
    static void rd(FILE* f, void* p, size_t bytes) {
        char* c = (char*)p;
        const size_t CH = 1 << 26;
        for (size_t i = 0; i < bytes; i += CH)
            if (std::fread(c + i, 1, std::min(CH, bytes - i), f) != std::min(CH, bytes - i))
                throw std::runtime_error("checkpoint is truncated");
    }
    void save(const std::string& path) const {
        std::string tmp = path + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot write " + tmp);
        std::string txt = tree_text();
        uint32_t len = (uint32_t)txt.size();
        uint64_t chunks = tree.total_chunks, used = store.bytes_used(), na = avg.size();
        wr(f, &MAGIC, 4);
        wr(f, &len, 4);
        wr(f, txt.data(), len);
        wr(f, &iterations, 8);
        wr(f, &seconds, 8);
        wr(f, &chunks, 8);
        wr(f, &used, 8);
        wr(f, &na, 8);
        wr(f, store.cptr.data(), chunks * 4);
        wr(f, store.cexp.data(), chunks);
        wr(f, store.arena, used);
        wr(f, avg.data(), na * 4);
        if (std::fclose(f) != 0) throw std::runtime_error("error writing " + tmp + " (disk full?)");
        std::remove(path.c_str());
        if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("cannot rename " + tmp);
    }
    static std::string read_header(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open " + path);
        uint32_t magic = 0, len = 0;
        if (std::fread(&magic, 4, 1, f) != 1 || magic != MAGIC || std::fread(&len, 4, 1, f) != 1) {
            std::fclose(f);
            throw std::runtime_error(path + " is not a v5 checkpoint");
        }
        std::string txt(len, '\0');
        if (std::fread(&txt[0], 1, len, f) != len) txt.clear();
        std::fclose(f);
        return txt;
    }
    // Returns false if the file does not exist. Throws if it does not match.
    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        uint32_t magic = 0, len = 0;
        if (std::fread(&magic, 4, 1, f) != 1 || magic != MAGIC) {
            std::fclose(f);
            throw std::runtime_error(path + " is not a v5 checkpoint");
        }
        rd(f, &len, 4);
        std::string txt(len, '\0');
        rd(f, &txt[0], len);
        if (txt != tree_text()) {
            std::fclose(f);
            throw std::runtime_error(path + " was trained with a different configuration, bucket level or card "
                                     "abstraction.\nUse a new checkpoint folder, or restore the old settings.\n"
                                     "Checkpoint:\n" + txt + "Current:\n" + tree_text());
        }
        uint64_t chunks = 0, used = 0, na = 0;
        rd(f, &iterations, 8);
        rd(f, &seconds, 8);
        rd(f, &chunks, 8);
        rd(f, &used, 8);
        rd(f, &na, 8);
        if (chunks != tree.total_chunks || na != avg.size()) {
            std::fclose(f);
            throw std::runtime_error("checkpoint size mismatch");
        }
        if (used > store.cap) {
            std::fclose(f);
            std::ostringstream o;
            o << path << " holds " << used / 1073741824.0 << " GiB of regrets, more than memory_gb = " << opt.memory_gb
              << ". Raise memory_gb to resume it.";
            throw std::runtime_error(o.str());
        }
        rd(f, store.cptr.data(), chunks * 4);
        rd(f, store.cexp.data(), chunks);
        rd(f, store.arena, used);
        rd(f, avg.data(), na * 4);
        std::fclose(f);
        store.used = used;
        return true;
    }

    // Strategy export --------------------------------------------------------------
    // Format "PBS5": header text, iterations, chunk count, data bytes, then a
    // uint32 per chunk (data offset in 16-byte units, 0xFFFFFFFF = never
    // reached) and the data: one byte per action (probability x 255).
    void export_strategy(const std::string& path) const {
        std::string tmp = path + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot write " + tmp);
        std::string txt = tree_text();
        uint32_t magic = 0x35534250, len = (uint32_t)txt.size();   // "PBS5"
        uint64_t chunks = tree.total_chunks;
        std::vector<uint32_t> table(chunks, 0xFFFFFFFFu);
        uint64_t data = 0;
        for (const Node& nd : tree.nodes) {
            if (nd.type != 0) continue;
            uint64_t nch = (tree.K[nd.street] + CHUNK - 1) / CHUNK;
            for (uint64_t c = 0; c < nch; ++c) {
                bool have = nd.street == 0 || store.get(nd.offset + c) != nullptr;
                if (!have) continue;
                table[nd.offset + c] = (uint32_t)(data / 16);
                data += (uint64_t)CHUNK * nd.nact;
            }
        }
        if (data / 16 >= 0xFFFFFFFFull) throw std::runtime_error("strategy too large to export");
        wr(f, &magic, 4);
        wr(f, &len, 4);
        wr(f, txt.data(), len);
        wr(f, &iterations, 8);
        wr(f, &chunks, 8);
        wr(f, &data, 8);
        wr(f, table.data(), chunks * 4);
        std::vector<uint8_t> buf;
        buf.reserve(1 << 20);
        for (const Node& nd : tree.nodes) {
            if (nd.type != 0) continue;
            int na = nd.nact;
            uint64_t nch = (tree.K[nd.street] + CHUNK - 1) / CHUNK;
            for (uint64_t c = 0; c < nch; ++c) {
                if (table[nd.offset + c] == 0xFFFFFFFFu) continue;
                for (int i = 0; i < CHUNK; ++i) {
                    int b = (int)c * CHUNK + i;
                    float p[16];
                    if (b < tree.K[nd.street]) play_strategy(nd, b, p);
                    else for (int a = 0; a < na; ++a) p[a] = 0;
                    for (int a = 0; a < na; ++a) buf.push_back((uint8_t)std::lround(std::min(1.f, p[a]) * 255.0));
                }
                if (buf.size() >= (1 << 20)) { wr(f, buf.data(), buf.size()); buf.clear(); }
            }
        }
        wr(f, buf.data(), buf.size());
        if (std::fclose(f) != 0) throw std::runtime_error("error writing " + tmp);
        std::remove(path.c_str());
        if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("cannot rename " + tmp);
    }

    // Sampling -------------------------------------------------------------------
    void deal(Rng& r, Dealt& d) const {
        int n = tree.n;
        uint64_t used = 0;
        auto draw = [&]() {
            for (;;) {
                int c = (int)r.below(52);
                if (!(used & (1ull << c))) { used |= 1ull << c; return c; }
            }
        };
        for (int i = 0; i < n; ++i) { d.hole[i][0] = draw(); d.hole[i][1] = draw(); }
        for (int i = 0; i < 5; ++i) d.board[i] = draw();
        int lv = tree.level;
        for (int i = 0; i < n; ++i) {
            int a = d.hole[i][0], b = d.hole[i][1];
            d.bucket[i][0] = holes().cls[holes().index[a][b]];
            d.bucket[i][1] = (int16_t)abs->at_level(lv, 1, abs->flop_bucket(d.board, a, b));
            d.bucket[i][2] = (int16_t)abs->at_level(lv, 2, abs->turn_bucket(d.board, a, b));
            d.bucket[i][3] = (int16_t)abs->at_level(lv, 3, abs->river_bucket(d.board, a, b));
        }
        d.values_done = false;
    }
    void ensure_values(Dealt& d) const {
        if (d.values_done) return;
        const Eval& E = evaltab();
        uint64_t bsm = 0;
        for (int i = 0; i < 5; ++i) bsm |= E.sbit[d.board[i]];
        for (int i = 0; i < tree.n; ++i)
            d.value[i] = eval_suitmask(bsm | E.sbit[d.hole[i][0]] | E.sbit[d.hole[i][1]]);
        d.values_done = true;
    }

    // Halve every regret in a chunk (stochastic rounding) and double its scale.
    void halve_chunk(int16_t* c, int na, uint64_t ci, Rng& r) {
        uint64_t bits = r.next();
        int k = 0;
        for (int i = 0; i < CHUNK * na; ++i) {
            int v = c[i];
            int h = v >> 1;               // floor(v / 2)
            if ((v & 1) && ((bits >> (k++ & 63)) & 1)) h += 1;
            c[i] = (int16_t)h;
        }
        if (cexp_at(ci) < 60) store.cexp[ci]++;
    }
    inline int cexp_at(uint64_t ci) const { return store.cexp[ci]; }

    // Traversal ------------------------------------------------------------------
    float traverse(int idx, int p, Dealt& d, Rng& r, bool prune) {
        const Node& nd = tree.nodes[idx];
        if (nd.type == 1) return tree.terms[nd.offset].fold_util[p];
        if (nd.type == 2) {
            ensure_values(d);
            float u[MAX_PLAYERS];
            showdown_utility(tree, tree.terms[nd.offset], d.value, u);
            return u[p];
        }
        int q = nd.player, na = nd.nact, st = nd.street;
        int b = d.bucket[q][st];
        uint64_t ci = nd.offset + b / CHUNK;
        float sigma[16];
        if (q == p) {
            int16_t* c = store.get_or_alloc(ci, na);
            int16_t* R = c ? c + (size_t)(b % CHUNK) * na : nullptr;
            float pos = 0;
            if (R) for (int a = 0; a < na; ++a) { sigma[a] = R[a] > 0 ? (float)R[a] : 0.f; pos += sigma[a]; }
            if (pos > 0) for (int a = 0; a < na; ++a) sigma[a] /= pos;
            else for (int a = 0; a < na; ++a) sigma[a] = 1.f / na;
            float v[16];
            bool explored[16];
            float ev = 0;
            double sc = 0;
            if (R) sc = std::ldexp(1.0, -store.cexp[ci]) / unit;   // utility -> regret units
            bool do_prune = R && prune && st < 3 && r.uniform() < 0.95;
            float prune_units = (float)(prune_true * sc);
            for (int a = 0; a < na; ++a) {
                const Node& ch = tree.nodes[nd.child + a];
                if (do_prune && R[a] < prune_units && ch.type == 0) { explored[a] = false; continue; }
                explored[a] = true;
                v[a] = traverse(nd.child + a, p, d, r, prune);
                ev += sigma[a] * v[a];
            }
            if (!R) return ev;
            // Update with stochastic rounding; halve the chunk until nothing overflows.
            sc = std::ldexp(1.0, -store.cexp[ci]) / unit;
            for (int guard = 0; guard < 40; ++guard) {
                bool over = false;
                for (int a = 0; a < na && !over; ++a)
                    if (explored[a] && R[a] + (v[a] - ev) * sc > 32000.0) over = true;
                if (!over) break;
                halve_chunk(c, na, ci, r);
                sc *= 0.5;
            }
            float fl = std::max(-32000.f, (float)(floor_units * std::ldexp(1.0, -store.cexp[ci])));
            fl = std::min(fl, -1.f);
            for (int a = 0; a < na; ++a) {
                if (!explored[a]) continue;
                double x = R[a] + (v[a] - ev) * sc;
                double fx = std::floor(x);
                if (r.uniform() < x - fx) fx += 1;
                if (fx < fl) fx = fl;
                if (fx > 32767) fx = 32767;
                R[a] = (int16_t)fx;
            }
            return ev;
        }
        const int16_t* c = store.get(ci);
        float pos = 0;
        if (c) {
            const int16_t* R = c + (size_t)(b % CHUNK) * na;
            for (int a = 0; a < na; ++a) { sigma[a] = R[a] > 0 ? (float)R[a] : 0.f; pos += sigma[a]; }
        }
        if (pos > 0) for (int a = 0; a < na; ++a) sigma[a] /= pos;
        else for (int a = 0; a < na; ++a) sigma[a] = 1.f / na;
        if (st == 0) {
            float* S = &avg[nd.avg + (size_t)b * na];
            for (int a = 0; a < na; ++a) S[a] += sigma[a];
        }
        double x = r.uniform();
        int a = 0;
        for (; a < na - 1; ++a) { x -= sigma[a]; if (x <= 0) break; }
        return traverse(nd.child + a, p, d, r, prune);
    }

    void discount(double f) {
        uint64_t n = store.bytes_used() / 2;
        int T = std::max(1, opt.threads);
        uint64_t ch = (n + T - 1) / T;
        parallel_for(T, T, [&](int t, int) {
            Rng r(0x51ED + t + iterations);
            uint64_t lo = t * ch, hi = std::min(n, lo + ch);
            for (uint64_t i = std::max<uint64_t>(lo, 8); i < hi; ++i) {
                int16_t v = store.arena[i];
                if (!v) continue;
                double x = v * f, fx = std::floor(x);
                if (r.uniform() < x - fx) fx += 1;
                store.arena[i] = (int16_t)fx;
            }
            uint64_t m = avg.size(), c2 = (m + T - 1) / T;
            for (uint64_t i = t * c2; i < std::min(m, (uint64_t)(t + 1) * c2); ++i) avg[i] = (float)(avg[i] * f);
        });
    }

    // Train until `minutes` of wall time or `max_iters` iterations, whichever
    // comes first (0 = no limit), or until Ctrl-C.
    void train(double minutes, uint64_t max_iters, const std::string& path, uint64_t seed) {
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        double start = now_seconds(), last_ck = start, last_report = start;
        uint64_t start_iters = iterations;
        int T = std::max(1, opt.threads);
        uint64_t batch = 2000;
        bool warned_full = false;
        std::printf("Training %d players, %s bb, bucket level %d (%d/%d/%d): %zu nodes, %.2f GiB if every "
                    "infoset is reached (cap %.1f GiB), %d threads\n",
                    tree.n, tree.prof.label().c_str(), tree.level, tree.K[1], tree.K[2], tree.K[3], tree.nodes.size(),
                    tree.full_bytes() / 1073741824.0, opt.memory_gb, T);
        std::printf("Starting at iteration %llu. Press Ctrl-C to save and stop.\n", (unsigned long long)iterations);
        std::fflush(stdout);
        while (!g_stop) {
            double el = now_seconds() - start;
            if (minutes > 0 && el >= minutes * 60) break;
            if (max_iters > 0 && iterations >= max_iters) break;
            uint64_t todo = batch;
            if (max_iters > 0) todo = std::min(todo, max_iters - iterations);
            if (iterations < opt.linear_until) {
                uint64_t next = (iterations / opt.discount_interval + 1) * opt.discount_interval;
                todo = std::min(todo, next - iterations);
            }
            bool prune = iterations >= opt.prune_after;
            std::atomic<uint64_t> done(0);
            std::vector<std::thread> pool;
            for (int t = 0; t < T; ++t)
                pool.emplace_back([&, t]() {
                    Rng r(seed ^ (iterations * 6364136223846793005ull) ^ (0x9E37ull * (t + 1)));
                    Dealt d;
                    while (!g_stop) {
                        uint64_t k = done.fetch_add(1);
                        if (k >= todo) break;
                        deal(r, d);
                        for (int p = 0; p < tree.n; ++p) traverse(0, p, d, r, prune);
                    }
                });
            for (auto& th : pool) th.join();
            iterations += std::min<uint64_t>(done.load(), todo);
            if (iterations < opt.linear_until && iterations % opt.discount_interval == 0) {
                double t = (double)(iterations / opt.discount_interval);
                discount(t / (t + 1));
            }
            if (store.full && !warned_full) {
                warned_full = true;
                std::printf("  WARNING: the regret memory cap (memory_gb = %.1f) is full. Infosets reached from now "
                            "on are not trained. Use a coarser bucket level for this blueprint.\n", opt.memory_gb);
            }
            double now = now_seconds();
            if (now - last_report >= 30) {
                double rate = (iterations - start_iters) / std::max(1e-9, now - start);
                std::printf("  iteration %llu  (%.0f iterations/s, %.2f GiB of regrets, %.1f min this run, %.1f h "
                            "total)\n",
                            (unsigned long long)iterations, rate, store.bytes_used() / 1073741824.0,
                            (now - start) / 60, (seconds + now - last_ck) / 3600);
                std::fflush(stdout);
                last_report = now;
            }
            if (batch < 200000 && now - start > 1) {
                double rate = (iterations - start_iters) / (now - start);
                batch = std::max<uint64_t>(2000, (uint64_t)rate);
            }
            if (now - last_ck >= opt.checkpoint_minutes * 60) {
                seconds += now - last_ck;
                last_ck = now;
                save(path);
                std::printf("  checkpoint saved to %s\n", path.c_str());
                std::fflush(stdout);
            }
        }
        seconds += now_seconds() - last_ck;
        save(path);
        std::printf("Stopped at iteration %llu (%.2f h trained in total, %.2f GiB of regrets). Saved %s\n",
                    (unsigned long long)iterations, seconds / 3600, store.bytes_used() / 1073741824.0, path.c_str());
        std::fflush(stdout);
    }
};

// Finest bucket level whose full tree fits in `memory_gb`, or -1 if none.
inline int choose_level(int players, const Stacks& stack_bb, const GameConfig& g, const CardAbstraction& a,
                        double memory_gb, std::vector<double>* gb_out = nullptr) {
    int pick = -1;
    Tree t;
    int K0[4] = {169, 1, 1, 1};
    t.build(players, stack_bb, g, K0, 0);
    for (int l = 0; l < a.nlevels(); ++l) {
        const auto& L = a.levels[l];
        int K[4] = {169, L.K[1], L.K[2], L.K[3]};
        double gb = t.bytes_for(K) / 1073741824.0;
        if (gb_out) gb_out->push_back(gb);
        if (pick < 0 && gb <= memory_gb) pick = l;
    }
    return pick;
}

}  // namespace pb
