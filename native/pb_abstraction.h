// Card abstraction (v5).
//
// Preflop: the 169 strategically distinct starting hands (lossless).
// River:   opponent cluster hand strength (OCHS): the hand's equity against
//          each of 8 groups of opponent hands (preflop classes grouped by
//          strength), clustered with k-means. Looked up from a table of every
//          canonical river board.
// Turn:    histogram of the hand's river equity (against a random hand) over
//          the 46 possible rivers, 50 bins, clustered with k-means under earth
//          mover's distance (exact in one dimension).
// Flop:    potential-aware: histogram over the turn buckets the hand reaches
//          (47 turn cards), clustered with k-means under an approximate earth
//          mover's distance (Ganzfried & Sandholm 2014) whose ground distance
//          is the EMD between turn centroids.
//
// Bucket ids are sorted by strength (0 = weakest).
//
// Levels: level 0 holds the finest buckets. Each further level merges the
// level-0 buckets of every street into fewer buckets (weighted k-means over
// the level-0 centroids). A blueprint trains at one level; large trees use a
// coarser level so they fit in memory.
#pragma once

#include <atomic>
#include <chrono>
#include <numeric>
#include <thread>

#include "pb_core.h"

namespace pb {

inline void parallel_for(int n, int threads, const std::function<void(int, int)>& fn) {
    std::atomic<int> next(0);
    std::vector<std::thread> pool;
    threads = std::max(1, std::min(threads, n));
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t]() {
            for (;;) {
                int i = next.fetch_add(1);
                if (i >= n) break;
                fn(i, t);
            }
        });
    for (auto& th : pool) th.join();
}

inline double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

constexpr int OCHS = 8;          // opponent groups for the river
constexpr int TURN_BINS = 50;    // equity histogram bins for the turn

struct CardAbstraction {
    int K[4] = {169, 0, 0, 0};                  // level-0 bucket counts
    BoardIndex flop_idx, turn_idx, river_idx;
    std::vector<uint16_t> flop_tab, turn_tab, river_tab;   // [canonical board][hole in canonical frame]
    std::vector<uint8_t> ochs_group;            // [hole] opponent group 0..7
    std::vector<float> river_cent;              // [K_river][OCHS]
    std::vector<float> turn_cent;               // [K_turn][TURN_BINS] cumulative
    std::vector<float> flop_cent;               // [K_flop][flop feature buckets] masses
    int flop_feature_level = 0;                 // turn level used for the flop features
    struct Level {
        int K[4] = {169, 0, 0, 0};
        std::vector<uint16_t> map[4];           // level-0 bucket -> bucket at this level (streets 1..3)
    };
    std::vector<Level> levels;
    uint64_t hash = 0;
    bool loaded = false;

    int nlevels() const { return (int)levels.size(); }
    inline int at_level(int level, int street, int b) const {
        if (street == 0 || level == 0 || b < 0) return b;
        return levels[level].map[street][b];
    }

    // Lookups (level 0) -------------------------------------------------------
    inline int map_hole(int perm, int c1, int c2) const {
        const uint8_t* p = suitperms().p[perm];
        return holes().index[apply_perm(c1, p)][apply_perm(c2, p)];
    }
    inline int flop_bucket(const int* b3, int c1, int c2) const {
        int perm, id = flop_idx.lookup(b3, &perm);
        return flop_tab[(size_t)id * NUM_HOLES + map_hole(perm, c1, c2)];
    }
    inline int turn_bucket(const int* b4, int c1, int c2) const {
        int perm, id = turn_idx.lookup(b4, &perm);
        return turn_tab[(size_t)id * NUM_HOLES + map_hole(perm, c1, c2)];
    }
    inline int river_bucket(const int* b5, int c1, int c2) const {
        int perm, id = river_idx.lookup(b5, &perm);
        return river_tab[(size_t)id * NUM_HOLES + map_hole(perm, c1, c2)];
    }
    inline int bucket(int street, const int* board, int c1, int c2) const {
        if (street == 0) return holes().cls[holes().index[c1][c2]];
        if (street == 1) return flop_bucket(board, c1, c2);
        if (street == 2) return turn_bucket(board, c1, c2);
        return river_bucket(board, c1, c2);
    }
    // Level-0 buckets of all 1326 holes on a board of n cards (0 = preflop).
    // -1 if the hole touches the board.
    void all_buckets(const int* board, int n, int16_t* out) const {
        const Holes& H = holes();
        uint64_t bm = 0;
        for (int i = 0; i < n; ++i) bm |= 1ull << board[i];
        if (n == 0) { for (int h = 0; h < NUM_HOLES; ++h) out[h] = H.cls[h]; return; }
        const BoardIndex& bi = n == 3 ? flop_idx : n == 4 ? turn_idx : river_idx;
        const std::vector<uint16_t>& tb = n == 3 ? flop_tab : n == 4 ? turn_tab : river_tab;
        int perm, id = bi.lookup(board, &perm);
        const uint16_t* tab = tb.data() + (size_t)id * NUM_HOLES;
        const uint8_t* p = suitperms().p[perm];
        for (int h = 0; h < NUM_HOLES; ++h) {
            if (H.mask[h] & bm) { out[h] = -1; continue; }
            out[h] = (int16_t)tab[H.index[apply_perm(H.c1[h], p)][apply_perm(H.c2[h], p)]];
        }
    }

    // Persistence --------------------------------------------------------------
    template <class T>
    static void write_vec(const std::string& path, const std::vector<T>& v) {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot write " + path);
        uint64_t n = v.size();
        std::fwrite(&n, 8, 1, f);
        if (n) std::fwrite(v.data(), sizeof(T), n, f);
        if (std::fclose(f) != 0) throw std::runtime_error("error writing " + path);
    }
    template <class T>
    static void read_vec(const std::string& path, std::vector<T>& v) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot read " + path + " (build the abstraction first)");
        uint64_t n = 0;
        if (std::fread(&n, 8, 1, f) != 1) throw std::runtime_error("bad file " + path);
        v.resize(n);
        if (n && std::fread(v.data(), sizeof(T), n, f) != n) throw std::runtime_error("short file " + path);
        std::fclose(f);
    }
    void compute_hash() {
        uint64_t h = 1469598103934665603ull;
        h = fnv1a("v5,", h);
        for (int s = 0; s < 4; ++s) h = fnv1a(std::to_string(K[s]) + ",", h);
        auto mix = [&](const void* p, size_t n, size_t step) {
            const unsigned char* c = (const unsigned char*)p;
            for (size_t i = 0; i < n; i += step) { h ^= c[i]; h *= 1099511628211ull; }
        };
        mix(flop_tab.data(), flop_tab.size() * 2, 97);
        mix(turn_tab.data(), turn_tab.size() * 2, 97);
        mix(river_tab.data(), river_tab.size() * 2, 997);
        for (auto& L : levels)
            for (int s = 1; s < 4; ++s) mix(L.map[s].data(), L.map[s].size() * 2, 1);
        hash = h;
    }
    void save(const std::string& dir) const {
        write_vec(dir + "/flop.bin", flop_tab);
        write_vec(dir + "/turn.bin", turn_tab);
        write_vec(dir + "/river.bin", river_tab);
        write_vec(dir + "/ochs_groups.bin", ochs_group);
        write_vec(dir + "/river_centroids.bin", river_cent);
        write_vec(dir + "/turn_centroids.bin", turn_cent);
        write_vec(dir + "/flop_centroids.bin", flop_cent);
        std::vector<uint16_t> lv;
        for (size_t l = 1; l < levels.size(); ++l)
            for (int s = 1; s < 4; ++s) lv.insert(lv.end(), levels[l].map[s].begin(), levels[l].map[s].end());
        write_vec(dir + "/levels.bin", lv);
        std::ofstream m(dir + "/meta.txt");
        m << "# Card abstraction built by pbcfr (v5). Level 0 is the finest.\n";
        m << "version = 5\n";
        m << "levels = " << levels.size() << "\n";
        const char* nm[4] = {"", "flop", "turn", "river"};
        for (int s = 1; s < 4; ++s) {
            m << nm[s] << "_buckets =";
            for (auto& L : levels) m << " " << L.K[s];
            m << "\n";
        }
        m << "flop_feature_level = " << flop_feature_level << "\n";
        m << "hash = " << hash << "\n";
    }
    void load(const std::string& dir) {
        KV meta;
        meta.load(dir + "/meta.txt");
        if ((int)meta.num("version", 0) != 5)
            throw std::runtime_error(dir + " is not a v5 card abstraction (build one with train_cfr.py --abstraction)");
        int nl = (int)meta.num("levels", 1);
        levels.assign(nl, Level());
        const char* nm[4] = {"", "flop_buckets", "turn_buckets", "river_buckets"};
        for (int s = 1; s < 4; ++s) {
            std::vector<double> k = meta.list(nm[s]);
            if ((int)k.size() != nl) throw std::runtime_error("bad meta.txt in " + dir);
            for (int l = 0; l < nl; ++l) levels[l].K[s] = (int)k[l];
            K[s] = levels[0].K[s];
        }
        flop_feature_level = (int)meta.num("flop_feature_level", 0);
        read_vec(dir + "/flop.bin", flop_tab);
        read_vec(dir + "/turn.bin", turn_tab);
        read_vec(dir + "/river.bin", river_tab);
        read_vec(dir + "/ochs_groups.bin", ochs_group);
        read_vec(dir + "/river_centroids.bin", river_cent);
        read_vec(dir + "/turn_centroids.bin", turn_cent);
        read_vec(dir + "/flop_centroids.bin", flop_cent);
        std::vector<uint16_t> lv;
        read_vec(dir + "/levels.bin", lv);
        size_t pos = 0;
        for (int l = 1; l < nl; ++l)
            for (int s = 1; s < 4; ++s) {
                if (pos + K[s] > lv.size()) throw std::runtime_error("levels.bin is too short");
                levels[l].map[s].assign(lv.begin() + pos, lv.begin() + pos + K[s]);
                pos += K[s];
            }
        build_indexes();
        if (flop_tab.size() != flop_idx.boards.size() * NUM_HOLES ||
            turn_tab.size() != turn_idx.boards.size() * NUM_HOLES ||
            river_tab.size() != river_idx.boards.size() * NUM_HOLES)
            throw std::runtime_error("abstraction tables do not match the board index");
        compute_hash();
        loaded = true;
    }
    void build_indexes() {
        flop_idx.build(3);
        turn_idx.build(4);
        river_idx.build(5);
    }

    // Building -------------------------------------------------------------------
    struct BuildOptions {
        int threads = 4;
        int river_sample_points = 2000000;
        int turn_sample_points = 400000;
        int flop_sample_points = 200000;
        int flop_feature_buckets = 200;   // turn buckets used to describe flop hands
        int max_iters = 30;
        uint64_t seed = 12345;
    };

    static void random_board(Rng& r, int n, uint64_t used, int* out) {
        int k = 0;
        while (k < n) {
            int c = (int)r.below(52);
            if (used & (1ull << c)) continue;
            used |= 1ull << c;
            out[k++] = c;
        }
    }

    // fk/tk/rk: bucket counts per level (level 0 first), same length.
    void build(const std::vector<int>& fk, const std::vector<int>& tk, const std::vector<int>& rk,
               const BuildOptions& o) {
        int nl = (int)fk.size();
        if (nl < 1 || (int)tk.size() != nl || (int)rk.size() != nl)
            throw std::runtime_error("flop_buckets, turn_buckets and river_buckets need the same number of levels");
        levels.assign(nl, Level());
        for (int l = 0; l < nl; ++l) {
            levels[l].K[1] = fk[l]; levels[l].K[2] = tk[l]; levels[l].K[3] = rk[l];
            if (l > 0 && (fk[l] > fk[l - 1] || tk[l] > tk[l - 1] || rk[l] > rk[l - 1]))
                throw std::runtime_error("each bucket level must have no more buckets than the one before");
        }
        K[1] = fk[0]; K[2] = tk[0]; K[3] = rk[0];
        double t0 = now_seconds();
        build_indexes();
        std::printf("Canonical boards: %zu flops, %zu turns, %zu rivers (%.0f s)\n", flop_idx.boards.size(),
                    turn_idx.boards.size(), river_idx.boards.size(), now_seconds() - t0);
        std::fflush(stdout);
        build_ochs_groups(o);
        build_river(o);
        std::printf("River: %d buckets (%.0f s)\n", K[3], now_seconds() - t0);
        std::fflush(stdout);
        build_turn(o);
        std::printf("Turn: %d buckets (%.0f s)\n", K[2], now_seconds() - t0);
        std::fflush(stdout);
        build_flop(o);
        std::printf("Flop: %d buckets (%.0f s)\n", K[1], now_seconds() - t0);
        std::fflush(stdout);
        compute_hash();
        loaded = true;
    }

    // Opponent groups: the 169 classes sorted by preflop equity against a random hand,
    // cut into 8 groups with equal numbers of combinations.
    void build_ochs_groups(const BuildOptions& o) {
        std::vector<double> eq(169, 0);
        int reps[169];
        for (int h = 0; h < NUM_HOLES; ++h) reps[holes().cls[h]] = h;
        parallel_for(169, o.threads, [&](int c, int) {
            Rng r(o.seed + 31 * c);
            int h = reps[c];
            int a = holes().c1[h], b = holes().c2[h];
            uint64_t used0 = holes().mask[h];
            double won = 0;
            const int N = 20000;
            for (int i = 0; i < N; ++i) {
                int x[7];
                random_board(r, 7, used0, x);
                int me[7] = {x[0], x[1], x[2], x[3], x[4], a, b};
                uint32_t v1 = eval_cards(me, 7), v2 = eval_cards(x, 7);
                won += v1 > v2 ? 1 : v1 == v2 ? 0.5 : 0;
            }
            eq[c] = won / N;
        });
        std::vector<int> order(169);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int x, int y) { return eq[x] < eq[y]; });
        std::vector<int> grp(169);
        int acc = 0;
        for (int c : order) {
            int combos = c < 13 ? 6 : c < 91 ? 4 : 12;
            grp[c] = std::min(OCHS - 1, (acc + combos / 2) * OCHS / NUM_HOLES);
            acc += combos;
        }
        ochs_group.assign(NUM_HOLES, 0);
        for (int h = 0; h < NUM_HOLES; ++h) ochs_group[h] = (uint8_t)grp[holes().cls[h]];
    }

    // OCHS features of every hole on a river board: out[h*OCHS + g]; valid[h] = false if it touches the board.
    void river_features(const int* b5, float* out, bool* valid) const {
        const Holes& H = holes();
        const Eval& E = evaltab();
        uint64_t bm = 0, bsm = 0;
        for (int i = 0; i < 5; ++i) { bm |= 1ull << b5[i]; bsm |= E.sbit[b5[i]]; }
        static thread_local std::vector<std::pair<uint32_t, int16_t>> v;
        v.clear();
        int tot[OCHS] = {0}, cnt[OCHS][52];
        std::memset(cnt, 0, sizeof(cnt));
        for (int h = 0; h < NUM_HOLES; ++h) {
            valid[h] = !(H.mask[h] & bm);
            if (!valid[h]) continue;
            v.push_back({eval_suitmask(bsm | E.sbit[H.c1[h]] | E.sbit[H.c2[h]]), (int16_t)h});
            int g = ochs_group[h];
            tot[g]++; cnt[g][H.c1[h]]++; cnt[g][H.c2[h]]++;
        }
        std::sort(v.begin(), v.end());
        int less_tot[OCHS] = {0}, less_card[OCHS][52];
        std::memset(less_card, 0, sizeof(less_card));
        size_t i = 0;
        while (i < v.size()) {
            size_t j = i;
            int eq_tot[OCHS] = {0}, eq_card[OCHS][52];
            std::memset(eq_card, 0, sizeof(eq_card));
            while (j < v.size() && v[j].first == v[i].first) {
                int h = v[j].second, g = ochs_group[h];
                eq_tot[g]++; eq_card[g][H.c1[h]]++; eq_card[g][H.c2[h]]++;
                ++j;
            }
            for (size_t k = i; k < j; ++k) {
                int h = v[k].second, a = H.c1[h], b = H.c2[h], gh = ochs_group[h];
                for (int g = 0; g < OCHS; ++g) {
                    int self = g == gh ? 1 : 0;
                    int less = less_tot[g] - less_card[g][a] - less_card[g][b];
                    int ties = eq_tot[g] - eq_card[g][a] - eq_card[g][b] + self;
                    int all = tot[g] - cnt[g][a] - cnt[g][b] + self;
                    out[(size_t)h * OCHS + g] = all > 0 ? (float)((less + 0.5 * ties) / all) : 0.5f;
                }
            }
            for (size_t k = i; k < j; ++k) {
                int h = v[k].second, g = ochs_group[h];
                less_tot[g]++; less_card[g][H.c1[h]]++; less_card[g][H.c2[h]]++;
            }
            i = j;
        }
    }

    static inline float l2sq(const float* a, const float* b, int d) {
        float s = 0;
        for (int i = 0; i < d; ++i) { float x = a[i] - b[i]; s += x * x; }
        return s;
    }
    static int nearest_dense(const float* p, const float* cent, int k, int dim, bool l1cdf) {
        int best = 0;
        float bd = 1e30f;
        for (int c = 0; c < k; ++c) {
            const float* q = cent + (size_t)c * dim;
            float s = 0;
            if (l1cdf) {
                for (int i = 0; i < dim; ++i) s += std::fabs(p[i] - q[i]);
            } else {
                for (int i = 0; i < dim; ++i) { float x = p[i] - q[i]; s += x * x; }
            }
            if (s < bd) { bd = s; best = c; }
        }
        return best;
    }

    // Weighted k-means++ and Lloyd iterations for dense points. l1cdf: L1 distance
    // (EMD between CDFs); otherwise squared Euclidean. Returns the assignment.
    static std::vector<int> kmeans_dense(const std::vector<float>& pts, const std::vector<float>* w, size_t n,
                                         int dim, int k, bool l1cdf, int threads, int max_iters, uint64_t seed,
                                         std::vector<float>& cent, const char* label) {
        Rng r(seed + 99);
        k = (int)std::min<size_t>(k, n);
        cent.assign((size_t)k * dim, 0);
        auto dist = [&](const float* a, const float* b) {
            float s = 0;
            if (l1cdf) for (int i = 0; i < dim; ++i) s += std::fabs(a[i] - b[i]);
            else for (int i = 0; i < dim; ++i) { float x = a[i] - b[i]; s += x * x; }
            return s;
        };
        auto W = [&](size_t i) { return w ? (*w)[i] : 1.f; };
        std::vector<float> d2(n, 1e30f);
        size_t first = r.below((uint32_t)n);
        std::copy(&pts[first * dim], &pts[first * dim] + dim, &cent[0]);
        int T = std::max(1, threads);
        size_t chunk = (n + T - 1) / T;
        for (int c = 1; c < k; ++c) {
            const float* last = &cent[(size_t)(c - 1) * dim];
            std::vector<double> part(T, 0);
            parallel_for(T, T, [&](int t, int) {
                size_t lo = t * chunk, hi = std::min(n, lo + chunk);
                double s = 0;
                for (size_t i = lo; i < hi; ++i) {
                    float d = dist(&pts[i * dim], last);
                    if (l1cdf) d = d * d;
                    d2[i] = std::min(d2[i], d);
                    s += d2[i] * W(i);
                }
                part[t] = s;
            });
            double tot = 0;
            for (double s : part) tot += s;
            double x = r.uniform() * tot;
            size_t pick = n - 1;
            for (size_t i = 0; i < n; ++i) { x -= d2[i] * W(i); if (x <= 0) { pick = i; break; } }
            std::copy(&pts[pick * dim], &pts[pick * dim] + dim, &cent[(size_t)c * dim]);
        }
        std::vector<int> assign(n, -1);
        for (int it = 0; it < max_iters; ++it) {
            std::atomic<size_t> changed(0);
            std::vector<double> cost(T, 0);
            parallel_for(T, T, [&](int t, int) {
                size_t lo = t * chunk, hi = std::min(n, lo + chunk), ch = 0;
                for (size_t i = lo; i < hi; ++i) {
                    int best = nearest_dense(&pts[i * dim], cent.data(), k, dim, l1cdf);
                    if (assign[i] != best) { assign[i] = best; ++ch; }
                    cost[t] += dist(&pts[i * dim], &cent[(size_t)best * dim]) * W(i);
                }
                changed += ch;
            });
            std::vector<double> sum((size_t)k * dim, 0);
            std::vector<double> cnt(k, 0);
            for (size_t i = 0; i < n; ++i) {
                cnt[assign[i]] += W(i);
                const float* p = &pts[i * dim];
                double* s = &sum[(size_t)assign[i] * dim];
                for (int j = 0; j < dim; ++j) s[j] += p[j] * W(i);
            }
            for (int c = 0; c < k; ++c) {
                if (cnt[c] <= 0) {   // empty cluster: restart at a random point
                    size_t p = r.below((uint32_t)n);
                    std::copy(&pts[p * dim], &pts[p * dim] + dim, &cent[(size_t)c * dim]);
                    continue;
                }
                for (int j = 0; j < dim; ++j) cent[(size_t)c * dim + j] = (float)(sum[(size_t)c * dim + j] / cnt[c]);
            }
            double tc = 0;
            for (double c : cost) tc += c;
            if (label) {
                std::printf("  %s k-means iteration %d: mean distance %.5f, %zu points moved\n", label, it + 1,
                            tc / n, (size_t)changed);
                std::fflush(stdout);
            }
            if (it > 0 && changed < n / 1000 + 1) break;
        }
        // Final assignment to the final centroids.
        parallel_for(T, T, [&](int t, int) {
            size_t lo = t * chunk, hi = std::min(n, lo + chunk);
            for (size_t i = lo; i < hi; ++i) assign[i] = nearest_dense(&pts[i * dim], cent.data(), k, dim, l1cdf);
        });
        return assign;
    }

    // Reorder centroids by strength (ascending); returns old -> new id.
    static std::vector<int> sort_by_strength(std::vector<float>& cent, int dim, const std::vector<double>& strength) {
        int k = (int)strength.size();
        std::vector<int> order(k);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return strength[a] < strength[b]; });
        std::vector<float> c2(cent.size());
        std::vector<int> remap(k);
        for (int i = 0; i < k; ++i) {
            remap[order[i]] = i;
            std::copy(&cent[(size_t)order[i] * dim], &cent[(size_t)order[i] * dim] + dim, &c2[(size_t)i * dim]);
        }
        cent.swap(c2);
        return remap;
    }

    // Merge level-0 buckets of one street into the coarser levels.
    // dist(a, b): distance between level-0 centroids a and a candidate mean b (dense, dim).
    void build_levels(int street, const std::vector<float>& cent, int dim, const std::vector<double>& mass,
                      const std::vector<double>& strength, bool l1cdf, const BuildOptions& o) {
        int k0 = K[street];
        for (size_t l = 1; l < levels.size(); ++l) {
            int k = levels[l].K[street];
            std::vector<int> m(k0);
            if (k >= k0) {
                std::iota(m.begin(), m.end(), 0);
            } else {
                std::vector<float> w(k0);
                for (int i = 0; i < k0; ++i) w[i] = (float)std::max(1e-9, mass[i]);
                std::vector<float> c;
                m = kmeans_dense(cent, &w, k0, dim, k, l1cdf, o.threads, 100, o.seed + 1000 * l + street, c, nullptr);
            }
            // Sort the merged buckets by mass-weighted strength.
            int kk = 0;
            for (int x : m) kk = std::max(kk, x + 1);
            std::vector<double> s(kk, 0), ww(kk, 0);
            for (int i = 0; i < k0; ++i) { s[m[i]] += strength[i] * mass[i]; ww[m[i]] += mass[i]; }
            for (int j = 0; j < kk; ++j) s[j] = ww[j] > 0 ? s[j] / ww[j] : 0;
            std::vector<int> order(kk), remap(kk);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return s[a] < s[b]; });
            for (int j = 0; j < kk; ++j) remap[order[j]] = j;
            levels[l].map[street].resize(k0);
            for (int i = 0; i < k0; ++i) levels[l].map[street][i] = (uint16_t)remap[m[i]];
            levels[l].K[street] = kk;
        }
    }

    // River ------------------------------------------------------------------------
    void build_river(const BuildOptions& o) {
        int per_board = 100;
        int nb = std::max(1, o.river_sample_points / per_board);
        std::vector<float> pts((size_t)nb * per_board * OCHS);
        parallel_for(nb, o.threads, [&](int i, int) {
            Rng r(o.seed + 7919ull * i);
            int b[5];
            random_board(r, 5, 0, b);
            std::vector<float> f((size_t)NUM_HOLES * OCHS);
            bool valid[NUM_HOLES];
            river_features(b, f.data(), valid);
            for (int j = 0; j < per_board; ++j) {
                int h;
                do { h = (int)r.below(NUM_HOLES); } while (!valid[h]);
                std::copy(&f[(size_t)h * OCHS], &f[(size_t)h * OCHS] + OCHS, &pts[((size_t)i * per_board + j) * OCHS]);
            }
        });
        size_t n = (size_t)nb * per_board;
        std::vector<int> as = kmeans_dense(pts, nullptr, n, OCHS, K[3], false, o.threads, o.max_iters, o.seed + 5,
                                           river_cent, "river");
        K[3] = levels[0].K[3] = (int)(river_cent.size() / OCHS);
        std::vector<double> strength(K[3], 0), mass(K[3], 0);
        for (int c = 0; c < K[3]; ++c)
            for (int g = 0; g < OCHS; ++g) strength[c] += river_cent[(size_t)c * OCHS + g] / OCHS;
        std::vector<int> remap = sort_by_strength(river_cent, OCHS, strength);
        std::sort(strength.begin(), strength.end());
        for (size_t i = 0; i < n; ++i) mass[remap[as[i]]] += 1;
        // Table of every canonical river board.
        size_t nboards = river_idx.boards.size();
        river_tab.assign(nboards * NUM_HOLES, 0);
        std::atomic<int> done(0);
        int kr = K[3];
        parallel_for((int)nboards, o.threads, [&](int i, int) {
            int b[5];
            river_idx.cards_of(i, b);
            std::vector<float> f((size_t)NUM_HOLES * OCHS);
            bool valid[NUM_HOLES];
            river_features(b, f.data(), valid);
            for (int h = 0; h < NUM_HOLES; ++h) {
                if (!valid[h]) continue;
                river_tab[(size_t)i * NUM_HOLES + h] =
                    (uint16_t)nearest_dense(&f[(size_t)h * OCHS], river_cent.data(), kr, OCHS, false);
            }
            int d = ++done;
            if (d % 20000 == 0) { std::printf("  river boards assigned: %d / %zu\n", d, nboards); std::fflush(stdout); }
        });
        build_levels(3, river_cent, OCHS, mass, strength, false, o);
    }

    // Turn -------------------------------------------------------------------------
    // CDF of river equity (vs a random hand) over TURN_BINS bins, for every hole.
    void turn_cdfs(const int* b4, std::vector<float>& cdf /* NUM_HOLES*TURN_BINS */, bool* valid) const {
        std::vector<uint8_t> hist((size_t)NUM_HOLES * TURN_BINS, 0);
        cdf.assign((size_t)NUM_HOLES * TURN_BINS, 0.f);
        uint64_t bm = 0;
        for (int i = 0; i < 4; ++i) bm |= 1ull << b4[i];
        int b5[5] = {b4[0], b4[1], b4[2], b4[3], 0};
        float eq[NUM_HOLES];
        for (int r = 0; r < 52; ++r) {
            if (bm & (1ull << r)) continue;
            b5[4] = r;
            river_equities(b5, eq);
            for (int h = 0; h < NUM_HOLES; ++h)
                if (eq[h] >= 0) hist[(size_t)h * TURN_BINS + std::min(TURN_BINS - 1, (int)(eq[h] * TURN_BINS))]++;
        }
        for (int h = 0; h < NUM_HOLES; ++h) {
            valid[h] = !(holes().mask[h] & bm);
            if (!valid[h]) continue;
            float acc = 0;
            for (int i = 0; i < TURN_BINS; ++i) {
                acc += hist[(size_t)h * TURN_BINS + i];
                cdf[(size_t)h * TURN_BINS + i] = acc / 46.f;
            }
        }
    }
    static double cdf_mean(const float* cdf) {   // mean equity from a CDF
        double s = 0;
        for (int i = 0; i < TURN_BINS; ++i) s += 1.0 - cdf[i];
        return s / TURN_BINS;
    }

    void build_turn(const BuildOptions& o) {
        int per_board = 100;
        int nb = std::max(1, o.turn_sample_points / per_board);
        std::vector<float> pts((size_t)nb * per_board * TURN_BINS);
        parallel_for(nb, o.threads, [&](int i, int) {
            Rng r(o.seed * 31 + 104729ull * i);
            int b[4];
            random_board(r, 4, 0, b);
            std::vector<float> cdf;
            bool valid[NUM_HOLES];
            turn_cdfs(b, cdf, valid);
            for (int j = 0; j < per_board; ++j) {
                int h;
                do { h = (int)r.below(NUM_HOLES); } while (!valid[h]);
                std::copy(&cdf[(size_t)h * TURN_BINS], &cdf[(size_t)h * TURN_BINS] + TURN_BINS,
                          &pts[((size_t)i * per_board + j) * TURN_BINS]);
            }
        });
        size_t n = (size_t)nb * per_board;
        std::vector<int> as = kmeans_dense(pts, nullptr, n, TURN_BINS, K[2], true, o.threads, o.max_iters, o.seed + 6,
                                           turn_cent, "turn");
        K[2] = levels[0].K[2] = (int)(turn_cent.size() / TURN_BINS);
        std::vector<double> strength(K[2]), mass(K[2], 0);
        for (int c = 0; c < K[2]; ++c) strength[c] = cdf_mean(&turn_cent[(size_t)c * TURN_BINS]);
        std::vector<int> remap = sort_by_strength(turn_cent, TURN_BINS, strength);
        std::sort(strength.begin(), strength.end());
        for (size_t i = 0; i < n; ++i) mass[remap[as[i]]] += 1;
        size_t nboards = turn_idx.boards.size();
        turn_tab.assign(nboards * NUM_HOLES, 0);
        std::atomic<int> done(0);
        int kt = K[2];
        parallel_for((int)nboards, o.threads, [&](int i, int) {
            int b[4];
            turn_idx.cards_of(i, b);
            std::vector<float> cdf;
            bool valid[NUM_HOLES];
            turn_cdfs(b, cdf, valid);
            for (int h = 0; h < NUM_HOLES; ++h) {
                if (!valid[h]) continue;
                turn_tab[(size_t)i * NUM_HOLES + h] =
                    (uint16_t)nearest_dense(&cdf[(size_t)h * TURN_BINS], turn_cent.data(), kt, TURN_BINS, true);
            }
            int d = ++done;
            if (d % 2000 == 0) { std::printf("  turn boards assigned: %d / %zu\n", d, nboards); std::fflush(stdout); }
        });
        build_levels(2, turn_cent, TURN_BINS, mass, strength, true, o);
        turn_strength = strength;
        turn_mass = mass;
    }
    std::vector<double> turn_strength, turn_mass;   // build only

    // Flop -------------------------------------------------------------------------
    struct Sparse { std::vector<std::pair<uint16_t, float>> e; };
    struct GroundSorted {
        int k = 0;
        std::vector<std::pair<float, uint16_t>> s;   // [k][k] sorted by distance
    };

    // Flop features: histogram over the feature-level turn buckets.
    void flop_histograms(const int* b3, std::vector<Sparse>& out) const {
        out.assign(NUM_HOLES, Sparse());
        uint64_t bm = (1ull << b3[0]) | (1ull << b3[1]) | (1ull << b3[2]);
        std::vector<std::vector<uint16_t>> lists(NUM_HOLES);
        int b4[4] = {b3[0], b3[1], b3[2], 0};
        int fl = flop_feature_level;
        for (int t = 0; t < 52; ++t) {
            if (bm & (1ull << t)) continue;
            b4[3] = t;
            int perm, id = turn_idx.lookup(b4, &perm);
            const uint16_t* tab = &turn_tab[(size_t)id * NUM_HOLES];
            const uint8_t* p = suitperms().p[perm];
            uint64_t m4 = bm | (1ull << t);
            for (int h = 0; h < NUM_HOLES; ++h) {
                if (holes().mask[h] & m4) continue;
                int b = tab[holes().index[apply_perm(holes().c1[h], p)][apply_perm(holes().c2[h], p)]];
                lists[h].push_back((uint16_t)at_level(fl, 2, b));
            }
        }
        for (int h = 0; h < NUM_HOLES; ++h) {
            if (holes().mask[h] & bm) continue;
            auto& l = lists[h];
            std::sort(l.begin(), l.end());
            float w = 1.f / (float)l.size();
            for (size_t i = 0; i < l.size();) {
                size_t j = i;
                while (j < l.size() && l[j] == l[i]) ++j;
                out[h].e.push_back({l[i], w * (float)(j - i)});
                i = j;
            }
        }
    }

    // Greedy transport: each point bin moves its mass to the nearest mean bins
    // that still have mass left. Stops once the cost exceeds `limit`.
    // taken: zeroed scratch array of size k; touched: scratch of size k.
    static float approx_emd(const Sparse& p, const float* mean, const GroundSorted& g, float* taken, int* touched,
                            float limit = 1e30f) {
        int k = g.k, nt = 0;
        float cost = 0;
        for (const auto& e : p.e) {
            float need = e.second;
            const std::pair<float, uint16_t>* row = &g.s[(size_t)e.first * k];
            for (int j = 0; j < k && need > 1e-9f; ++j) {
                int idx = row[j].second;
                float avail = mean[idx] - taken[idx];
                if (avail <= 1e-12f) continue;
                if (taken[idx] == 0) touched[nt++] = idx;
                float mv = avail < need ? avail : need;
                cost += mv * row[j].first;
                taken[idx] += mv;
                need -= mv;
            }
            if (cost >= limit) break;
        }
        for (int i = 0; i < nt; ++i) taken[touched[i]] = 0;
        return cost;
    }

    void build_flop(const BuildOptions& o) {
        // Feature level: the coarsest turn level with at least flop_feature_buckets buckets.
        flop_feature_level = 0;
        for (int l = 0; l < (int)levels.size(); ++l)
            if (levels[l].K[2] >= o.flop_feature_buckets) flop_feature_level = l;
        int kt = levels[flop_feature_level].K[2], kf = K[1];
        // Feature-level turn centroids (CDFs) and the ground distances between them.
        std::vector<double> cw(kt, 0);
        std::vector<float> tc((size_t)kt * TURN_BINS, 0);
        for (int b = 0; b < K[2]; ++b) {
            int m = at_level(flop_feature_level, 2, b);
            double w = std::max(1e-9, turn_mass[b]);
            cw[m] += w;
            for (int i = 0; i < TURN_BINS; ++i) tc[(size_t)m * TURN_BINS + i] += (float)(w * turn_cent[(size_t)b * TURN_BINS + i]);
        }
        for (int m = 0; m < kt; ++m)
            for (int i = 0; i < TURN_BINS; ++i) tc[(size_t)m * TURN_BINS + i] /= (float)std::max(1e-12, cw[m]);
        GroundSorted g;
        g.k = kt;
        g.s.resize((size_t)kt * kt);
        for (int a = 0; a < kt; ++a) {
            for (int b = 0; b < kt; ++b) {
                float d = 0;
                for (int i = 0; i < TURN_BINS; ++i) d += std::fabs(tc[(size_t)a * TURN_BINS + i] - tc[(size_t)b * TURN_BINS + i]);
                g.s[(size_t)a * kt + b] = {d / TURN_BINS, (uint16_t)b};
            }
            std::sort(g.s.begin() + (size_t)a * kt, g.s.begin() + (size_t)(a + 1) * kt);
        }
        std::vector<double> tstr(kt);
        for (int m = 0; m < kt; ++m) tstr[m] = cdf_mean(&tc[(size_t)m * TURN_BINS]);
        // Sample points.
        int per_board = 50;
        int nb = std::max(1, o.flop_sample_points / per_board);
        std::vector<Sparse> pts((size_t)nb * per_board);
        parallel_for(nb, o.threads, [&](int i, int) {
            Rng r(o.seed * 77 + 15485863ull * i);
            int b[3];
            random_board(r, 3, 0, b);
            std::vector<Sparse> hist;
            flop_histograms(b, hist);
            uint64_t bm = (1ull << b[0]) | (1ull << b[1]) | (1ull << b[2]);
            for (int j = 0; j < per_board; ++j) {
                int h;
                do { h = (int)r.below(NUM_HOLES); } while (holes().mask[h] & bm);
                pts[(size_t)i * per_board + j] = hist[h];
            }
        });
        size_t n = pts.size();
        kf = (int)std::min<size_t>(kf, n);
        int T = std::max(1, o.threads);
        std::vector<int> assign = kmeans_sparse(pts, nullptr, kf, g, T, o.max_iters, o.seed + 3, flop_cent, "flop");
        K[1] = levels[0].K[1] = kf;
        // Strength of each flop centroid: expected turn strength.
        std::vector<double> strength(kf, 0), mass(kf, 0);
        for (int c = 0; c < kf; ++c)
            for (int j = 0; j < kt; ++j) strength[c] += flop_cent[(size_t)c * kt + j] * tstr[j];
        std::vector<int> remap = sort_by_strength(flop_cent, kt, strength);
        std::sort(strength.begin(), strength.end());
        for (size_t i = 0; i < n; ++i) mass[remap[assign[i]]] += 1;
        // Assign every canonical flop.
        size_t nboards = flop_idx.boards.size();
        flop_tab.assign(nboards * NUM_HOLES, 0);
        std::atomic<int> done(0);
        parallel_for((int)nboards, o.threads, [&](int i, int) {
            int b[3];
            flop_idx.cards_of(i, b);
            std::vector<Sparse> hist;
            flop_histograms(b, hist);
            std::vector<float> taken(kt, 0.f);
            std::vector<int> touched(kt);
            for (int h = 0; h < NUM_HOLES; ++h) {
                if (hist[h].e.empty()) continue;
                int best = 0;
                float bd = 1e30f;
                for (int c = 0; c < kf; ++c) {
                    float d = approx_emd(hist[h], &flop_cent[(size_t)c * kt], g, taken.data(), touched.data(), bd);
                    if (d < bd) { bd = d; best = c; }
                }
                flop_tab[(size_t)i * NUM_HOLES + h] = (uint16_t)best;
            }
            int d = ++done;
            if (d % 200 == 0) { std::printf("  flop boards assigned: %d / %zu\n", d, nboards); std::fflush(stdout); }
        });
        // Coarser flop levels: merge level-0 flop centroids under the same distance.
        for (size_t l = 1; l < levels.size(); ++l) {
            int k = levels[l].K[1];
            std::vector<int> m(kf);
            if (k >= kf) {
                std::iota(m.begin(), m.end(), 0);
            } else {
                std::vector<Sparse> cp(kf);
                for (int c = 0; c < kf; ++c)
                    for (int j = 0; j < kt; ++j)
                        if (flop_cent[(size_t)c * kt + j] > 1e-7f) cp[c].e.push_back({(uint16_t)j, flop_cent[(size_t)c * kt + j]});
                std::vector<float> cc, w(kf);
                for (int c = 0; c < kf; ++c) w[c] = (float)std::max(1e-9, mass[c]);
                m = kmeans_sparse(cp, &w, k, g, T, 100, o.seed + 77 * l, cc, nullptr);
            }
            int kk = 0;
            for (int x : m) kk = std::max(kk, x + 1);
            std::vector<double> s(kk, 0), ww(kk, 0);
            for (int i = 0; i < kf; ++i) { s[m[i]] += strength[i] * mass[i]; ww[m[i]] += mass[i]; }
            for (int j = 0; j < kk; ++j) s[j] = ww[j] > 0 ? s[j] / ww[j] : 0;
            std::vector<int> order(kk), rm(kk);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return s[a] < s[b]; });
            for (int j = 0; j < kk; ++j) rm[order[j]] = j;
            levels[l].map[1].resize(kf);
            for (int i = 0; i < kf; ++i) levels[l].map[1][i] = (uint16_t)rm[m[i]];
            levels[l].K[1] = kk;
        }
    }

    // k-means++ and Lloyd iterations for sparse histogram points under approximate EMD.
    static std::vector<int> kmeans_sparse(const std::vector<Sparse>& pts, const std::vector<float>* w, int k,
                                          const GroundSorted& g, int T, int max_iters, uint64_t seed,
                                          std::vector<float>& cent, const char* label) {
        size_t n = pts.size();
        int kt = g.k;
        k = (int)std::min<size_t>(k, n);
        Rng r(seed);
        auto W = [&](size_t i) { return w ? (*w)[i] : 1.f; };
        cent.assign((size_t)k * kt, 0);
        auto set_cent = [&](int c, const Sparse& p) {
            float* m = &cent[(size_t)c * kt];
            std::fill(m, m + kt, 0.f);
            for (auto& e : p.e) m[e.first] += e.second;
        };
        set_cent(0, pts[r.below((uint32_t)n)]);
        std::vector<float> d2(n, 1e30f);
        size_t chunk = (n + T - 1) / T;
        for (int c = 1; c < k; ++c) {
            std::vector<double> part(T, 0);
            parallel_for(T, T, [&](int t, int) {
                std::vector<float> taken(kt, 0.f);
                std::vector<int> touched(kt);
                size_t lo = t * chunk, hi = std::min(n, lo + chunk);
                for (size_t i = lo; i < hi; ++i) {
                    float d = approx_emd(pts[i], &cent[(size_t)(c - 1) * kt], g, taken.data(), touched.data());
                    d2[i] = std::min(d2[i], d * d);
                    part[t] += d2[i] * W(i);
                }
            });
            double tot = 0;
            for (double s : part) tot += s;
            double x = r.uniform() * tot;
            size_t pick = n - 1;
            for (size_t i = 0; i < n; ++i) { x -= d2[i] * W(i); if (x <= 0) { pick = i; break; } }
            set_cent(c, pts[pick]);
        }
        std::vector<int> assign(n, -1);
        auto assign_all = [&](std::atomic<size_t>* changed, std::vector<double>* cost) {
            parallel_for(T, T, [&](int t, int) {
                std::vector<float> taken(kt, 0.f);
                std::vector<int> touched(kt);
                size_t lo = t * chunk, hi = std::min(n, lo + chunk), ch = 0;
                for (size_t i = lo; i < hi; ++i) {
                    int best = 0;
                    float bd = 1e30f;
                    if (assign[i] >= 0) {
                        best = assign[i];
                        bd = approx_emd(pts[i], &cent[(size_t)best * kt], g, taken.data(), touched.data());
                    }
                    for (int c = 0; c < k; ++c) {
                        float d = approx_emd(pts[i], &cent[(size_t)c * kt], g, taken.data(), touched.data(), bd);
                        if (d < bd) { bd = d; best = c; }
                    }
                    if (assign[i] != best) { assign[i] = best; ++ch; }
                    if (cost) (*cost)[t] += bd * W(i);
                }
                if (changed) *changed += ch;
            });
        };
        for (int it = 0; it < max_iters; ++it) {
            std::atomic<size_t> changed(0);
            std::vector<double> cost(T, 0);
            assign_all(&changed, &cost);
            std::vector<double> sum((size_t)k * kt, 0), cnt(k, 0);
            for (size_t i = 0; i < n; ++i) {
                cnt[assign[i]] += W(i);
                for (auto& e : pts[i].e) sum[(size_t)assign[i] * kt + e.first] += e.second * W(i);
            }
            for (int c = 0; c < k; ++c) {
                if (cnt[c] <= 0) { set_cent(c, pts[r.below((uint32_t)n)]); continue; }
                for (int j = 0; j < kt; ++j) cent[(size_t)c * kt + j] = (float)(sum[(size_t)c * kt + j] / cnt[c]);
            }
            double tc = 0;
            for (double c : cost) tc += c;
            if (label) {
                std::printf("  %s k-means iteration %d: mean EMD %.5f, %zu points moved\n", label, it + 1, tc / n,
                            (size_t)changed);
                std::fflush(stdout);
            }
            if (it > 0 && changed < n / 1000 + 1) break;
        }
        assign_all(nullptr, nullptr);
        return assign;
    }
};

}  // namespace pb
