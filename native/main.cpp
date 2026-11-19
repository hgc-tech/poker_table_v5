// pbcfr (v5): card abstraction builder, memory estimator, MCCFR trainer and
// exploitability estimate.
//
//   pbcfr abstraction --out DIR [--config FILE] [--threads T]
//   pbcfr memory      [--config FILE] [--players 2,3,4,5,6] [--stacks 10,20,40]
//   pbcfr train       --players N --stack BB|BB-BB-.. --out FILE [--config FILE] [--abstraction DIR]
//                     [--minutes M] [--iterations I] [--threads T] [--level L] [--export FILE]
//   pbcfr export      --in CHECKPOINT --out STRATEGY [--abstraction DIR]
//   pbcfr preflop     --in CHECKPOINT [--abstraction DIR]
//   pbcfr lbr         --in CHECKPOINT [--hands H] [--abstraction DIR]   (heads-up only)
//
// Settings come from the config file (see cfr_config.txt); command-line
// options override it.
#include "pb_train.h"

using namespace pb;

struct Args {
    std::map<std::string, std::string> kv;
    std::string cmd;
    Args(int argc, char** argv) {
        if (argc > 1) cmd = argv[1];
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) == 0) {
                std::string k = a.substr(2), v = "1";
                if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) v = argv[++i];
                kv[k] = v;
            }
        }
    }
    std::string get(const std::string& k, const std::string& d) const {
        auto it = kv.find(k);
        return it == kv.end() ? d : it->second;
    }
    double num(const std::string& k, double d) const {
        auto it = kv.find(k);
        return it == kv.end() ? d : std::atof(it->second.c_str());
    }
    bool has(const std::string& k) const { return kv.count(k) > 0; }
};

static std::vector<double> parse_list(const std::string& s) {
    std::vector<double> v;
    std::string t = s;
    for (char& c : t) if (c == ',') c = ' ';
    std::istringstream in(t);
    double x;
    while (in >> x) v.push_back(x);
    return v;
}

static KV load_config(const Args& a) {
    KV kv;
    std::string path = a.get("config", "cfr_config.txt");
    std::ifstream f(path);
    if (f) kv.load(path);
    else if (a.has("config")) throw std::runtime_error("cannot open " + path);
    return kv;
}

static int usage() {
    std::printf(
        "pbcfr (v5): poker blueprint trainer\n\n"
        "  pbcfr abstraction --out DIR [--config FILE] [--threads T]\n"
        "  pbcfr memory      [--config FILE] [--players 2,3,4,5,6] [--stacks 10,15,20]\n"
        "  pbcfr train       --players N --stack BB --out FILE [--config FILE] [--abstraction DIR]\n"
        "                    [--minutes M] [--iterations I] [--threads T] [--level L] [--export FILE]\n"
        "  pbcfr export      --in CHECKPOINT --out STRATEGY [--abstraction DIR]\n"
        "  pbcfr preflop     --in CHECKPOINT [--abstraction DIR]\n"
        "  pbcfr lbr         --in CHECKPOINT [--hands H] [--abstraction DIR]\n");
    return 1;
}

static int threads_of(const Args& a, const KV& kv) {
    int t = (int)a.num("threads", kv.num("threads", 0));
    if (t <= 0) t = (int)std::max(1u, std::thread::hardware_concurrency());
    return t;
}

static std::vector<int> ilist(const KV& kv, const char* k, std::vector<double> def) {
    std::vector<double> v = kv.list(k, def);
    std::vector<int> o;
    for (double x : v) o.push_back((int)x);
    return o;
}

int cmd_abstraction(const Args& a) {
    KV kv = load_config(a);
    std::string out = a.get("out", kv.str("abstraction_dir", "abstraction"));
    CardAbstraction::BuildOptions o;
    o.threads = threads_of(a, kv);
    o.river_sample_points = (int)a.num("river-sample", kv.num("river_sample_points", o.river_sample_points));
    o.turn_sample_points = (int)a.num("turn-sample", kv.num("turn_sample_points", o.turn_sample_points));
    o.flop_sample_points = (int)a.num("flop-sample", kv.num("flop_sample_points", o.flop_sample_points));
    o.flop_feature_buckets = (int)kv.num("flop_feature_buckets", o.flop_feature_buckets);
    o.max_iters = (int)a.num("kmeans-iterations", kv.num("kmeans_iterations", o.max_iters));
    std::vector<int> fk = ilist(kv, "flop_buckets", {1000, 500, 200}), tk = ilist(kv, "turn_buckets", {1000, 500, 200}),
                     rk = ilist(kv, "river_buckets", {1000, 500, 200});
    for (auto* v : {&fk, &tk, &rk})
        for (int x : *v) if (x < 1 || x > 65535) throw std::runtime_error("bucket counts must be 1..65535");
    std::printf("Building the v5 card abstraction with %d threads -> %s/\n", o.threads, out.c_str());
    for (size_t l = 0; l < fk.size(); ++l)
        std::printf("  level %zu: flop %d, turn %d, river %d buckets\n", l, fk[l], l < tk.size() ? tk[l] : 0,
                    l < rk.size() ? rk[l] : 0);
    std::fflush(stdout);
    CardAbstraction ab;
    ab.build(fk, tk, rk, o);
    std::string mk = "mkdir -p \"" + out + "\"";
    if (std::system(mk.c_str()) != 0) {}
    ab.save(out);
    std::printf("Saved %s/ (hash %llu)\n", out.c_str(), (unsigned long long)ab.hash);
    return 0;
}

// Bucket level for a blueprint: "bucket_level@NxS", "bucket_level@N" or "bucket_level";
// "auto" (default) = the finest level that fits in memory_gb.
static std::string level_setting(const KV& kv, int n, const Stacks& s) {
    std::ostringstream k1;
    k1 << "bucket_level@" << n << "x" << s.label();
    if (kv.has(k1.str())) return kv.str(k1.str(), "auto");
    return kv.str(kv.pkey("bucket_level", n), "auto");
}

int cmd_memory(const Args& a) {
    KV kv = load_config(a);
    std::vector<int> fk = ilist(kv, "flop_buckets", {1000, 500, 200}), tk = ilist(kv, "turn_buckets", {1000, 500, 200}),
                     rk = ilist(kv, "river_buckets", {1000, 500, 200});
    size_t nl = std::min(fk.size(), std::min(tk.size(), rk.size()));
    double mem = a.num("memory", kv.num("memory_gb", 22));
    std::vector<double> players = parse_list(a.get("players", "2,3,4,5,6"));
    std::vector<double> stacks = kv.list("stacks_bb", {10, 15, 20, 25, 30, 40});
    if (a.has("stacks")) stacks = parse_list(a.get("stacks", ""));
    std::printf("Bucket levels (flop/turn/river):");
    for (size_t l = 0; l < nl; ++l) std::printf("  L%zu = %d/%d/%d", l, fk[l], tk[l], rk[l]);
    std::printf("\nTraining memory per blueprint in GiB (2-byte regrets, preflop average only), if every\n"
                "infoset is reached. Training uses the finest level that fits in memory_gb = %.1f GiB.\n"
                "Exported strategies (used for play) are about half the training size.\n\n", mem);
    std::printf("%-8s %-12s %12s", "Players", "Stack", "Nodes");
    for (size_t l = 0; l < nl; ++l) std::printf("   L%zu GiB", l);
    std::printf("   Level used\n");
    std::set<std::string> skip;
    for (auto& s : kv.v["skip"]) skip.insert(s);
    for (double pn : players) {
        int n = (int)pn;
        GameConfig g;
        g.from_kv(kv, n);
        std::vector<Stacks> st;
        for (double s : a.has("stacks") ? stacks : kv.list_p("stacks_bb", n, stacks)) st.push_back(Stacks(s));
        if (!a.has("stacks"))
            for (auto& p : kv.v["profiles@" + std::to_string(n)]) {
                Stacks s = Stacks::parse(p);
                s.check(n);
                st.push_back(s);
            }
        for (const Stacks& s : st) {
            std::string nm = std::to_string(n) + "x" + s.label();
            if (skip.count(nm)) continue;
            Tree t;
            int K0[4] = {169, 1, 1, 1};
            t.build(n, s, g, K0, 0);
            std::printf("%-8d %-12s %12zu", n, s.label().c_str(), t.nodes.size());
            int pick = -1;
            for (size_t l = 0; l < nl; ++l) {
                int K[4] = {169, fk[l], tk[l], rk[l]};
                double gb = t.bytes_for(K) / 1073741824.0;
                std::printf(" %8.2f", gb);
                if (pick < 0 && gb <= mem) pick = (int)l;
            }
            std::string ls = level_setting(kv, n, s);
            if (ls != "auto") std::printf("   L%s (set in config)\n", ls.c_str());
            else if (pick >= 0) std::printf("   L%d\n", pick);
            else std::printf("   none fits: add a coarser level or fewer sizes\n");
            std::fflush(stdout);
        }
    }
    return 0;
}

static void load_abs(CardAbstraction& ab, const Args& a, const KV& kv) {
    std::string dir = a.get("abstraction", kv.str("abstraction_dir", "abstraction"));
    ab.load(dir);
}

int cmd_train(const Args& a) {
    KV kv = load_config(a);
    int n = (int)a.num("players", 0);
    if (n < 2 || n > 6 || !a.has("stack")) throw std::runtime_error("--players 2..6 and --stack BB are required");
    Stacks s = Stacks::parse(a.get("stack", ""));
    s.check(n);
    GameConfig g;
    g.from_kv(kv, n);
    CardAbstraction ab;
    load_abs(ab, a, kv);
    std::string out = a.get("out", "");
    if (out.empty()) throw std::runtime_error("--out FILE is required");
    Trainer tr;
    tr.opt.from_kv(kv);
    tr.opt.memory_gb = a.num("memory", tr.opt.memory_gb);
    tr.opt.threads = threads_of(a, kv);
    int level;
    std::string ls = a.get("level", level_setting(kv, n, s));
    std::vector<double> gbs;
    int autol = choose_level(n, s, g, ab, tr.opt.memory_gb, &gbs);
    if (ls == "auto") {
        level = autol;
        if (level < 0)
            throw std::runtime_error("no bucket level fits in memory_gb for this blueprint; add a coarser level, "
                                     "use fewer raise sizes, or raise memory_gb");
    } else {
        level = std::atoi(ls.c_str());
        if (level < 0 || level >= ab.nlevels()) throw std::runtime_error("bucket level out of range");
    }
    // An existing checkpoint keeps its level.
    try {
        std::string txt = Trainer::read_header(out);
        KV st;
        st.parse(txt);
        int cl = (int)st.num("level", level);
        if (cl != level && ls == "auto" && cl >= 0 && cl < ab.nlevels()) level = cl;
    } catch (...) {
    }
    tr.init(n, s, g, ab, level);
    if (tr.load(out))
        std::printf("Resuming %s from iteration %llu (%.2f h trained)\n", out.c_str(),
                    (unsigned long long)tr.iterations, tr.seconds / 3600);
    double minutes = a.num("minutes", 0);
    uint64_t iters = (uint64_t)a.num("iterations", 0);
    tr.train(minutes, iters, out, (uint64_t)a.num("seed", 1) + tr.iterations);
    if (a.has("export")) {
        tr.export_strategy(a.get("export", ""));
        std::printf("Exported strategy to %s\n", a.get("export", "").c_str());
    }
    return g_stop ? 130 : 0;   // 130: stopped by Ctrl-C
}

// Load a checkpoint with the settings stored in it.
static void load_checkpoint(Trainer& tr, CardAbstraction& ab, const std::string& in) {
    std::string txt = Trainer::read_header(in);
    KV stored;
    stored.parse(txt);
    GameConfig g;
    g.from_kv(stored);
    int level = (int)stored.num("level", 0);
    if (level < 0 || level >= ab.nlevels()) throw std::runtime_error(in + " uses a bucket level this abstraction lacks");
    // Reserve exactly what the checkpoint holds (plus slack).
    FILE* f = std::fopen(in.c_str(), "rb");
    uint32_t magic, len;
    uint64_t used = 0;
    if (f) {
        std::fread(&magic, 4, 1, f);
        std::fread(&len, 4, 1, f);
        std::fseek(f, (long)len + 8 + 8 + 8, SEEK_CUR);
        std::fread(&used, 8, 1, f);
        std::fclose(f);
    }
    tr.opt.memory_gb = used / 1073741824.0 + 0.01;
    tr.init((int)stored.num("players", 0), Stacks::from_kv(stored), g, ab, level);
    if (!tr.load(in)) throw std::runtime_error("cannot read " + in);
}

int cmd_export(const Args& a) {
    KV kv = load_config(a);
    CardAbstraction ab;
    load_abs(ab, a, kv);
    std::string in = a.get("in", ""), out = a.get("out", "");
    Trainer tr;
    load_checkpoint(tr, ab, in);
    tr.export_strategy(out);
    std::printf("Exported %s (%llu iterations) to %s\n", in.c_str(), (unsigned long long)tr.iterations, out.c_str());
    return 0;
}

// Print the first-in preflop strategy for every position (diagnostics).
int cmd_preflop(const Args& a) {
    KV kv = load_config(a);
    CardAbstraction ab;
    load_abs(ab, a, kv);
    Trainer tr;
    load_checkpoint(tr, ab, a.get("in", ""));
    const Tree& t = tr.tree;
    int idx = 0;
    std::printf("%d players, %.0f bb, %llu iterations, level %d\n", t.n, t.stack / 100.0,
                (unsigned long long)tr.iterations, t.level);
    while (t.nodes[idx].type == 0 && t.nodes[idx].street == 0) {
        const Node& nd = t.nodes[idx];
        int na = nd.nact;
        double freq[16] = {0}, combos = 0;
        for (int h = 0; h < NUM_HOLES; ++h) {
            float p[16];
            tr.play_strategy(nd, holes().cls[h], p);
            for (int x = 0; x < na; ++x) freq[x] += p[x];
            combos += 1;
        }
        std::printf("Position %d first in:", nd.player);
        for (int x = 0; x < na; ++x) std::printf("  code %d %.1f%%", t.codes[nd.child + x], 100 * freq[x] / combos);
        std::printf("\n");
        int fold = t.child_by_code(idx, A_FOLD);
        if (fold < 0) break;
        idx = fold;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Local best response (Lisy & Bowling 2017) against a heads-up blueprint.
// LBR knows its own cards and tracks the blueprint's range by Bayes' rule. At
// each decision it picks the action with the best expected chips, assuming the
// hand is checked down after a call: fold, call, or a bet/raise from the
// blueprint's own sizes (using the blueprint's real fold probability). The
// blueprint plays its strategy. LBR's average winnings are a lower bound on
// how much the blueprint can be exploited (in the real game).
struct LBR {
    const Trainer& tr;
    const CardAbstraction& ab;
    int samples = 300;
    explicit LBR(const Trainer& t, const CardAbstraction& a) : tr(t), ab(a) {}

    // Equity of `hole` vs a weighted range on a partial board (pot share).
    double equity(const int* hole, const int* board, int nb, const std::vector<double>& w, Rng& r) const {
        uint64_t used = holes().mask[holes().index[hole[0]][hole[1]]];
        for (int i = 0; i < nb; ++i) used |= 1ull << board[i];
        std::vector<double> cdf(NUM_HOLES);
        double acc = 0;
        for (int h = 0; h < NUM_HOLES; ++h) { acc += (holes().mask[h] & used) ? 0 : w[h]; cdf[h] = acc; }
        if (acc <= 0) return 0.5;
        if (nb == 5) {   // exact on the river
            int me[7] = {board[0], board[1], board[2], board[3], board[4], hole[0], hole[1]};
            uint32_t v = eval_cards(me, 7);
            double won = 0;
            for (int h = 0; h < NUM_HOLES; ++h) {
                if ((holes().mask[h] & used) || w[h] <= 0) continue;
                int op[7] = {board[0], board[1], board[2], board[3], board[4], holes().c1[h], holes().c2[h]};
                uint32_t u = eval_cards(op, 7);
                won += w[h] * (v > u ? 1 : v == u ? 0.5 : 0);
            }
            return won / acc;
        }
        double won = 0;
        int done = 0;
        for (int it = 0; it < samples; ++it) {
            double x = r.uniform() * acc;
            int h = (int)(std::lower_bound(cdf.begin(), cdf.end(), x) - cdf.begin());
            h = std::min(h, NUM_HOLES - 1);
            if (holes().mask[h] & used) continue;
            uint64_t u2 = used | holes().mask[h];
            int full[5];
            for (int i = 0; i < nb; ++i) full[i] = board[i];
            for (int i = nb; i < 5; ++i)
                for (;;) { int c = (int)r.below(52); if (!(u2 & (1ull << c))) { u2 |= 1ull << c; full[i] = c; break; } }
            int me[7] = {full[0], full[1], full[2], full[3], full[4], hole[0], hole[1]};
            int op[7] = {full[0], full[1], full[2], full[3], full[4], holes().c1[h], holes().c2[h]};
            uint32_t a = eval_cards(me, 7), b = eval_cards(op, 7);
            won += a > b ? 1 : a == b ? 0.5 : 0;
            done++;
        }
        return done ? won / done : 0.5;
    }

    // One hand; returns LBR's chip result in big blinds.
    double play(int L, Rng& r) const {
        const Tree& t = tr.tree;
        int O = 1 - L;
        uint64_t used = 0;
        auto draw = [&]() { for (;;) { int c = (int)r.below(52); if (!(used & (1ull << c))) { used |= 1ull << c; return c; } } };
        int hole[2][2], board[5];
        for (int i = 0; i < 2; ++i) { hole[i][0] = draw(); hole[i][1] = draw(); }
        for (int i = 0; i < 5; ++i) board[i] = draw();
        BetState s;
        s.init(2, t.stacks, t.sb(), t.bbc(), t.ante());
        std::vector<double> w(NUM_HOLES, 1.0);
        uint64_t lm = holes().mask[holes().index[hole[L][0]][hole[L][1]]];
        for (int h = 0; h < NUM_HOLES; ++h) if (holes().mask[h] & lm) w[h] = 0;
        int16_t bk[4][NUM_HOLES];
        bool have[4] = {false, false, false, false};
        auto buckets = [&](int street) -> const int16_t* {
            if (!have[street]) {
                int nb = street == 0 ? 0 : street + 2;
                ab.all_buckets(board, nb, bk[street]);
                for (int h = 0; h < NUM_HOLES; ++h) bk[street][h] = (int16_t)ab.at_level(t.level, street, bk[street][h]);
                have[street] = true;
            }
            return bk[street];
        };
        int idx = 0;
        std::vector<Act> acts;
        while (!s.done && t.nodes[idx].type == 0) {
            const Node& nd = t.nodes[idx];
            int q = s.to_act, na = nd.nact, street = s.street;
            int nb = street == 0 ? 0 : street + 2;
            abstract_actions(s, t.g, acts);
            const int16_t* B = buckets(street);
            for (int h = 0; h < NUM_HOLES; ++h) if (B[h] < 0) w[h] = 0;
            int pick = 0;
            if (q == O) {
                float p[16];
                tr.play_strategy(nd, B[holes().index[hole[O][0]][hole[O][1]]], p);
                double x = r.uniform();
                for (; pick < na - 1; ++pick) { x -= p[pick]; if (x <= 0) break; }
                for (int h = 0; h < NUM_HOLES; ++h) {
                    if (w[h] <= 0) continue;
                    tr.play_strategy(nd, B[h], p);
                    w[h] *= p[pick];
                }
            } else {
                double best = -1e18;
                double pot = (double)s.pot();
                for (int a = 0; a < na; ++a) {
                    uint8_t code = t.codes[nd.child + a];
                    double ev;
                    if (code == A_FOLD) {
                        ev = -(double)s.total_c[L];
                    } else if (code == A_CALL) {
                        double tc = (double)s.to_call(L);
                        double wp = equity(hole[L], board, nb, w, r);
                        ev = wp * (pot + tc) - ((double)s.total_c[L] + tc);
                    } else {
                        BetState c = s;
                        c.apply(2, acts[a].to);
                        int ci = nd.child + a;
                        double myc = (double)c.total_c[L], potc = (double)c.pot();
                        double fp = 0, tot = 0;
                        std::vector<double> w2 = w;
                        if (t.nodes[ci].type == 0 && t.nodes[ci].player == O) {
                            int fi = -1;
                            const Node& cn = t.nodes[ci];
                            for (int k = 0; k < cn.nact; ++k) if (t.codes[cn.child + k] == A_FOLD) fi = k;
                            const int16_t* CB = buckets(c.street);
                            for (int h = 0; h < NUM_HOLES; ++h) {
                                if (w[h] <= 0) continue;
                                float p[16];
                                tr.play_strategy(cn, CB[h], p);
                                double f = fi >= 0 ? p[fi] : 0;
                                fp += w[h] * f;
                                tot += w[h];
                                w2[h] = w[h] * (1 - f);
                            }
                            fp = tot > 0 ? fp / tot : 0;
                        }
                        // If called, the opponent matches our raise (or is all-in for less).
                        double oppc = std::min<double>((double)c.street_c[L], (double)(s.street_c[O] + s.stack[O]));
                        double pot_called = potc - (double)c.street_c[O] + oppc;
                        double wp = equity(hole[L], board, nb, w2, r);
                        double mine = myc - (double)c.street_c[L] + std::min<double>((double)c.street_c[L], oppc);
                        ev = fp * (potc - myc) + (1 - fp) * (wp * pot_called - mine);
                    }
                    if (ev > best + 1e-9) { best = ev; pick = a; }
                }
            }
            uint8_t code = t.codes[nd.child + pick];
            s.apply(code == A_FOLD ? 0 : code == A_CALL ? 1 : 2, acts[pick].to);
            idx = nd.child + pick;
        }
        // Settle in chips.
        double fin;
        if (s.live() < 2) {
            fin = (double)s.stack[L] + (s.folded[L] ? 0.0 : (double)s.pot());
        } else {
            int me[7] = {board[0], board[1], board[2], board[3], board[4], hole[L][0], hole[L][1]};
            int op[7] = {board[0], board[1], board[2], board[3], board[4], hole[O][0], hole[O][1]};
            uint32_t a = eval_cards(me, 7), b = eval_cards(op, 7);
            double cL = (double)s.total_c[L], cO = (double)s.total_c[O], m = std::min(cL, cO);
            double main = 2 * m, back = cL - m;   // uncalled chips return
            fin = (double)s.stack[L] + back + (a > b ? main + (cO - m) : a == b ? main / 2 : 0);
        }
        return (fin - (double)t.stacks[L]) / 100.0;
    }
};

int cmd_lbr(const Args& a) {
    KV kv = load_config(a);
    CardAbstraction ab;
    load_abs(ab, a, kv);
    Trainer tr;
    load_checkpoint(tr, ab, a.get("in", ""));
    if (tr.tree.n != 2) throw std::runtime_error("lbr works on heads-up (2-player) blueprints only");
    LBR lbr(tr, ab);
    lbr.samples = (int)a.num("samples", 300);
    long hands = (long)a.num("hands", 20000);
    int T = threads_of(a, kv);
    std::vector<double> sum(T, 0), sq(T, 0);
    std::vector<long> cnt(T, 0);
    std::atomic<long> next(0);
    double t0 = now_seconds();
    std::vector<std::thread> pool;
    for (int t = 0; t < T; ++t)
        pool.emplace_back([&, t]() {
            Rng r((uint64_t)a.num("seed", 7) * 1000 + t);
            for (;;) {
                long i = next.fetch_add(1);
                if (i >= hands) break;
                double v = lbr.play((int)(i & 1), r);
                sum[t] += v; sq[t] += v * v; cnt[t]++;
            }
        });
    for (auto& th : pool) th.join();
    double S = 0, Q = 0, N = 0;
    for (int t = 0; t < T; ++t) { S += sum[t]; Q += sq[t]; N += cnt[t]; }
    double mean = S / N, sd = std::sqrt(std::max(0.0, Q / N - mean * mean)), ci = 1.96 * sd / std::sqrt(N);
    std::printf("LBR vs %d players, %.0f bb (%llu iterations, level %d): %.0f hands in %.0f s\n", tr.tree.n,
                tr.tree.stack / 100.0, (unsigned long long)tr.iterations, tr.tree.level, N, now_seconds() - t0);
    std::printf("LBR wins %.1f +- %.1f bb per 100 hands (%.0f +- %.0f mbb/hand). Lower is better for the bot.\n",
                100 * mean, 100 * ci, 1000 * mean, 1000 * ci);
    std::printf("RESULT %.6f %.6f %.0f\n", mean, ci, N);
    return 0;
}

int main(int argc, char** argv) {
    Args a(argc, argv);
    try {
        if (a.cmd == "abstraction") return cmd_abstraction(a);
        if (a.cmd == "memory") return cmd_memory(a);
        if (a.cmd == "train") return cmd_train(a);
        if (a.cmd == "export") return cmd_export(a);
        if (a.cmd == "preflop") return cmd_preflop(a);
        if (a.cmd == "lbr") return cmd_lbr(a);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    return usage();
}
