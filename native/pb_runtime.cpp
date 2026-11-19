// Run-time library loaded by Python (ctypes): blueprint lookup, Bayesian
// range tracking, opponent modeling and depth-limited search.
//
// C interface (all strings are UTF-8, results stay valid until the next call
// on the same thread):
//   void* pb_open(const char* abstraction_dir)
//   int   pb_load(void* rt, const char* strategy_path)        -> index or -1
//   const char* pb_decide(void* rt, const char* request)      -> JSON
//   const char* pb_contexts(void* rt, const char* request)    -> JSON
//   const char* pb_info(void* rt)                             -> JSON
//   const char* pb_error(void)
//   void  pb_close(void* rt)
//
// Search (per decision):
//   1. Replay the hand. Every observed action is mapped onto the blueprint
//      tree (pseudo-harmonic translation for bet sizes), and each player's
//      range over the 1326 hole-card combinations is updated by Bayes' rule
//      with the strategy we believe that player uses.
//   2. Build a subgame from the current state to the end of the betting round
//      or `depth` further actions, whichever comes first. At each leaf every
//      player still holding chips picks one of four continuation strategies
//      (blueprint, or the blueprint biased toward folding, calling or
//      raising); the leaf value is a Monte Carlo rollout of those strategies.
//   3. Solve the subgame with Linear MCCFR (external sampling). Hands are
//      sampled from the ranges. Infosets use the exact hand (169 classes
//      preflop, 1326 combinations after the flop).
//   4. Restricted Nash response: each opponent is, with probability w, a fixed
//      player who uses our model of them (the blueprint tilted toward their
//      observed action frequencies) and otherwise a rational player whose
//      strategy is solved. w grows with the number of observed decisions, up
//      to rnr_max. The bot's strategy is a best response to this mixture,
//      which bounds how far it deviates from equilibrium.
#include <memory>
#include <mutex>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "pb_abstraction.h"
#include "pb_game.h"

#ifdef _WIN32
#define PB_API extern "C" __declspec(dllexport)
#else
#define PB_API extern "C" __attribute__((visibility("default")))
#endif

using namespace pb;

namespace {

constexpr int NCTX = 16;
const char* CTX_NAMES[NCTX] = {
    "preflop unopened", "preflop unopened", "preflop facing open", "preflop facing 3-bet+",
    "flop first to bet", "flop as preflop aggressor", "flop facing bet", "flop facing raise",
    "turn first to bet", "turn as flop aggressor", "turn facing bet", "turn facing raise",
    "river first to bet", "river as turn aggressor", "river facing bet", "river facing raise"};

inline int context_of(const BetState& s, int q) {
    int sit;
    if (s.raises == 0) sit = (s.street > 0 && s.prev_aggr == q) ? 1 : 0;
    else sit = s.raises == 1 ? 2 : 3;
    return s.street * 4 + sit;
}
inline int category(uint8_t code) { return code == A_FOLD ? 0 : code == A_CALL ? 1 : 2; }

thread_local std::string g_err;
thread_local std::string g_out;

struct Strategy {
    Tree tree;
    const uint8_t* map = nullptr;       // memory-mapped strategy file
    size_t maplen = 0;
    const uint32_t* table = nullptr;    // per chunk: data offset in 16-byte units
    const uint8_t* data = nullptr;
    int level = 0;
    uint64_t chunks = 0, data_bytes = 0;
    ~Strategy() { if (map) munmap((void*)map, maplen); }
    uint64_t iterations = 0;
    std::string path, name;
    double stack_bb = 0;                // largest starting stack
    Stacks prof;                        // starting stack per position
    bool base_ready = false;
    double base[NCTX][3];
    std::mutex mu, base_mu;
    bool ready = false;
    int players = 0;
    uint64_t slots = 0;
    long data_offset = 0;
    std::string header;

    inline void probs(int node, int bucket, float* out) const {
        const Node& nd = tree.nodes[node];
        uint32_t off = bucket >= 0 ? table[nd.offset + bucket / CHUNK] : 0xFFFFFFFFu;
        if (off == 0xFFFFFFFFu) { for (int a = 0; a < nd.nact; ++a) out[a] = 1.f / nd.nact; return; }
        const uint8_t* q = data + (size_t)off * 16 + (size_t)(bucket % CHUNK) * nd.nact;
        float s = 0;
        for (int a = 0; a < nd.nact; ++a) s += q[a];
        if (s <= 0) { for (int a = 0; a < nd.nact; ++a) out[a] = 1.f / nd.nact; return; }
        for (int a = 0; a < nd.nact; ++a) out[a] = q[a] / s;
    }
};

struct Runtime {
    CardAbstraction abs;
    std::vector<std::unique_ptr<Strategy>> bps;
};

// River buckets for a few hands on one board (cached per thread by board).
struct RiverCache {
    uint64_t board = ~0ull;
    uint32_t vals[NUM_HOLES];
    void prepare(const int* b5) {
        uint64_t bm = 0;
        for (int i = 0; i < 5; ++i) bm |= 1ull << b5[i];
        if (bm == board) return;
        board = bm;
        const Holes& H = holes();
        const Eval& E = evaltab();
        uint64_t bsm = 0;
        for (int i = 0; i < 5; ++i) bsm |= E.sbit[b5[i]];
        for (int h = 0; h < NUM_HOLES; ++h)
            vals[h] = (H.mask[h] & bm) ? 0xFFFFFFFFu : eval_suitmask(bsm | E.sbit[H.c1[h]] | E.sbit[H.c2[h]]);
    }
    float equity(int hidx) const {
        const Holes& H = holes();
        uint64_t hm = H.mask[hidx];
        uint32_t v = vals[hidx];
        int less = 0, tie = 0;
        for (int h = 0; h < NUM_HOLES; ++h) {
            if (vals[h] == 0xFFFFFFFFu || (H.mask[h] & hm)) continue;
            less += vals[h] < v;
            tie += vals[h] == v;
        }
        return (float)((less + 0.5 * tie) / 990.0);
    }
};
thread_local RiverCache t_river;

inline int bucket_of(const CardAbstraction& A, int street, const int* board, int hidx, int level) {
    const Holes& H = holes();
    return A.at_level(level, street, A.bucket(street, board, H.c1[hidx], H.c2[hidx]));
}

// Settle a finished hand and return each player's ICM equity x scale.
void settle(const BetState& s, const uint32_t* value, const std::vector<double>& payouts, double scale,
            float* util) {
    int n = s.n;
    double fin[MAX_PLAYERS], start[MAX_PLAYERS], eq[MAX_PLAYERS];
    for (int i = 0; i < n; ++i) { fin[i] = (double)s.stack[i]; start[i] = (double)s.start[i]; }
    if (s.live() < 2) {
        double pot = (double)s.pot();
        for (int i = 0; i < n; ++i) if (!s.folded[i]) fin[i] += pot;
    } else {
        int64_t lv[MAX_PLAYERS];
        int nl = 0;
        for (int i = 0; i < n; ++i) if (s.total_c[i] > 0) lv[nl++] = s.total_c[i];
        std::sort(lv, lv + nl);
        nl = (int)(std::unique(lv, lv + nl) - lv);
        int64_t prev = 0;
        for (int l = 0; l < nl; ++l) {
            double portion = 0;
            for (int i = 0; i < n; ++i)
                portion += (double)(std::min(s.total_c[i], lv[l]) - std::min(s.total_c[i], prev));
            uint32_t best = 0;
            int cnt = 0;
            for (int i = 0; i < n; ++i) {
                if (s.folded[i] || s.total_c[i] < lv[l]) continue;
                if (cnt == 0 || value[i] > best) { best = value[i]; cnt = 1; }
                else if (value[i] == best) cnt++;
            }
            if (cnt == 0) {
                int64_t mx = -1;
                for (int i = 0; i < n; ++i) if (!s.folded[i]) mx = std::max(mx, s.total_c[i]);
                for (int i = 0; i < n; ++i) if (!s.folded[i] && s.total_c[i] == mx) cnt++;
                for (int i = 0; i < n; ++i) if (!s.folded[i] && s.total_c[i] == mx) fin[i] += portion / cnt;
            } else {
                for (int i = 0; i < n; ++i)
                    if (!s.folded[i] && s.total_c[i] >= lv[l] && value[i] == best) fin[i] += portion / cnt;
            }
            prev = lv[l];
        }
    }
    icm(fin, start, n, payouts, eq);
    for (int i = 0; i < n; ++i) util[i] = (float)(eq[i] * scale);
}

// Child of blueprint node `b` for action `code`, or the closest one.
int nearest_child(const Tree& t, int b, uint8_t code) {
    if (b < 0 || t.nodes[b].type != 0) return -1;
    const Node& nd = t.nodes[b];
    int best = -1, bd = 1000;
    for (int a = 0; a < nd.nact; ++a) {
        uint8_t c = t.codes[nd.child + a];
        if (c == code) return nd.child + a;
        int d;
        if (code >= A_SIZE0 && c >= A_SIZE0) d = std::abs((int)c - (int)code);
        else if (code == A_FOLD && c == A_CALL) d = 50;
        else if (code >= A_SIZE0 && c == A_CALL) d = 100;
        else continue;
        if (d < bd) { bd = d; best = nd.child + a; }
    }
    return best;
}

// Blueprint child of node `b` for an action taken at real state s (any size menu):
// raises map onto the blueprint's sizes by the pseudo-harmonic rule.
int map_child(const Tree& t, int b, const BetState& s, const Act& act) {
    if (b < 0 || t.nodes[b].type != 0) return -1;
    if (act.code == A_FOLD || act.code == A_CALL) return nearest_child(t, b, act.code);
    std::vector<Act> bacts;
    abstract_actions(s, t.g, bacts);
    int i = map_raise(s, bacts, act.to);
    uint8_t code = bacts[i].code < A_SIZE0 ? (uint8_t)A_ALLIN : bacts[i].code;
    return nearest_child(t, b, code);
}

// Map blueprint probabilities onto the options available at state s (any size menu).
void map_probs(const Tree& t, int b, const BetState& s, const float* bp, const std::vector<Act>& real, float* out) {
    int na = (int)real.size();
    for (int i = 0; i < na; ++i) out[i] = 0;
    const Node& nd = t.nodes[b];
    std::vector<Act> bacts;
    abstract_actions(s, t.g, bacts);
    int allin = -1, call = -1, fold = -1, top = -1;
    for (int i = 0; i < na; ++i) {
        if (real[i].code == A_ALLIN) allin = i;
        else if (real[i].code == A_CALL) call = i;
        else if (real[i].code == A_FOLD) fold = i;
        else top = i;
    }
    for (int a = 0; a < nd.nact; ++a) {
        uint8_t c = t.codes[nd.child + a];
        int j = -1;
        if (c == A_FOLD) j = fold >= 0 ? fold : call;
        else if (c == A_CALL) j = call;
        else {
            double f = -1;
            for (auto& x : bacts) if (x.code == c) f = x.frac;
            if (c == A_ALLIN || f < 0) j = allin >= 0 ? allin : top;   // sizes snapped to all-in at this state
            else {
                double bd = 1e9;
                for (int i = 0; i < na; ++i) {
                    if (real[i].code < A_SIZE0 || real[i].code == A_ALLIN) continue;
                    double d = std::fabs(std::log1p(real[i].frac) - std::log1p(f));
                    if (d < bd) { bd = d; j = i; }
                }
                if (j < 0) j = allin >= 0 ? allin : call;
            }
        }
        if (j < 0) j = call >= 0 ? call : 0;
        out[j] += bp[a];
    }
    float tot = 0;
    for (int i = 0; i < na; ++i) tot += out[i];
    if (tot <= 0) { for (int i = 0; i < na; ++i) out[i] = 1.f / na; return; }
    for (int i = 0; i < na; ++i) out[i] /= tot;
}

// Opponent model for one position: tilt factors per context and category.
struct Model {
    double weight = 0;            // RNR weight on the fixed model
    float tilt[NCTX][3];
    double post[NCTX][3];
    double dev[NCTX];             // total-variation distance from the blueprint's frequencies
    double counts[NCTX][3];
    double total = 0;
    Model() {
        for (int c = 0; c < NCTX; ++c)
            for (int k = 0; k < 3; ++k) { tilt[c][k] = 1; post[c][k] = 0; counts[c][k] = 0; }
        for (int c = 0; c < NCTX; ++c) dev[c] = 0;
    }
};

// Our model of a player: the blueprint tilted toward that player's observed
// frequencies in this context. Multiplying by posterior/baseline ratios keeps
// the blueprint's hand ranking; the hand-independent part (weighted by how far
// the observed frequencies are from the blueprint's) lets the range widen
// beyond the blueprint's support, as it must for a player who jams any hand.
void apply_tilt(const Model& m, int ctx, const std::vector<Act>& acts, float* pr, double w) {
    if (w <= 0) return;
    int na = (int)acts.size();
    float tmp[16], s = 0;
    int ncat[3] = {0, 0, 0};
    for (int i = 0; i < na; ++i) ncat[category(acts[i].code)]++;
    double avail = 0;
    for (int k = 0; k < 3; ++k) if (ncat[k]) avail += m.post[ctx][k];
    for (int i = 0; i < na; ++i) { tmp[i] = pr[i] * m.tilt[ctx][category(acts[i].code)]; s += tmp[i]; }
    double kappa = std::min(1.0, m.dev[ctx]);
    for (int i = 0; i < na; ++i) {
        int c = category(acts[i].code);
        double flat = avail > 0 ? m.post[ctx][c] / avail / ncat[c] : 1.0 / na;
        double tilted = s > 0 ? tmp[i] / s : pr[i];
        tmp[i] = (float)((1 - kappa) * tilted + kappa * flat);
    }
    for (int i = 0; i < na; ++i) pr[i] = (float)((1 - w) * pr[i] + w * tmp[i]);
}

void apply_bias(int k, const std::vector<Act>& acts, float* pr) {
    if (k == 0) return;
    int na = (int)acts.size();
    float s = 0;
    for (int i = 0; i < na; ++i) {
        int cat = category(acts[i].code);
        if (cat == k - 1) pr[i] *= 5.f;
        s += pr[i];
    }
    if (s > 0) for (int i = 0; i < na; ++i) pr[i] /= s;
}

// Baseline action frequencies per context, from blueprint self-play.
void ensure_loaded(Strategy& st);

void compute_baseline(Runtime& rt, Strategy& S) {
    ensure_loaded(S);
    std::lock_guard<std::mutex> lk(S.base_mu);
    if (S.base_ready) return;
    double cnt[NCTX][3];
    for (int c = 0; c < NCTX; ++c) for (int k = 0; k < 3; ++k) cnt[c][k] = 1e-3;
    const Tree& t = S.tree;
    Rng r(4242);
    std::vector<Act> acts;
    int n = t.n;
    for (int hand = 0; hand < 30000; ++hand) {
        uint64_t used = 0;
        auto draw = [&]() { for (;;) { int c = (int)r.below(52); if (!(used & (1ull << c))) { used |= 1ull << c; return c; } } };
        int hidx[MAX_PLAYERS], board[5];
        for (int i = 0; i < n; ++i) { int a = draw(), b = draw(); hidx[i] = holes().index[a][b]; }
        for (int i = 0; i < 5; ++i) board[i] = draw();
        BetState s;
        s.init(n, t.stacks, t.sb(), t.bbc(), t.ante());
        int node = 0;
        while (!s.done && node >= 0 && t.nodes[node].type == 0) {
            int q = s.to_act;
            int bk = bucket_of(rt.abs, s.street, board, hidx[q], S.level);
            float pr[16];
            S.probs(node, bk, pr);
            double x = r.uniform();
            int a = 0;
            const Node& nd = t.nodes[node];
            for (; a < nd.nact - 1; ++a) { x -= pr[a]; if (x <= 0) break; }
            uint8_t code = t.codes[nd.child + a];
            cnt[context_of(s, q)][category(code)] += 1;
            abstract_actions(s, t.g, acts);
            int64_t to = 0;
            for (auto& ac : acts) if (ac.code == code) to = ac.to;
            s.apply(code == A_FOLD ? 0 : code == A_CALL ? 1 : 2, to);
            node = nd.child + a;
        }
    }
    for (int c = 0; c < NCTX; ++c) {
        double tot = cnt[c][0] + cnt[c][1] + cnt[c][2];
        for (int k = 0; k < 3; ++k) S.base[c][k] = cnt[c][k] / tot;
    }
    S.base_ready = true;
}

// ---------------------------------------------------------------------------
// Request parsing.
struct Request {
    int n = 0, hero = -1;
    int64_t stacks[MAX_PLAYERS] = {0};
    int64_t sb = 0, bb = 0, ante = 0;
    int hole[2] = {-1, -1};
    std::vector<int> board;
    struct RA { int pos, kind; int64_t to; };
    std::vector<RA> actions;
    std::vector<double> payouts;
    std::string mode = "search";
    int blueprint = -1;
    double search_ms = 1000, rnr_max = 0.75, rnr_n0 = 60, prior = 20;
    double max_jam_spr = 4, jam_free_bb = 30;   // see jam_allowed()
    int iters = 0, threads = 1, depth = 6, equity_samples = 3000;
    uint64_t seed = 1;
    int search_preflop = 2;      // 0 never, 1 always, 2 only after a bet size far from the blueprint's
    // Search size menus (pot fractions); empty = the blueprint's sizes.
    std::vector<double> m_bet, m_raise, m_inner_bet, m_inner_raise, m_pre_open, m_pre_raise;
    double counts[MAX_PLAYERS][NCTX][3];
    Request() { std::memset(counts, 0, sizeof(counts)); }

    void parse(const char* txt) {
        std::istringstream in(txt);
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string k;
            if (!(ls >> k)) continue;
            if (k == "players") ls >> n;
            else if (k == "stacks") { for (int i = 0; i < n; ++i) ls >> stacks[i]; }
            else if (k == "blinds") ls >> sb >> bb >> ante;
            else if (k == "hero") ls >> hero;
            else if (k == "hole") ls >> hole[0] >> hole[1];
            else if (k == "board") { int c; while (ls >> c) board.push_back(c); }
            else if (k == "actions") { RA a; while (ls >> a.pos >> a.kind >> a.to) actions.push_back(a); }
            else if (k == "payouts") { double x; while (ls >> x) payouts.push_back(x); }
            else if (k == "mode") ls >> mode;
            else if (k == "blueprint") ls >> blueprint;
            else if (k == "search_ms") ls >> search_ms;
            else if (k == "iterations") ls >> iters;
            else if (k == "threads") ls >> threads;
            else if (k == "depth") ls >> depth;
            else if (k == "seed") ls >> seed;
            else if (k == "rnr_max") ls >> rnr_max;
            else if (k == "max_jam_spr") ls >> max_jam_spr;
            else if (k == "jam_free_bb") ls >> jam_free_bb;
            else if (k == "rnr_n0") ls >> rnr_n0;
            else if (k == "prior") ls >> prior;
            else if (k == "equity_samples") ls >> equity_samples;
            else if (k == "search_preflop") { ls >> search_preflop; }
            else if (k == "search_bet" || k == "search_raise" || k == "search_inner_bet" || k == "search_inner_raise" ||
                     k == "search_preflop_open" || k == "search_preflop_raise") {
                std::vector<double>& v = k == "search_bet" ? m_bet : k == "search_raise" ? m_raise
                                       : k == "search_inner_bet" ? m_inner_bet : k == "search_inner_raise" ? m_inner_raise
                                       : k == "search_preflop_open" ? m_pre_open : m_pre_raise;
                v.clear();
                double x;
                while (ls >> x) v.push_back(x);
            }
            else if (k == "model") {
                int p, c;
                double f, ps, ag;
                ls >> p >> c >> f >> ps >> ag;
                if (p >= 0 && p < MAX_PLAYERS && c >= 0 && c < NCTX) {
                    counts[p][c][0] = f; counts[p][c][1] = ps; counts[p][c][2] = ag;
                }
            }
        }
        if (n < 2 || n > MAX_PLAYERS) throw std::runtime_error("players must be 2..6");
        if (bb <= 0) throw std::runtime_error("missing blinds");
    }
};

int choose_blueprint(Runtime& rt, const Request& q) {
    if (q.blueprint >= 0 && q.blueprint < (int)rt.bps.size()) return q.blueprint;
    int hero = q.hero >= 0 ? q.hero : 0;
    // Distance to a blueprint: the hero's effective stack (own stack capped by the largest
    // other one), plus half the average distance of the other players' stacks, which set the
    // ICM pressure. With equal-stack blueprints only, the hero's effective stack dominates.
    Stacks real;
    real.k = q.n;
    for (int i = 0; i < q.n; ++i) real.bb[i] = std::max(1.0, (double)q.stacks[i] / (double)q.bb);
    double eff[MAX_PLAYERS], beff[MAX_PLAYERS];
    real.effective(q.n, eff);
    int best = -1;
    double bd = 1e9;
    for (size_t i = 0; i < rt.bps.size(); ++i) {
        const Strategy& s = *rt.bps[i];
        double d = s.players == q.n ? 0 : 100 + std::abs(s.players - q.n);
        if (s.players == q.n) {
            s.prof.effective(q.n, beff);
            d += std::fabs(std::log(beff[hero] / eff[hero]));
            double o = 0;
            for (int j = 0; j < q.n; ++j) if (j != hero) o += std::fabs(std::log(s.prof.at(j) / real.bb[j]));
            d += 0.5 * o / (q.n - 1);
        } else {
            d += std::fabs(std::log(s.stack_bb / eff[hero]));
        }
        if (d < bd) { bd = d; best = (int)i; }
    }
    if (best < 0) throw std::runtime_error("no blueprint loaded");
    if (rt.bps[best]->players != q.n) throw std::runtime_error("no blueprint for " + std::to_string(q.n) + " players");
    return best;
}

// ---------------------------------------------------------------------------
// Hand replay with range tracking.
struct Replay {
    BetState s;
    int node = 0;                       // blueprint node
    bool on_tree = true;
    bool off_size = false;              // some bet was far from the blueprint size it was mapped to
    std::vector<double> range[MAX_PLAYERS];   // weights over 1326 combos
    int16_t buckets[4][NUM_HOLES];
    bool have_buckets[4] = {false, false, false, false};
    std::vector<int> board;
    int level = 0;
    struct Obs { int pos, ctx, cat; };
    std::vector<Obs> obs;
};

double model_weight(const Request& q, int pos) {
    double tot = 0;
    for (int c = 0; c < NCTX; ++c) for (int k = 0; k < 3; ++k) tot += q.counts[pos][c][k];
    return q.rnr_max * tot / (tot + q.rnr_n0);
}

void build_models(const Request& q, const Strategy& S, Model* models) {
    for (int p = 0; p < q.n; ++p) {
        Model& m = models[p];
        m.total = 0;
        for (int c = 0; c < NCTX; ++c) {
            double n = q.counts[p][c][0] + q.counts[p][c][1] + q.counts[p][c][2];
            m.total += n;
            for (int k = 0; k < 3; ++k) {
                m.counts[c][k] = q.counts[p][c][k];
                double b = S.base[c][k];
                m.post[c][k] = (q.prior * b + q.counts[p][c][k]) / (q.prior + n);
                double r = b > 1e-4 ? m.post[c][k] / b : 1.0;
                m.tilt[c][k] = (float)std::min(5.0, std::max(0.2, r));
            }
            m.dev[c] = 0;
            for (int k = 0; k < 3; ++k) m.dev[c] += 0.5 * std::fabs(m.post[c][k] - S.base[c][k]);
        }
        m.weight = p == q.hero ? 0.0 : model_weight(q, p);
    }
}

void ensure_buckets(Runtime& rt, Replay& R, int street) {
    if (R.have_buckets[street]) return;
    int nb = street == 0 ? 0 : street + 2;
    rt.abs.all_buckets(R.board.data(), nb, R.buckets[street]);
    for (int h = 0; h < NUM_HOLES; ++h) R.buckets[street][h] = (int16_t)rt.abs.at_level(R.level, street, R.buckets[street][h]);
    R.have_buckets[street] = true;
}

// Apply one observed action; update the actor's range.
void replay_action(Runtime& rt, const Strategy& S, const Model* models, Replay& R, int kind, int64_t to,
                   bool track_ranges) {
    BetState& s = R.s;
    const Tree& t = S.tree;
    int q = s.to_act;
    std::vector<Act> acts;
    abstract_actions(s, t.g, acts);
    uint8_t code;
    if (kind == 0) code = s.to_call(q) > 0 ? A_FOLD : A_CALL;
    else if (kind == 1 || !s.can_raise(q)) code = A_CALL;
    else {
        int64_t maxto = s.max_raise_to(q);
        to = std::min(std::max(to, s.min_raise_to(q)), maxto);
        if (to >= maxto) code = A_ALLIN;
        else {
            // Pseudo-harmonic mapping between the two nearest sizes (as pot fractions).
            int mapped = map_raise(s, acts, to);
            code = acts[mapped].code;
            if (code < A_SIZE0) code = A_ALLIN;   // no raise in the blueprint here: closest is all-in
            int64_t mto = acts[mapped].to;
            // Off the blueprint's sizes: more than one big blind and 15% away from the mapped size.
            if (std::llabs(to - mto) > std::max<int64_t>(s.bb, (int64_t)(0.15 * (double)mto))) R.off_size = true;
        }
    }
    int ctx = context_of(s, q);
    R.obs.push_back({q, ctx, category(code)});
    if (R.on_tree && t.nodes[R.node].type == 0) {
        int child = nearest_child(t, R.node, code);
        if (track_ranges) {
            ensure_buckets(rt, R, s.street);
            const Node& nd = t.nodes[R.node];
            int ai = child >= 0 ? child - nd.child : -1;
            std::vector<Act> bacts(nd.nact);
            for (int a = 0; a < nd.nact; ++a) bacts[a] = {t.codes[nd.child + a], 0};
            if (ai >= 0) {
                double w = models[q].weight;
                for (int h = 0; h < NUM_HOLES; ++h) {
                    if (R.range[q][h] <= 0) continue;
                    int b = R.buckets[s.street][h];
                    if (b < 0) { R.range[q][h] = 0; continue; }
                    float pr[16];
                    S.probs(R.node, b, pr);
                    apply_tilt(models[q], ctx, bacts, pr, w);
                    R.range[q][h] *= (pr[ai] + 1e-3);
                }
                double sum = 0;
                for (int h = 0; h < NUM_HOLES; ++h) sum += R.range[q][h];
                if (sum > 0) for (int h = 0; h < NUM_HOLES; ++h) R.range[q][h] /= sum;
            }
        }
        if (child < 0) R.on_tree = false;
        else R.node = child;
    }
    int street_before = s.street;
    s.apply(kind == 0 ? 0 : (kind == 1 || !s.can_raise(q)) ? 1 : 2, to);
    (void)street_before;
}

void start_replay(Runtime& rt, const Strategy& S, const Request& q, const Model* models, Replay& R,
                  bool track_ranges) {
    R.level = S.level;
    R.s.init(q.n, q.stacks, q.sb, q.bb, q.ante);
    R.node = 0;
    R.on_tree = true;
    R.board = q.board;
    uint64_t bm = 0;
    for (int c : q.board) bm |= 1ull << c;
    for (int p = 0; p < q.n; ++p) {
        R.range[p].assign(NUM_HOLES, 0);
        for (int h = 0; h < NUM_HOLES; ++h) R.range[p][h] = (holes().mask[h] & bm) ? 0 : 1;
    }
    for (auto& a : q.actions) {
        if (R.s.done) break;
        if (a.pos != R.s.to_act)
            throw std::runtime_error("action by position " + std::to_string(a.pos) + " but position " +
                                     std::to_string(R.s.to_act) + " is to act");
        replay_action(rt, S, models, R, a.kind, a.to, track_ranges);
    }
}

// ---------------------------------------------------------------------------
// Subgame.
struct SNode {
    BetState s;
    int bnode = -1;
    int type = 0;           // 0 decision, 1 terminal, 2 leaf
    int child = -1, nact = 0;
    std::vector<Act> acts;
    size_t roff = 0;        // regret offset (decision) or choice offset (leaf)
    int ctx = 0;
};

struct Search {
    Runtime* rt;
    const Strategy* S;
    const Request* q;
    const Model* models;
    std::vector<SNode> nodes;
    std::vector<float> R;            // regrets
    int H = NUM_HOLES;               // infoset hand index size
    int street = 0;
    std::vector<int> board;          // known board
    const std::vector<double>* ranges;
    int hero = 0, hero_h = 0;
    std::vector<double> root_acc;
    double scale = 1;
    std::vector<double> alias_p[MAX_PLAYERS];
    std::vector<int> alias_a[MAX_PLAYERS];
    const int16_t* cur_buckets = nullptr;
    std::atomic<long> iter_count{0};
    std::mutex acc_mu;
    int depth_used = 0;
    bool drop_root_allin = false;

    // Row index of a hand for player `pl`. The hero keeps its exact hand; opponents share rows by
    // blueprint bucket, so a short search learns their responses from far more samples.
    int hidx(int h, int pl) const {
        if (street == 0) return holes().cls[h];
        if (pl != hero && cur_buckets && cur_buckets[h] >= 0) return cur_buckets[h];
        return h;
    }

    // Search size menu: the full menu at the root, a shorter one deeper in the subgame.
    void search_actions(const BetState& s, bool root, std::vector<Act>& out) const {
        const GameConfig& g = S->tree.g;
        const std::vector<double>* m = nullptr;
        if (s.street == 0) {
            if (root) m = s.raises == 0 ? &q->m_pre_open : &q->m_pre_raise;
        } else if (root) {
            m = s.raises == 0 ? &q->m_bet : &q->m_raise;
        } else if (s.live() < 3) {
            m = s.raises == 0 ? &q->m_inner_bet : &q->m_inner_raise;
        }
        if (!m || m->empty()) { abstract_actions(s, g, out); return; }
        int cap = raise_cap(s, g) + (root ? 1 : 0);
        menu_actions(s, *m, cap, g.snap, out);
    }
    void build(const BetState& s0, int b0, int depth, size_t max_slots) {
        for (;;) {
            nodes.clear();
            size_t slots = 0;
            nodes.push_back(SNode());
            build_rec(0, s0, b0, depth, slots);
            if (slots <= max_slots || depth <= 1) { R.assign(slots, 0.f); depth_used = depth; return; }
            depth--;
        }
    }
    void build_rec(int idx, const BetState& s, int b, int depth, size_t& slots) {
        SNode& nd = nodes[idx];
        nd.s = s;
        nd.bnode = b;
        if (s.done) { nd.type = 1; return; }
        if ((s.street != street || depth <= 0) && idx != 0) {
            nd.type = 2;
            nd.roff = slots;
            slots += (size_t)MAX_PLAYERS * H * 4;
            return;
        }
        nd.type = 0;
        search_actions(s, idx == 0, nd.acts);
        if (idx == 0 && drop_root_allin && nd.acts.size() > 1 && nd.acts.back().code == A_ALLIN) nd.acts.pop_back();
        nd.nact = (int)nd.acts.size();
        nd.ctx = context_of(s, s.to_act);
        nd.roff = slots;
        slots += (size_t)H * nd.nact;
        int first = (int)nodes.size();
        nd.child = first;
        std::vector<Act> acts = nd.acts;
        for (size_t a = 0; a < acts.size(); ++a) nodes.push_back(SNode());
        for (size_t a = 0; a < acts.size(); ++a) {
            BetState c = s;
            c.apply(acts[a].code == A_FOLD ? 0 : acts[a].code == A_CALL ? 1 : 2, acts[a].to);
            int cb = map_child(S->tree, b, s, acts[a]);
            build_rec(first + (int)a, c, cb, depth - 1, slots);
        }
    }

    void build_alias() {
        for (int p = 0; p < q->n; ++p) {
            const std::vector<double>& w = ranges[p];
            double sum = 0;
            for (double x : w) sum += x;
            alias_p[p].assign(NUM_HOLES, 0);
            alias_a[p].assign(NUM_HOLES, 0);
            std::vector<double> sc(NUM_HOLES);
            std::vector<int> small, large;
            for (int h = 0; h < NUM_HOLES; ++h) {
                sc[h] = sum > 0 ? w[h] * NUM_HOLES / sum : 1.0;
                (sc[h] < 1 ? small : large).push_back(h);
            }
            while (!small.empty() && !large.empty()) {
                int l = small.back(); small.pop_back();
                int g = large.back();
                alias_p[p][l] = sc[l];
                alias_a[p][l] = g;
                sc[g] -= 1 - sc[l];
                if (sc[g] < 1) { large.pop_back(); small.push_back(g); }
            }
            for (int h : large) alias_p[p][h] = 1;
            for (int h : small) alias_p[p][h] = 1;
        }
    }
    int sample_hand(int p, Rng& r) const {
        int i = (int)r.below(NUM_HOLES);
        return r.uniform() < alias_p[p][i] ? i : alias_a[p][i];
    }

    struct Sample {
        int hand[MAX_PLAYERS];
        bool fixed[MAX_PLAYERS];
        int full[5];
        uint32_t value[MAX_PLAYERS];
        bool values = false;
    };

    bool draw_sample(Sample& sm, int traverser, Rng& r) const {
        const BetState& s0 = nodes[0].s;
        uint64_t used = 0;
        for (int c : board) used |= 1ull << c;
        int order[MAX_PLAYERS], k = 0;
        order[k++] = hero;
        for (int p = 0; p < q->n; ++p) if (p != hero) order[k++] = p;
        for (int i = 0; i < k; ++i) {
            int p = order[i];
            if (s0.folded[p]) { sm.hand[p] = -1; continue; }
            int h = -1;
            if (p == hero && traverser == hero && r.uniform() < 0.5) h = hero_h;
            else {
                for (int tries = 0; tries < 200; ++tries) {
                    int c = sample_hand(p, r);
                    if (!(holes().mask[c] & used) && ranges[p][c] > 0) { h = c; break; }
                }
                if (h < 0) {   // fall back to a uniform legal hand
                    for (int tries = 0; tries < 10000 && h < 0; ++tries) {
                        int c = (int)r.below(NUM_HOLES);
                        if (!(holes().mask[c] & used)) h = c;
                    }
                }
            }
            if (h < 0 || (holes().mask[h] & used)) return false;
            sm.hand[p] = h;
            used |= holes().mask[h];
        }
        int nb = (int)board.size();
        for (int i = 0; i < nb; ++i) sm.full[i] = board[i];
        for (int i = nb; i < 5; ++i) {
            for (;;) {
                int c = (int)r.below(52);
                if (!(used & (1ull << c))) { used |= 1ull << c; sm.full[i] = c; break; }
            }
        }
        for (int p = 0; p < q->n; ++p) {
            sm.fixed[p] = false;
            if (p != hero && p != traverser && sm.hand[p] >= 0) sm.fixed[p] = r.uniform() < models[p].weight;
        }
        sm.values = false;
        return true;
    }
    void ensure_values(Sample& sm) const {
        if (sm.values) return;
        for (int p = 0; p < q->n; ++p) {
            if (sm.hand[p] < 0) { sm.value[p] = 0; continue; }
            int c[7] = {sm.full[0], sm.full[1], sm.full[2], sm.full[3], sm.full[4], holes().c1[sm.hand[p]],
                        holes().c2[sm.hand[p]]};
            sm.value[p] = eval_cards(c, 7);
        }
        sm.values = true;
    }

    // Blueprint strategy at a subgame decision; leaves `pr` unchanged if there is none.
    void blueprint_probs(const SNode& nd, int p, int hand, float* pr) const {
        if (nd.bnode < 0 || S->tree.nodes[nd.bnode].type != 0 || S->tree.nodes[nd.bnode].player != p) return;
        int b = cur_buckets[hand];
        if (b < 0) return;
        float bp[16];
        S->probs(nd.bnode, b, bp);
        map_probs(S->tree, nd.bnode, nd.s, bp, nd.acts, pr);
    }

    // Strategy of a fixed (modeled) player at a subgame decision.
    bool fixed_probs(const SNode& nd, int p, int hand, float* pr) const {
        if (nd.bnode < 0 || S->tree.nodes[nd.bnode].type != 0 || S->tree.nodes[nd.bnode].player != p) return false;
        int b = cur_buckets[hand];
        if (b < 0) return false;
        float bp[16];
        S->probs(nd.bnode, b, bp);
        map_probs(S->tree, nd.bnode, nd.s, bp, nd.acts, pr);
        apply_tilt(models[p], nd.ctx, nd.acts, pr, 1.0);
        return true;
    }

    // Monte Carlo rollout with continuation strategies; returns utilities.
    void rollout(const SNode& leaf, Sample& sm, const int* choice, Rng& r, float* util) const {
        BetState s = leaf.s;
        int b = leaf.bnode;
        const Tree& t = S->tree;
        std::vector<Act> acts;
        acts.reserve(16);
        while (!s.done) {
            int p = s.to_act;
            abstract_actions(s, t.g, acts);
            int na = (int)acts.size();
            float pr[16];
            bool ok = b >= 0 && t.nodes[b].type == 0 && t.nodes[b].player == p;
            if (ok) {
                int bk = bucket_of(rt->abs, s.street, sm.full, sm.hand[p], S->level);
                float bp[16];
                S->probs(b, bk, bp);
                map_probs(t, b, s, bp, acts, pr);
                if (sm.fixed[p]) apply_tilt(models[p], context_of(s, p), acts, pr, 1.0);
                apply_bias(choice[p], acts, pr);
            } else {
                for (int i = 0; i < na; ++i) pr[i] = acts[i].code == A_CALL ? 1.f : 0.f;
            }
            double x = r.uniform();
            int a = 0;
            for (; a < na - 1; ++a) { x -= pr[a]; if (x <= 0) break; }
            uint8_t code = acts[a].code;
            b = ok ? nearest_child(t, b, code) : -1;
            s.apply(code == A_FOLD ? 0 : code == A_CALL ? 1 : 2, acts[a].to);
        }
        if (s.live() >= 2) ensure_values(sm);
        settle(s, sm.value, q->payouts, scale, util);
    }

    inline void regret_match(const float* r, int na, float* out) const {
        float s = 0;
        for (int a = 0; a < na; ++a) { out[a] = r[a] > 0 ? r[a] : 0; s += out[a]; }
        if (s > 0) for (int a = 0; a < na; ++a) out[a] /= s;
        else for (int a = 0; a < na; ++a) out[a] = 1.f / na;
    }
    static int pick(const float* pr, int na, Rng& r) {
        double x = r.uniform();
        int a = 0;
        for (; a < na - 1; ++a) { x -= pr[a]; if (x <= 0) break; }
        return a;
    }

    float traverse(int ni, int p, Sample& sm, Rng& r, double t) {
        SNode& nd = nodes[ni];
        if (nd.type == 1) {
            float u[MAX_PLAYERS];
            if (nd.s.live() >= 2) ensure_values(sm);
            settle(nd.s, sm.value, q->payouts, scale, u);
            return u[p];
        }
        if (nd.type == 2) {
            int choice[MAX_PLAYERS] = {0};
            bool p_chooses = false;
            for (int x = 0; x < q->n; ++x) {
                if (nd.s.folded[x] || nd.s.stack[x] == 0 || sm.hand[x] < 0) continue;
                if (x == p) { p_chooses = true; continue; }
                if (sm.fixed[x]) continue;
                float pr[4];
                regret_match(&R[nd.roff + ((size_t)x * H + hidx(sm.hand[x], x)) * 4], 4, pr);
                choice[x] = pick(pr, 4, r);
            }
            float u[MAX_PLAYERS];
            if (!p_chooses) { rollout(nd, sm, choice, r, u); return u[p]; }
            float* Rp = &R[nd.roff + ((size_t)p * H + hidx(sm.hand[p], p)) * 4];
            float sig[4], v[4], ev = 0;
            regret_match(Rp, 4, sig);
            Rng base = r;
            for (int k = 0; k < 4; ++k) {
                Rng rr = base;            // common random numbers across the four options
                choice[p] = k;
                rollout(nd, sm, choice, rr, u);
                v[k] = u[p];
                ev += sig[k] * v[k];
            }
            r.next();
            for (int k = 0; k < 4; ++k) Rp[k] += (float)(t * (v[k] - ev));
            return ev;
        }
        int x = nd.s.to_act, na = nd.nact;
        float pr[16];
        if (sm.fixed[x] && fixed_probs(nd, x, sm.hand[x], pr)) {
            return traverse(nd.child + pick(pr, na, r), p, sm, r, t);
        }
        float* Rx = &R[nd.roff + (size_t)hidx(sm.hand[x], x) * na];
        regret_match(Rx, na, pr);
        // A row with no positive regret yet plays the blueprint, not uniform: uniform responses
        // (calling or folding to a jam half the time with any hand) make jams look profitable.
        bool any_pos = false;
        for (int a = 0; a < na; ++a) any_pos |= Rx[a] > 0;
        if (!any_pos) blueprint_probs(nd, x, sm.hand[x], pr);
        if (x != p) return traverse(nd.child + pick(pr, na, r), p, sm, r, t);
        float v[16], ev = 0;
        for (int a = 0; a < na; ++a) { v[a] = traverse(nd.child + a, p, sm, r, t); ev += pr[a] * v[a]; }
        for (int a = 0; a < na; ++a) Rx[a] += (float)(t * (v[a] - ev));
        if (ni == 0 && hidx(sm.hand[p], p) == hidx(hero_h, p)) {
            float now[16];
            regret_match(Rx, na, now);
            bool pos_now = false;
            for (int a = 0; a < na; ++a) pos_now |= Rx[a] > 0;
            if (!pos_now) blueprint_probs(nd, x, sm.hand[x], now);
            std::lock_guard<std::mutex> lk(acc_mu);
            for (int a = 0; a < na; ++a) root_acc[a] += t * now[a];
        }
        return ev;
    }

    void run(double ms, int max_iters, int threads, uint64_t seed) {
        root_acc.assign(nodes[0].nact, 0.0);
        double t0 = now_seconds();
        std::vector<int> travs;
        for (int p = 0; p < q->n; ++p)
            if (!nodes[0].s.folded[p] && nodes[0].s.stack[p] > 0) travs.push_back(p);
        std::atomic<long> it(0);
        auto work = [&](int tid) {
            Rng r(seed * 1000003ull + (uint64_t)tid * 7919ull + 17);
            Sample sm;
            for (;;) {
                long i = it.fetch_add(1);
                if (max_iters > 0 && i >= max_iters) break;
                if (i > 8 && (now_seconds() - t0) * 1000 >= ms) break;
                double t = (double)(i + 1);
                for (int p : travs) {
                    if (!draw_sample(sm, p, r)) continue;
                    traverse(0, p, sm, r, t);
                }
                iter_count = i + 1;
            }
        };
        std::vector<std::thread> pool;
        for (int k = 1; k < threads; ++k) pool.emplace_back(work, k);
        work(0);
        for (auto& th : pool) th.join();
    }
};

// Equity of the hero against the opponents' ranges (share of the pot).
double hero_equity(const Request& q, const BetState& s, const std::vector<double>* ranges, int samples,
                   uint64_t seed) {
    if (q.hole[0] < 0) return -1;
    Rng r(seed + 99);
    uint64_t base = (1ull << q.hole[0]) | (1ull << q.hole[1]);
    for (int c : q.board) base |= 1ull << c;
    std::vector<std::vector<double>> cdf(q.n);
    std::vector<int> opp;
    for (int p = 0; p < q.n; ++p) {
        if (p == q.hero || s.folded[p]) continue;
        opp.push_back(p);
        cdf[p].resize(NUM_HOLES);
        double acc = 0;
        for (int h = 0; h < NUM_HOLES; ++h) {
            acc += (holes().mask[h] & base) ? 0 : ranges[p][h];
            cdf[p][h] = acc;
        }
        if (acc <= 0) {
            acc = 0;
            for (int h = 0; h < NUM_HOLES; ++h) { acc += (holes().mask[h] & base) ? 0 : 1; cdf[p][h] = acc; }
        }
    }
    if (opp.empty()) return 1.0;
    double won = 0;
    int done = 0;
    for (int it = 0; it < samples * 3 && done < samples; ++it) {
        uint64_t used = base;
        int hands[MAX_PLAYERS];
        bool ok = true;
        for (int p : opp) {
            int h = -1;
            for (int tries = 0; tries < 50; ++tries) {
                double x = r.uniform() * cdf[p].back();
                int c = (int)(std::lower_bound(cdf[p].begin(), cdf[p].end(), x) - cdf[p].begin());
                c = std::min(c, NUM_HOLES - 1);
                if (!(holes().mask[c] & used)) { h = c; break; }
            }
            if (h < 0) { ok = false; break; }
            hands[p] = h;
            used |= holes().mask[h];
        }
        if (!ok) continue;
        int full[5];
        for (size_t i = 0; i < q.board.size(); ++i) full[i] = q.board[i];
        for (int i = (int)q.board.size(); i < 5; ++i)
            for (;;) { int c = (int)r.below(52); if (!(used & (1ull << c))) { used |= 1ull << c; full[i] = c; break; } }
        int hc[7] = {full[0], full[1], full[2], full[3], full[4], q.hole[0], q.hole[1]};
        uint32_t hv = eval_cards(hc, 7), best = hv;
        int ties = 1;
        bool lost = false;
        for (int p : opp) {
            int c[7] = {full[0], full[1], full[2], full[3], full[4], holes().c1[hands[p]], holes().c2[hands[p]]};
            uint32_t v = eval_cards(c, 7);
            if (v > hv) { lost = true; break; }
            if (v == hv) ties++;
        }
        (void)best;
        if (!lost) won += 1.0 / ties;
        done++;
    }
    return done ? won / done : -1;
}

std::string jstr(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if ((unsigned char)c >= 32) o += c;
    }
    return o;
}

std::string jnum(double x) {
    if (!std::isfinite(x)) return "null";
    std::ostringstream o;
    o.precision(6);
    o << x;
    return o.str();
}

// May the hero move all-in here? Not when the effective stack is more than `max_jam_spr` times the
// pot (after calling), except preflop with at most `jam_free_bb` big blinds. Undertrained blueprints
// and short searches both overrate deep jams, because their opponents call or fold to them with
// little regard to their cards.
bool jam_allowed(const Request& q, const BetState& s) {
    if (q.max_jam_spr <= 0) return true;
    int h = s.to_act;
    int64_t other = 0;
    for (int i = 0; i < s.n; ++i)
        if (i != h && !s.folded[i]) other = std::max(other, s.stack[i] + s.street_c[i]);
    double eff = (double)std::min(s.stack[h] + s.street_c[h], other) - (double)s.street_c[h];
    double pot = (double)(s.pot() + s.to_call(h));
    if (s.street == 0 && eff <= q.jam_free_bb * (double)q.bb) return true;
    return eff <= q.max_jam_spr * pot;
}

std::string decide(Runtime& rt, const char* txt) {
    Request q;
    q.parse(txt);
    int bi = choose_blueprint(rt, q);
    Strategy& S = *rt.bps[bi];
    compute_baseline(rt, S);
    Model models[MAX_PLAYERS];
    build_models(q, S, models);
    Replay R;
    start_replay(rt, S, q, models, R, true);
    BetState& s = R.s;
    if (s.done) throw std::runtime_error("the hand is over");
    if (s.to_act != q.hero)
        throw std::runtime_error("position " + std::to_string(s.to_act) + " is to act, not the hero");
    int street = s.street;
    if ((int)q.board.size() != (street == 0 ? 0 : street + 2)) throw std::runtime_error("board does not match street");
    const Tree& t = S.tree;
    int hh = holes().index[q.hole[0]][q.hole[1]];
    ensure_buckets(rt, R, street);
    int hero_bucket = R.buckets[street][hh];
    bool bp_ok = R.on_tree && t.nodes[R.node].type == 0 && t.nodes[R.node].player == q.hero;
    // Preflop search also runs to exploit a live opponent whose observed play is far from the blueprint.
    bool exploit = false;
    for (int p = 0; p < q.n; ++p) {
        if (p == q.hero || s.folded[p]) continue;
        double dev = 0;
        for (int c = 0; c < NCTX; ++c) dev = std::max(dev, models[p].dev[c]);
        exploit |= models[p].weight * dev >= 0.05;
    }
    bool pre_search = q.search_preflop == 1 || (q.search_preflop == 2 && (!bp_ok || R.off_size || exploit));
    std::vector<Act> real;
    abstract_actions(s, t.g, real);
    bool do_search = q.mode == "search" && real.size() > 1 && (street > 0 || pre_search);
    Search sr;
    sr.rt = &rt;
    sr.S = &S;
    sr.q = &q;
    if (do_search) sr.search_actions(s, true, real);   // the search's full size menu
    int na = (int)real.size();
    float bp_real[16];
    if (bp_ok) {
        float bp[16];
        S.probs(R.node, hero_bucket, bp);
        map_probs(t, R.node, s, bp, real, bp_real);
    } else {
        for (int i = 0; i < na; ++i) bp_real[i] = real[i].code == A_CALL ? 1.f : 0.f;
    }
    std::vector<float> final_p(bp_real, bp_real + na);
    bool jam_ok = jam_allowed(q, s);
    long iters = 0;
    double ms_used = 0;
    bool searched = false;
    int depth_used = 0;
    if (do_search) {
        double t0 = now_seconds();
        sr.models = models;
        sr.street = street;
        sr.H = street == 0 ? 169 : NUM_HOLES;
        sr.board = q.board;
        sr.ranges = R.range;
        sr.hero = q.hero;
        sr.hero_h = hh;
        sr.cur_buckets = R.buckets[street];
        double chips = 0;
        for (int i = 0; i < q.n; ++i) chips += (double)q.stacks[i];
        sr.scale = chips / (double)q.bb;
        sr.drop_root_allin = !jam_ok;
        sr.build(s, bp_ok ? R.node : -1, q.depth, 60000000);
        sr.build_alias();
        sr.run(q.search_ms, q.iters, std::max(1, q.threads), q.seed);
        iters = sr.iter_count;
        double sum = 0;
        for (double x : sr.root_acc) sum += x;
        if (sum > 0) {
            for (int i = 0; i < na; ++i) final_p[i] = 0.f;
            for (int i = 0; i < (int)sr.root_acc.size(); ++i) final_p[i] = (float)(sr.root_acc[i] / sum);
            searched = true;
        }
        depth_used = sr.depth_used;
        ms_used = (now_seconds() - t0) * 1000;
    }
    if (!jam_ok && real[na - 1].code == A_ALLIN && final_p[na - 1] > 0) {
        // Move the all-in share to the largest other raise, or to check/call if there is none.
        int tgt = -1;
        for (int i = 0; i < na - 1; ++i) if (real[i].code >= A_SIZE0) tgt = i;
        if (tgt < 0) for (int i = 0; i < na - 1; ++i) if (real[i].code == A_CALL) tgt = i;
        if (tgt >= 0) { final_p[tgt] += final_p[na - 1]; final_p[na - 1] = 0.f; }
    }
    double eq = hero_equity(q, s, R.range, q.equity_samples, q.seed);
    std::ostringstream o;
    o << "{\"blueprint\":\"" << S.name << "\",\"players\":" << t.n << ",\"stack_bb\":" << jnum(S.stack_bb)
      << ",\"iterations_trained\":" << S.iterations << ",\"on_tree\":" << (bp_ok ? "true" : "false")
      << ",\"searched\":" << (searched ? "true" : "false") << ",\"search_iterations\":" << iters
      << ",\"search_ms\":" << jnum(ms_used) << ",\"depth\":" << depth_used << ",\"bucket\":" << hero_bucket
      << ",\"equity\":" << jnum(eq) << ",\"street\":" << street << ",\"options\":[";
    for (int i = 0; i < na; ++i) {
        if (i) o << ",";
        double frac = real[i].code >= A_SIZE0 ? real[i].frac : 0;
        o << "{\"code\":" << (int)real[i].code << ",\"label\":\"" << act_label(real[i], s)
          << "\",\"to\":" << real[i].to << ",\"pot_frac\":" << jnum(frac) << ",\"blueprint\":" << jnum(bp_real[i])
          << ",\"prob\":" << jnum(final_p[i]) << "}";
    }
    o << "],\"snapped\":[";
    {
        // Sizes in this spot's list that commit most of the stack and become all-in.
        bool f1 = true;
        if (s.can_raise(q.hero)) {
            const auto& sz = t.g.sizes(s.street, s.raises, s.live());
            int64_t maxto = s.max_raise_to(q.hero), minto = s.min_raise_to(q.hero), tc = s.to_call(q.hero);
            for (double f : sz) {
                int64_t to = std::max(s.cur_bet() + (int64_t)std::llround(f * (double)(s.pot() + tc)), minto);
                if (to >= t.g.snap * (double)maxto || to >= maxto) { o << (f1 ? "" : ",") << jnum(f); f1 = false; }
            }
        }
    }
    o << "],\"max_to\":" << s.max_raise_to(q.hero) << ",\"min_to\":" << s.min_raise_to(q.hero)
      << ",\"to_call\":" << s.to_call(q.hero) << ",\"pot\":" << s.pot() << ",\"reads\":[";
    bool first = true;
    for (int p = 0; p < q.n; ++p) {
        if (p == q.hero) continue;
        if (!first) o << ",";
        first = false;
        o << "{\"pos\":" << p << ",\"weight\":" << jnum(models[p].weight) << ",\"observed\":" << jnum(models[p].total)
          << ",\"fold_to_3bet\":" << jnum(models[p].post[3][0]) << ",\"cbet\":" << jnum(models[p].post[5][2])
          << ",\"river_call\":" << jnum(models[p].post[14][1]) << ",\"open\":" << jnum(models[p].post[0][2]) << "}";
    }
    o << "],\"range_top\":[";
    // Most likely classes in each live opponent's range (diagnostics).
    first = true;
    for (int p = 0; p < q.n; ++p) {
        if (p == q.hero || s.folded[p]) continue;
        double cls[169] = {0}, tot = 0;
        for (int h = 0; h < NUM_HOLES; ++h) { cls[holes().cls[h]] += R.range[p][h]; tot += R.range[p][h]; }
        if (!first) o << ",";
        first = false;
        // Share of all 1326 combos that carry weight (range width).
        double width = 0, mx = 0;
        for (int h = 0; h < NUM_HOLES; ++h) mx = std::max(mx, R.range[p][h]);
        for (int h = 0; h < NUM_HOLES; ++h) width += mx > 0 ? R.range[p][h] / mx : 0;
        o << "{\"pos\":" << p << ",\"width\":" << jnum(width / NUM_HOLES) << "}";
    }
    o << "]}";
    return o.str();
}

// Contexts of every voluntary action in a hand (for opponent statistics).
std::string contexts(Runtime& rt, const char* txt) {
    Request q;
    q.parse(txt);
    int bi = choose_blueprint(rt, q);
    Strategy& S = *rt.bps[bi];
    ensure_loaded(S);
    Model models[MAX_PLAYERS];
    Replay R;
    R.s.init(q.n, q.stacks, q.sb, q.bb, q.ante);
    R.board = q.board;
    for (auto& a : q.actions) {
        if (R.s.done) break;
        if (a.pos != R.s.to_act) throw std::runtime_error("action out of turn in hand replay");
        replay_action(rt, S, models, R, a.kind, a.to, false);
    }
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < R.obs.size(); ++i) {
        if (i) o << ",";
        o << "[" << R.obs[i].pos << "," << R.obs[i].ctx << "," << R.obs[i].cat << "]";
    }
    o << "]";
    return o.str();
}

// Read a strategy file's header now; build the tree and map the data on first use.
Strategy* load_strategy(Runtime& rt, const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    uint32_t magic = 0, len = 0;
    if (std::fread(&magic, 4, 1, f) != 1 || magic != 0x35534250) {
        std::fclose(f);
        throw std::runtime_error(path + " is not a v5 strategy file");
    }
    if (std::fread(&len, 4, 1, f) != 1) throw std::runtime_error("bad strategy file");
    std::string txt(len, '\0');
    if (std::fread(&txt[0], 1, len, f) != len) throw std::runtime_error("bad strategy file");
    auto st = std::make_unique<Strategy>();
    if (std::fread(&st->iterations, 8, 1, f) != 1 || std::fread(&st->chunks, 8, 1, f) != 1 ||
        std::fread(&st->data_bytes, 8, 1, f) != 1)
        throw std::runtime_error("bad strategy file");
    st->data_offset = 4 + 4 + (long)len + 24;
    std::fclose(f);
    st->header = txt;
    KV kv;
    kv.parse(txt);
    if ((uint64_t)std::strtoull(kv.str("abstraction", "0").c_str(), nullptr, 10) != rt.abs.hash)
        throw std::runtime_error(path + " was trained with a different card abstraction");
    st->players = (int)kv.num("players", 0);
    st->prof = Stacks::from_kv(kv);
    st->stack_bb = st->prof.max();
    st->level = (int)kv.num("level", 0);
    if (st->level < 0 || st->level >= rt.abs.nlevels()) throw std::runtime_error(path + ": unknown bucket level");
    st->path = path;
    st->slots = st->data_bytes;
    std::ostringstream nm;
    nm << st->players << " players, " << st->prof.label() << " bb";
    st->name = nm.str();
    st->tree.g.from_kv(kv);
    rt.bps.push_back(std::move(st));
    return rt.bps.back().get();
}

void ensure_loaded(Strategy& st) {
    std::lock_guard<std::mutex> lk(st.mu);
    if (st.ready) return;
    KV kv;
    kv.parse(st.header);
    GameConfig g;
    g.from_kv(kv);
    std::vector<double> b = kv.list("buckets");
    if (b.size() != 4) throw std::runtime_error("bad bucket list in " + st.path);
    int K[4] = {(int)b[0], (int)b[1], (int)b[2], (int)b[3]};
    st.tree.build(st.players, st.prof, g, K, st.level);
    if (st.tree.total_chunks != st.chunks) throw std::runtime_error(st.path + ": tree does not match the stored strategy");
    int fd = open(st.path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + st.path);
    struct stat sb;
    fstat(fd, &sb);
    size_t need = (size_t)st.data_offset + st.chunks * 4 + st.data_bytes;
    if ((size_t)sb.st_size < need) { close(fd); throw std::runtime_error(st.path + " is truncated"); }
    void* m = mmap(nullptr, (size_t)sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED) throw std::runtime_error("cannot map " + st.path);
    madvise(m, (size_t)sb.st_size, MADV_RANDOM);
    st.map = (const uint8_t*)m;
    st.maplen = (size_t)sb.st_size;
    st.table = (const uint32_t*)(st.map + st.data_offset);
    st.data = st.map + st.data_offset + st.chunks * 4;
    st.ready = true;
}

}  // namespace

PB_API const char* pb_error() { return g_err.c_str(); }

PB_API void* pb_open(const char* dir) {
    try {
        auto* rt = new Runtime();
        rt->abs.load(dir);
        return rt;
    } catch (const std::exception& e) {
        g_err = e.what();
        return nullptr;
    }
}

PB_API int pb_load(void* h, const char* path) {
    try {
        Runtime& rt = *(Runtime*)h;
        load_strategy(rt, path);
        return (int)rt.bps.size() - 1;
    } catch (const std::exception& e) {
        g_err = e.what();
        return -1;
    }
}

PB_API const char* pb_decide(void* h, const char* req) {
    try {
        g_out = decide(*(Runtime*)h, req);
    } catch (const std::exception& e) {
        g_out = std::string("{\"error\":\"") + jstr(e.what()) + "\"}";
    }
    return g_out.c_str();
}

PB_API const char* pb_contexts(void* h, const char* req) {
    try {
        g_out = contexts(*(Runtime*)h, req);
    } catch (const std::exception& e) {
        g_out = std::string("{\"error\":\"") + jstr(e.what()) + "\"}";
    }
    return g_out.c_str();
}

PB_API const char* pb_info(void* h) {
    Runtime& rt = *(Runtime*)h;
    std::ostringstream o;
    o << "{\"abstraction\":[169," << rt.abs.K[1] << "," << rt.abs.K[2] << "," << rt.abs.K[3] << "],\"version\":5,\"blueprints\":[";
    for (size_t i = 0; i < rt.bps.size(); ++i) {
        if (i) o << ",";
        const Strategy& s = *rt.bps[i];
        std::vector<double> all;
        for (auto* v : {&s.tree.g.pre_open, &s.tree.g.pre_3bet, &s.tree.g.pre_later, &s.tree.g.post_bet,
                        &s.tree.g.post_raise, &s.tree.g.post_later, &s.tree.g.multi_bet, &s.tree.g.multi_raise})
            all.insert(all.end(), v->begin(), v->end());
        std::sort(all.begin(), all.end());
        all.erase(std::unique(all.begin(), all.end()), all.end());
        o << "{\"index\":" << i << ",\"sizes\":[";
        for (size_t k = 0; k < all.size(); ++k) o << (k ? "," : "") << jnum(all[k]);
        o << "],\"snap\":" << jnum(s.tree.g.snap) << ",\"payouts\":[";
        for (size_t k = 0; k < s.tree.g.payouts.size(); ++k) o << (k ? "," : "") << jnum(s.tree.g.payouts[k]);
        o << "],\"players\":" << s.players << ",\"stack_bb\":" << jnum(s.stack_bb) << ",\"stacks\":\""
          << s.prof.label() << "\""
          << ",\"iterations\":" << s.iterations << ",\"path\":\"" << jstr(s.path) << "\",\"bytes\":" << s.slots
          << ",\"level\":" << s.level << ",\"loaded\":" << (s.ready ? "true" : "false") << "}";
    }
    o << "],\"contexts\":[";
    for (int c = 0; c < NCTX; ++c) o << (c ? "," : "") << "\"" << CTX_NAMES[c] << "\"";
    o << "]}";
    g_out = o.str();
    return g_out.c_str();
}

PB_API void pb_close(void* h) { delete (Runtime*)h; }
