// Betting rules, action abstraction and the abstract game tree.
//
// BetState mirrors pokerbot/table.py exactly (blinds, antes, minimum raises,
// short all-ins that do not reopen the betting, street order). It is used
// both to build the abstract tree (big blind = 100 chips; equal stacks, or one
// stack per position for an unequal-stack blueprint) and to
// replay real hands at run time (real chips).
//
// Positions: 0 = button, then clockwise. Heads-up, the button posts the small
// blind. With 3+ players, 1 = small blind, 2 = big blind.
#pragma once

#include "pb_core.h"

namespace pb {

enum ActCode : uint8_t { A_FOLD = 0, A_CALL = 1, A_SIZE0 = 2, A_ALLIN = 15 };

// Starting stacks of a blueprint in big blinds: one number (everyone equal) or
// one per position ("10-30-30": button, small blind, big blind, ...).
struct Stacks {
    double bb[MAX_PLAYERS] = {0};
    int k = 1;                                   // 1 = equal stacks
    Stacks(double s = 0) { bb[0] = s; }
    static Stacks parse(const std::string& spec) {
        Stacks r;
        r.k = 0;
        std::string cur;
        for (size_t i = 0; i <= spec.size(); ++i) {
            if (i == spec.size() || spec[i] == '-' || spec[i] == '/') {
                if (cur.empty()) throw std::runtime_error("bad stack spec " + spec);
                if (r.k >= MAX_PLAYERS) throw std::runtime_error("too many stacks in " + spec);
                r.bb[r.k++] = std::atof(cur.c_str());
                cur.clear();
            } else cur += spec[i];
        }
        for (int i = 0; i < r.k; ++i) if (r.bb[i] <= 0) throw std::runtime_error("bad stack spec " + spec);
        bool eq = true;
        for (int i = 1; i < r.k; ++i) eq &= r.bb[i] == r.bb[0];
        if (eq) r.k = 1;
        return r;
    }
    // From a checkpoint or strategy header ("stack=" in chips, "stacks=" per position).
    static Stacks from_kv(const KV& kv) {
        if (kv.has("stacks")) {
            Stacks r;
            std::vector<double> v = kv.list("stacks");
            r.k = (int)v.size();
            for (int i = 0; i < r.k && i < MAX_PLAYERS; ++i) r.bb[i] = v[i] / 100.0;
            return r;
        }
        return Stacks(kv.num("stack", 0) / 100.0);
    }
    bool equal() const { return k == 1; }
    double at(int i) const { return k == 1 ? bb[0] : bb[i]; }
    double max() const { double m = 0; for (int i = 0; i < k; ++i) m = std::max(m, bb[i]); return m; }
    void check(int players) const {
        if (k != 1 && k != players)
            throw std::runtime_error("a stack profile needs one stack per player (" + std::to_string(players) + ")");
    }
    std::string label() const {        // "20" or "10-30-30"
        std::ostringstream o;
        for (int i = 0; i < k; ++i) { if (i) o << "-"; o << bb[i]; }
        return o.str();
    }
    // Effective stack of each position: its own stack, capped by the largest other stack.
    void effective(int n, double* eff) const {
        for (int i = 0; i < n; ++i) {
            double other = 0;
            for (int j = 0; j < n; ++j) if (j != i) other = std::max(other, at(j));
            eff[i] = std::min(at(i), other);
        }
    }
};

struct GameConfig {
    // Action abstraction (fractions of the pot after calling).
    std::vector<double> pre_open, pre_3bet, pre_later;
    std::vector<double> post_bet, post_raise, post_later;
    std::vector<double> multi_bet, multi_raise;
    int max_raises_pre = 4, max_raises_post = 3, max_raises_multi = 2;
    double snap = 0.6;            // sizes reaching this share of the max become all-in
    double ante_bb = 0.1;
    std::vector<double> payouts{0.65, 0.35};
    std::string text;             // normalized config, for hashing

    // players > 0: "key@players" entries override "key" entries.
    void from_kv(const KV& kv, int players = 0) {
        auto L = [&](const char* k, std::vector<double> d) { return kv.list_p(k, players, d); };
        auto N = [&](const char* k, double d) { return kv.num_p(k, players, d); };
        pre_open = L("preflop_open", {0.33, 0.5, 0.75});
        pre_3bet = L("preflop_3bet", {0.75});
        pre_later = L("preflop_later", {});
        post_bet = L("postflop_bet", {0.33, 0.75, 1.5});
        post_raise = L("postflop_raise", {0.75});
        post_later = L("postflop_later", {});
        multi_bet = L("multiway_bet", {0.33, 0.75});
        multi_raise = L("multiway_raise", {});
        max_raises_pre = (int)N("max_raises_preflop", 4);
        max_raises_post = (int)N("max_raises_postflop", 3);
        max_raises_multi = (int)N("max_raises_multiway", 2);
        snap = N("allin_snap", 0.6);
        ante_bb = N("ante_bb", 0.1);
        payouts = L("payouts", {0.65, 0.35});
        for (auto* v : {&pre_open, &pre_3bet, &pre_later, &post_bet, &post_raise, &post_later, &multi_bet, &multi_raise})
            if (v->size() > 12) throw std::runtime_error("at most 12 raise sizes per list");
        std::ostringstream o;
        auto put = [&](const char* k, const std::vector<double>& v) {
            o << k << "=";
            for (double x : v) o << x << " ";
            o << "\n";
        };
        put("preflop_open", pre_open); put("preflop_3bet", pre_3bet); put("preflop_later", pre_later);
        put("postflop_bet", post_bet); put("postflop_raise", post_raise); put("postflop_later", post_later);
        put("multiway_bet", multi_bet); put("multiway_raise", multi_raise);
        o << "max_raises_preflop=" << max_raises_pre << "\nmax_raises_postflop=" << max_raises_post
          << "\nmax_raises_multiway=" << max_raises_multi
          << "\nallin_snap=" << snap << "\nante_bb=" << ante_bb << "\n";
        put("payouts", payouts);
        text = o.str();
    }
    const std::vector<double>& sizes(int street, int raises, int live) const {
        if (street == 0) return raises == 0 ? pre_open : raises == 1 ? pre_3bet : pre_later;
        if (live >= 3) return raises == 0 ? multi_bet : raises == 1 ? multi_raise : post_later;
        return raises == 0 ? post_bet : raises == 1 ? post_raise : post_later;
    }
};

struct Act {
    uint8_t code;
    int64_t to;        // raise-to total on this street (0 for fold/call)
    float frac = 0;    // raise size as a fraction of the pot after calling (raises only)
};

struct BetState {
    int n = 0;
    int64_t start[MAX_PLAYERS] = {0};    // stacks at the start of the hand
    int64_t stack[MAX_PLAYERS] = {0};    // chips behind
    int64_t street_c[MAX_PLAYERS] = {0}, total_c[MAX_PLAYERS] = {0};
    int64_t acted[MAX_PLAYERS];          // full_level when the player last acted; -1 = not yet
    bool folded[MAX_PLAYERS] = {false};
    int street = 0, to_act = -1, raises = 0;
    int64_t bb = 100, last_raise = 100, full_level = 0;
    int last_aggr = -1, prev_aggr = -1;  // last bettor/raiser this street, previous street
    bool done = false;                   // hand over (fold or showdown reached)

    int64_t pot() const { int64_t s = 0; for (int i = 0; i < n; ++i) s += total_c[i]; return s; }
    int64_t cur_bet() const { int64_t m = 0; for (int i = 0; i < n; ++i) m = std::max(m, street_c[i]); return m; }
    int live() const { int k = 0; for (int i = 0; i < n; ++i) k += !folded[i]; return k; }
    int with_chips() const { int k = 0; for (int i = 0; i < n; ++i) k += (!folded[i] && stack[i] > 0); return k; }

    void post(int p, int64_t amt) {
        int64_t paid = std::min(amt, stack[p]);
        stack[p] -= paid; street_c[p] += paid; total_c[p] += paid;
    }
    // Start a hand: antes, blinds, first player to act.
    void init(int players, const int64_t* stacks, int64_t sb, int64_t bbig, int64_t ante) {
        n = players;
        bb = bbig;
        for (int i = 0; i < n; ++i) {
            start[i] = stack[i] = stacks[i];
            street_c[i] = total_c[i] = 0;
            folded[i] = false;
            acted[i] = -1;
        }
        if (ante > 0)
            for (int i = 0; i < n; ++i) { int64_t a = std::min(ante, stack[i]); stack[i] -= a; total_c[i] += a; }
        int sbp = n == 2 ? 0 : 1, bbp = n == 2 ? 1 : 2;
        post(sbp, sb);
        post(bbp, bbig);
        street = 0; raises = 0; last_aggr = prev_aggr = -1;
        last_raise = bbig;
        full_level = cur_bet();
        done = false;
        to_act = -1;
        begin_round(n == 2 ? 0 : (bbp + 1) % n);
    }
    bool needs(int p) const {
        if (folded[p] || stack[p] == 0) return false;
        return acted[p] < 0 || street_c[p] < cur_bet();
    }
    // Find the next player to act starting at seat `from` (inclusive);
    // -1 if the betting round is over.
    int find_actor(int from) const {
        if (live() < 2) return -1;
        bool any = false;
        for (int i = 0; i < n; ++i) any |= needs(i);
        if (!any) return -1;
        if (with_chips() == 1) {
            for (int i = 0; i < n; ++i)
                if (!folded[i] && stack[i] > 0 && street_c[i] >= cur_bet()) return -1;
        }
        for (int k = 0; k < n; ++k) {
            int p = (from + k) % n;
            if (needs(p)) return p;
        }
        return -1;
    }
    void begin_round(int first) {
        int p = find_actor(first);
        if (p >= 0) { to_act = p; return; }
        end_round();
    }
    // Betting round over: next street, or the hand is over.
    void end_round() {
        if (live() < 2 || street == 3) { done = true; to_act = -1; return; }
        street++;
        for (int i = 0; i < n; ++i) { street_c[i] = 0; acted[i] = -1; }
        prev_aggr = last_aggr;
        last_aggr = -1;
        raises = 0;
        last_raise = bb;
        full_level = 0;
        begin_round(1 % n);   // first live player after the button
    }
    // True when the remaining streets need no decisions (all-in runout).
    bool runout() const { return done && live() >= 2; }

    int64_t to_call(int p) const { return std::min(cur_bet() - street_c[p], stack[p]); }
    bool can_raise(int p) const {
        bool others = false;
        for (int i = 0; i < n; ++i) if (i != p && !folded[i] && stack[i] > 0) others = true;
        bool reopened = acted[p] < 0 || acted[p] < full_level;
        return stack[p] > to_call(p) && reopened && others;
    }
    int64_t min_raise_to(int p) const { return std::min(cur_bet() + last_raise, street_c[p] + stack[p]); }
    int64_t max_raise_to(int p) const { return street_c[p] + stack[p]; }

    // Apply a decision: kind 0 fold, 1 check/call, 2 raise to `to`.
    void apply(int kind, int64_t to) {
        int p = to_act;
        int64_t tc = to_call(p);
        if (kind == 0 && tc == 0) kind = 1;
        if (kind == 2 && !can_raise(p)) kind = 1;
        if (kind == 0) {
            folded[p] = true;
        } else if (kind == 1) {
            stack[p] -= tc; street_c[p] += tc; total_c[p] += tc;
        } else {
            to = std::min(std::max(to, min_raise_to(p)), max_raise_to(p));
            int64_t pay = to - street_c[p], inc = to - cur_bet();
            if (inc >= last_raise) { last_raise = inc; full_level = to; }
            stack[p] -= pay; street_c[p] = to; total_c[p] += pay;
            raises++;
            last_aggr = p;
        }
        acted[p] = full_level;
        int nxt = find_actor((p + 1) % n);
        if (nxt >= 0) to_act = nxt; else end_round();
    }
};

inline double size_frac(const BetState& s, int64_t to) {
    int p = s.to_act;
    double pot = (double)(s.pot() + s.to_call(p));
    return pot > 0 ? ((double)to - (double)s.cur_bet()) / pot : 0.0;
}

// Actions at the current decision from an explicit size list (pot fractions),
// with real raise-to amounts. `cap` = most raises allowed on this street.
inline void menu_actions(const BetState& s, const std::vector<double>& sz, int cap, double snap,
                         std::vector<Act>& out) {
    out.clear();
    int p = s.to_act;
    int64_t tc = s.to_call(p);
    if (tc > 0) out.push_back({A_FOLD, 0, 0.f});
    out.push_back({A_CALL, 0, 0.f});
    if (!s.can_raise(p)) return;
    if (s.raises >= cap) return;
    int64_t maxto = s.max_raise_to(p), minto = s.min_raise_to(p);
    int64_t pot = s.pot(), cb = s.cur_bet();
    int64_t last = -1;
    for (size_t k = 0; k < sz.size() && k < 13; ++k) {
        int64_t to = cb + (int64_t)std::llround(sz[k] * (double)(pot + tc));
        to = std::max(to, minto);
        if (to >= snap * (double)maxto || to >= maxto) continue;   // becomes all-in
        if (to <= last) continue;
        out.push_back({(uint8_t)(A_SIZE0 + k), to, (float)size_frac(s, to)});
        last = to;
    }
    out.push_back({A_ALLIN, maxto, (float)size_frac(s, maxto)});
}

inline int raise_cap(const BetState& s, const GameConfig& g) {
    return s.street == 0 ? g.max_raises_pre : (s.live() >= 3 ? g.max_raises_multi : g.max_raises_post);
}

// Abstract actions at the current decision, with real raise-to amounts.
inline void abstract_actions(const BetState& s, const GameConfig& g, std::vector<Act>& out) {
    menu_actions(s, g.sizes(s.street, s.raises, s.live()), raise_cap(s, g), g.snap, out);
}

// Index in `acts` of the option that best represents a raise to `to`
// (pseudo-harmonic mapping between the two nearest sizes; all-in if no size).
inline int map_raise(const BetState& s, const std::vector<Act>& acts, int64_t to) {
    int p = s.to_act;
    int64_t maxto = s.max_raise_to(p);
    int allin = -1;
    for (int i = 0; i < (int)acts.size(); ++i) if (acts[i].code == A_ALLIN) allin = i;
    if (to >= maxto && allin >= 0) return allin;
    double x = size_frac(s, to);
    int lo = -1, hi = -1;
    for (int i = 0; i < (int)acts.size(); ++i) {
        if (acts[i].code < A_SIZE0) continue;
        double f = acts[i].frac;
        if (f <= x) lo = i;
        else if (hi < 0) hi = i;
    }
    if (lo < 0 && hi < 0) {
        for (int i = 0; i < (int)acts.size(); ++i) if (acts[i].code == A_CALL) return i;
        return 0;
    }
    if (lo < 0) return hi;
    if (hi < 0) return lo;
    double A = acts[lo].frac, B = acts[hi].frac;
    double pA = ((B - x) * (1 + A)) / ((B - A) * (1 + x));
    return pA >= 0.5 ? lo : hi;
}

inline std::string act_label(const Act& a, const BetState& s) {
    if (a.code == A_FOLD) return "fold";
    if (a.code == A_CALL) return s.to_call(s.to_act) > 0 ? "call" : "check";
    if (a.code == A_ALLIN) return "all-in";
    std::ostringstream o;
    o << (int)std::lround(a.frac * 100) << "% pot";
    return o.str();
}

// ---------------------------------------------------------------------------
// Abstract game tree.
// Regrets and strategies are stored in chunks of CHUNK buckets of one node.
constexpr int CHUNK = 16;

struct Node {
    uint64_t offset;    // first chunk index (decision) or terminal index
    uint32_t avg;       // preflop: offset of the average strategy (in entries)
    int32_t child;      // first child node index
    uint8_t player, street, nact, type;   // type: 0 decision, 1 fold end, 2 showdown
};

struct Terminal {
    int32_t contrib[MAX_PLAYERS];
    uint8_t live;       // bit mask of players still in the hand
    float fold_util[MAX_PLAYERS];   // utilities when the hand ended by folds
};

struct Tree {
    int n = 0;
    int64_t stack = 0;                   // largest starting stack (chips)
    int64_t stacks[MAX_PLAYERS] = {0};   // starting stack per position (chips)
    Stacks prof;
    GameConfig g;
    int K[4] = {169, 0, 0, 0};
    std::vector<Node> nodes;
    std::vector<uint8_t> codes;          // action code per node (for the edge into it)
    std::vector<Terminal> terms;
    uint64_t total_slots = 0;            // (bucket, action) entries
    uint64_t total_chunks = 0;
    uint64_t avg_slots = 0;              // preflop average-strategy entries
    uint64_t slots_by_street[4] = {0};
    uint64_t nodes_by_street[4] = {0};
    uint64_t chunk_bytes_by_street[4] = {0};   // 2-byte regrets, chunk-padded
    uint64_t nact_by_street[4] = {0};          // sum of action counts over decision nodes
    int level = 0;
    double util_scale = 1.0;             // prize pool in big blinds

    int64_t sb() const { return 50; }
    int64_t bbc() const { return 100; }
    int64_t ante() const { return (int64_t)std::llround(g.ante_bb * 100); }

    void build(int players, const Stacks& stack_bb, const GameConfig& cfg, const int* buckets, int lvl = 0) {
        n = players;
        g = cfg;
        level = lvl;
        stack_bb.check(players);
        prof = stack_bb;
        stack = (int64_t)std::llround(stack_bb.max() * 100);
        for (int i = 0; i < n; ++i) stacks[i] = (int64_t)std::llround(stack_bb.at(i) * 100);
        for (int s = 0; s < 4; ++s) K[s] = buckets[s];
        nodes.clear(); codes.clear(); terms.clear();
        total_slots = total_chunks = avg_slots = 0;
        for (int s = 0; s < 4; ++s) slots_by_street[s] = nodes_by_street[s] = chunk_bytes_by_street[s] = nact_by_street[s] = 0;
        if (stack_bb.equal()) util_scale = stack_bb.at(0) * n;
        else { util_scale = 0; for (int i = 0; i < n; ++i) util_scale += stack_bb.at(i); }
        BetState s;
        s.init(n, stacks, sb(), bbc(), ante());
        nodes.push_back(Node());
        codes.push_back(255);
        build_rec(0, s);
    }
    void make_terminal(int idx, const BetState& s) {
        Terminal t;
        t.live = 0;
        for (int i = 0; i < MAX_PLAYERS; ++i) { t.contrib[i] = 0; t.fold_util[i] = 0; }
        for (int i = 0; i < n; ++i) {
            t.contrib[i] = (int32_t)s.total_c[i];
            if (!s.folded[i]) t.live |= (uint8_t)(1u << i);
        }
        Node& nd = nodes[idx];
        nd.type = s.live() < 2 ? 1 : 2;
        nd.nact = 0;
        nd.player = 255;
        nd.street = (uint8_t)s.street;
        nd.offset = terms.size();
        nd.avg = 0;
        if (nd.type == 1) {
            double fin[MAX_PLAYERS], start[MAX_PLAYERS], eq[MAX_PLAYERS];
            int64_t pot = s.pot();
            for (int i = 0; i < n; ++i) {
                start[i] = (double)stacks[i];
                fin[i] = (double)s.stack[i] + (!s.folded[i] ? (double)pot : 0.0);
            }
            icm(fin, start, n, g.payouts, eq);
            for (int i = 0; i < n; ++i) t.fold_util[i] = (float)(eq[i] * util_scale);
        }
        terms.push_back(t);
    }
    void build_rec(int idx, const BetState& s) {
        if (s.done) { make_terminal(idx, s); return; }
        std::vector<Act> acts;
        abstract_actions(s, g, acts);
        int na = (int)acts.size();
        int first = (int)nodes.size();
        {
            Node& nd = nodes[idx];
            nd.type = 0;
            nd.player = (uint8_t)s.to_act;
            nd.street = (uint8_t)s.street;
            nd.nact = (uint8_t)na;
            nd.child = first;
            nd.offset = total_chunks;
            nd.avg = s.street == 0 ? (uint32_t)avg_slots : 0;
        }
        uint64_t slots = (uint64_t)K[s.street] * na;
        uint64_t nch = (uint64_t)(K[s.street] + CHUNK - 1) / CHUNK;
        total_slots += slots;
        total_chunks += nch;
        if (s.street == 0) avg_slots += slots;
        slots_by_street[s.street] += slots;
        chunk_bytes_by_street[s.street] += nch * CHUNK * na * 2;
        nact_by_street[s.street] += na;
        nodes_by_street[s.street]++;
        for (int a = 0; a < na; ++a) { nodes.push_back(Node()); codes.push_back(acts[a].code); }
        for (int a = 0; a < na; ++a) {
            BetState c = s;
            c.apply(acts[a].code == A_FOLD ? 0 : acts[a].code == A_CALL ? 1 : 2, acts[a].to);
            build_rec(first + a, c);
        }
    }
    // Index of the child of node `idx` reached by action code `code`; -1 if none.
    // Training memory with every chunk allocated (bytes).
    // ... for other bucket counts on the same action tree.
    uint64_t bytes_for(const int* KK) const {
        uint64_t b = 0, ch = 0;
        for (int s = 0; s < 4; ++s) {
            uint64_t nch = (uint64_t)(KK[s] + CHUNK - 1) / CHUNK;
            b += nch * CHUNK * 2 * nact_by_street[s];
            ch += nch * nodes_by_street[s];
        }
        return b + ch * 5 + avg_slots * 4 + nodes.size() * (sizeof(Node) + 1);
    }
    uint64_t full_bytes() const {
        uint64_t b = 0;
        for (int s = 0; s < 4; ++s) b += chunk_bytes_by_street[s];
        return b + total_chunks * 5 + avg_slots * 4 + nodes.size() * (sizeof(Node) + 1);
    }
    int child_by_code(int idx, uint8_t code) const {
        const Node& nd = nodes[idx];
        for (int a = 0; a < nd.nact; ++a) if (codes[nd.child + a] == code) return nd.child + a;
        return -1;
    }
    std::string hash_text() const {
        std::ostringstream o;
        o << "players=" << n << "\nstack=" << stack << "\n";
        if (!prof.equal()) {            // equal-stack headers are unchanged, so old checkpoints still resume
            o << "stacks=";
            for (int i = 0; i < n; ++i) o << (i ? " " : "") << stacks[i];
            o << "\n";
        }
        o << g.text << "buckets=" << K[0] << " " << K[1]
          << " " << K[2] << " " << K[3] << "\nlevel=" << level << "\n";
        return o.str();
    }
};

// Showdown utility for every player given hand values of the live players.
// Side pots are settled as in pokerbot/table.py; odd chips are split evenly.
inline void showdown_utility(const Tree& t, const Terminal& term, const uint32_t* value, float* util) {
    int n = t.n;
    double fin[MAX_PLAYERS], start[MAX_PLAYERS], eq[MAX_PLAYERS];
    int64_t contrib[MAX_PLAYERS];
    for (int i = 0; i < n; ++i) {
        contrib[i] = term.contrib[i];
        start[i] = (double)t.stacks[i];
        fin[i] = (double)(t.stacks[i] - contrib[i]);
    }
    // Pot levels.
    int64_t levels[MAX_PLAYERS];
    int nl = 0;
    for (int i = 0; i < n; ++i) if (contrib[i] > 0) levels[nl++] = contrib[i];
    std::sort(levels, levels + nl);
    nl = (int)(std::unique(levels, levels + nl) - levels);
    int64_t prev = 0;
    for (int l = 0; l < nl; ++l) {
        int64_t lv = levels[l];
        double portion = 0;
        for (int i = 0; i < n; ++i) portion += (double)(std::min(contrib[i], lv) - std::min(contrib[i], prev));
        uint32_t best = 0;
        int cnt = 0;
        bool any = false;
        for (int i = 0; i < n; ++i) {
            if (!(term.live & (1u << i)) || contrib[i] < lv) continue;
            if (!any || value[i] > best) { best = value[i]; cnt = 1; any = true; }
            else if (value[i] == best) cnt++;
        }
        if (!any) {
            // Everyone who reached this level folded: the largest live contributors take it.
            int64_t mx = -1;
            for (int i = 0; i < n; ++i) if ((term.live & (1u << i)) && contrib[i] > mx) mx = contrib[i];
            for (int i = 0; i < n; ++i)
                if ((term.live & (1u << i)) && contrib[i] == mx) { best = value[i]; }
            cnt = 0;
            for (int i = 0; i < n; ++i) if ((term.live & (1u << i)) && contrib[i] == mx) cnt++;
            for (int i = 0; i < n; ++i)
                if ((term.live & (1u << i)) && contrib[i] == mx) fin[i] += portion / cnt;
        } else {
            for (int i = 0; i < n; ++i)
                if ((term.live & (1u << i)) && contrib[i] >= lv && value[i] == best) fin[i] += portion / cnt;
        }
        prev = lv;
    }
    icm(fin, start, n, t.g.payouts, eq);
    for (int i = 0; i < n; ++i) util[i] = (float)(eq[i] * t.util_scale);
}

}  // namespace pb
