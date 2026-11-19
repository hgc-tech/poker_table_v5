// Core utilities: cards, 7-card evaluator, hole-card indexing, suit
// isomorphism, random numbers, configuration files and ICM.
//
// A card is an int in [0, 52): rank = card / 4 (0 = deuce ... 12 = ace),
// suit = card % 4, the same encoding as pokerbot/cards.py.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
static inline int pb_popcount(uint32_t x) { return (int)__popcnt(x); }
static inline int pb_popcount64(uint64_t x) { return (int)__popcnt64(x); }
static inline int pb_hibit(uint32_t x) { unsigned long i; _BitScanReverse(&i, x); return (int)i; }
static inline int pb_lobit64(uint64_t x) { unsigned long i; _BitScanForward64(&i, x); return (int)i; }
#else
static inline int pb_popcount(uint32_t x) { return __builtin_popcount(x); }
static inline int pb_popcount64(uint64_t x) { return __builtin_popcountll(x); }
static inline int pb_hibit(uint32_t x) { return 31 - __builtin_clz(x); }
static inline int pb_lobit64(uint64_t x) { return __builtin_ctzll(x); }
#endif

namespace pb {

constexpr int NUM_HOLES = 1326;
constexpr int MAX_PLAYERS = 6;

inline int rank_of(int c) { return c >> 2; }
inline int suit_of(int c) { return c & 3; }

inline std::string card_name(int c) {
    static const char* R = "23456789TJQKA";
    static const char* S = "cdhs";
    std::string s;
    s += R[rank_of(c)];
    s += S[suit_of(c)];
    return s;
}

// ---------------------------------------------------------------------------
// Random numbers (xoshiro256**)
struct Rng {
    uint64_t s[4];
    explicit Rng(uint64_t seed = 1) { reseed(seed); }
    void reseed(uint64_t seed) {
        uint64_t z = seed + 0x9E3779B97F4A7C15ull;
        for (int i = 0; i < 4; ++i) {
            z += 0x9E3779B97F4A7C15ull;
            uint64_t x = z;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
            s[i] = x ^ (x >> 31);
        }
    }
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    inline double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    inline uint32_t below(uint32_t n) { return (uint32_t)(((next() >> 32) * (uint64_t)n) >> 32); }
};

// ---------------------------------------------------------------------------
// Hole-card indexing: the 1326 two-card combinations, a < b.
struct Holes {
    int16_t index[52][52];
    uint8_t c1[NUM_HOLES], c2[NUM_HOLES];
    uint64_t mask[NUM_HOLES];
    uint8_t cls[NUM_HOLES];          // preflop class 0..168
    Holes() {
        int k = 0;
        for (int a = 0; a < 52; ++a)
            for (int b = a + 1; b < 52; ++b) {
                index[a][b] = index[b][a] = (int16_t)k;
                c1[k] = (uint8_t)a; c2[k] = (uint8_t)b;
                mask[k] = (1ull << a) | (1ull << b);
                cls[k] = (uint8_t)preflop_class(a, b);
                ++k;
            }
        for (int a = 0; a < 52; ++a) index[a][a] = -1;
    }
    // 0..12 pairs (22..AA), 13..90 suited, 91..168 offsuit.
    static int preflop_class(int a, int b) {
        int r1 = rank_of(a), r2 = rank_of(b);
        if (r1 < r2) std::swap(r1, r2);
        if (r1 == r2) return r1;
        int tri = r1 * (r1 - 1) / 2 + r2;   // r1 > r2: 0..77
        return (suit_of(a) == suit_of(b) ? 13 : 91) + tri;
    }
};
inline const Holes& holes() { static Holes h; return h; }

inline std::string class_name(int cls) {
    static const char* R = "23456789TJQKA";
    std::string s;
    if (cls < 13) { s += R[cls]; s += R[cls]; return s; }
    bool suited = cls < 91;
    int tri = cls - (suited ? 13 : 91);
    int r1 = 1;
    while ((r1 + 1) * r1 / 2 <= tri) ++r1;
    int r2 = tri - r1 * (r1 - 1) / 2;
    s += R[r1]; s += R[r2]; s += suited ? 's' : 'o';
    return s;
}

// ---------------------------------------------------------------------------
// 7-card evaluator. Larger values are better hands.
struct Eval {
    uint64_t sbit[52];   // card -> bit in suit-major layout (suit*16 + rank)
    Eval() { for (int c = 0; c < 52; ++c) sbit[c] = 1ull << (suit_of(c) * 16 + rank_of(c)); }
};
inline const Eval& evaltab() { static Eval e; return e; }

inline int straight_high(uint32_t m) {
    uint32_t mm = (m << 1) | ((m >> 12) & 1);
    uint32_t s = mm & (mm >> 1) & (mm >> 2) & (mm >> 3) & (mm >> 4);
    return s ? pb_hibit(s) + 3 : -1;
}
inline uint32_t keep_top(uint32_t m, int k) {
    while (pb_popcount(m) > k) m &= m - 1;
    return m;
}

// sm: suit-major mask (suit*16 + rank). Returns category<<26 | x<<13 | kickers.
inline uint32_t eval_suitmask(uint64_t sm) {
    uint32_t s0 = sm & 0x1FFF, s1 = (sm >> 16) & 0x1FFF, s2 = (sm >> 32) & 0x1FFF, s3 = (sm >> 48) & 0x1FFF;
    uint32_t su[4] = {s0, s1, s2, s3};
    for (int i = 0; i < 4; ++i)
        if (pb_popcount(su[i]) >= 5) {
            int sf = straight_high(su[i]);
            if (sf >= 0) return (8u << 26) | ((uint32_t)sf << 13);
            return (5u << 26) | keep_top(su[i], 5);
        }
    uint32_t ranks = s0 | s1 | s2 | s3;
    uint32_t quads = s0 & s1 & s2 & s3;
    uint32_t three = ((s0 & s1) & (s2 | s3)) | ((s2 & s3) & (s0 | s1));
    uint32_t two = (s0 & s1) | (s0 & s2) | (s0 & s3) | (s1 & s2) | (s1 & s3) | (s2 & s3);
    if (quads) {
        int q = pb_hibit(quads);
        return (7u << 26) | ((uint32_t)q << 13) | keep_top(ranks & ~(1u << q), 1);
    }
    if (three) {
        int t = pb_hibit(three);
        uint32_t rest = two & ~(1u << t);
        if (rest) return (6u << 26) | ((uint32_t)(t * 16 + pb_hibit(rest)) << 13);
    }
    int st = straight_high(ranks);
    if (st >= 0) return (4u << 26) | ((uint32_t)st << 13);
    if (three) {
        int t = pb_hibit(three);
        return (3u << 26) | ((uint32_t)t << 13) | keep_top(ranks & ~(1u << t), 2);
    }
    if (two) {
        int p1 = pb_hibit(two);
        uint32_t rem = two & ~(1u << p1);
        if (rem) {
            int p2 = pb_hibit(rem);
            return (2u << 26) | ((uint32_t)(p1 * 16 + p2) << 13) |
                   keep_top(ranks & ~(1u << p1) & ~(1u << p2), 1);
        }
        return (1u << 26) | ((uint32_t)p1 << 13) | keep_top(ranks & ~(1u << p1), 3);
    }
    return keep_top(ranks, 5);
}
inline uint64_t to_suitmask(uint64_t cardmask) {
    const Eval& e = evaltab();
    uint64_t sm = 0;
    while (cardmask) { int c = pb_lobit64(cardmask); cardmask &= cardmask - 1; sm |= e.sbit[c]; }
    return sm;
}
inline uint32_t eval_cards(const int* cards, int n) {
    const Eval& e = evaltab();
    uint64_t sm = 0;
    for (int i = 0; i < n; ++i) sm |= e.sbit[cards[i]];
    return eval_suitmask(sm);
}

// Equity (win + tie/2) of every hole against one uniformly random opponent
// hand on a complete 5-card board. eq[h] = -1 for holes that touch the board.
inline void river_equities(const int* board5, float* eq) {
    const Holes& H = holes();
    const Eval& E = evaltab();
    uint64_t bmask = 0, bsm = 0;
    for (int i = 0; i < 5; ++i) { bmask |= 1ull << board5[i]; bsm |= E.sbit[board5[i]]; }
    static thread_local std::vector<std::pair<uint32_t, int16_t>> v;
    v.clear();
    for (int h = 0; h < NUM_HOLES; ++h) {
        if (H.mask[h] & bmask) { eq[h] = -1.f; continue; }
        v.push_back({eval_suitmask(bsm | E.sbit[H.c1[h]] | E.sbit[H.c2[h]]), (int16_t)h});
    }
    std::sort(v.begin(), v.end());
    int less_tot = 0;
    int less_card[52] = {0};
    size_t i = 0;
    while (i < v.size()) {
        size_t j = i;
        int eq_card[52] = {0};
        while (j < v.size() && v[j].first == v[i].first) {
            int h = v[j].second;
            eq_card[H.c1[h]]++; eq_card[H.c2[h]]++;
            ++j;
        }
        int eq_tot = (int)(j - i);
        for (size_t k = i; k < j; ++k) {
            int h = v[k].second, a = H.c1[h], b = H.c2[h];
            int less = less_tot - less_card[a] - less_card[b];
            int ties = eq_tot - eq_card[a] - eq_card[b] + 1;
            eq[h] = (float)((less + 0.5 * ties) / 990.0);
        }
        for (size_t k = i; k < j; ++k) {
            int h = v[k].second;
            less_card[H.c1[h]]++; less_card[H.c2[h]]++;
        }
        less_tot += eq_tot;
        i = j;
    }
}

// ---------------------------------------------------------------------------
// Suit isomorphism for boards.
struct SuitPerms {
    uint8_t p[24][4];
    SuitPerms() {
        int a[4] = {0, 1, 2, 3}, k = 0;
        do { for (int i = 0; i < 4; ++i) p[k][i] = (uint8_t)a[i]; ++k; } while (std::next_permutation(a, a + 4));
    }
};
inline const SuitPerms& suitperms() { static SuitPerms s; return s; }
inline int apply_perm(int c, const uint8_t* perm) { return (c & ~3) | perm[c & 3]; }

// Canonical form of a board: the suit permutation (first in fixed order)
// that gives the smallest card mask.
inline uint64_t canon_board(const int* b, int n, int* perm_out) {
    const SuitPerms& P = suitperms();
    uint64_t best = ~0ull;
    int bp = 0;
    for (int k = 0; k < 24; ++k) {
        uint64_t m = 0;
        for (int i = 0; i < n; ++i) m |= 1ull << apply_perm(b[i], P.p[k]);
        if (m < best) { best = m; bp = k; }
    }
    if (perm_out) *perm_out = bp;
    return best;
}

// Enumerate the canonical boards of n cards, in a fixed order.
struct BoardIndex {
    int n = 0;
    std::vector<uint64_t> boards;              // canonical masks
    std::vector<uint32_t> weight;              // number of raw boards per class
    std::unordered_map<uint64_t, int> id;
    void build(int ncards) {
        n = ncards;
        boards.clear(); weight.clear(); id.clear();
        int c[5];
        std::function<void(int, int)> rec = [&](int depth, int start) {
            if (depth == n) {
                uint64_t m = canon_board(c, n, nullptr);
                auto it = id.find(m);
                if (it == id.end()) {
                    id[m] = (int)boards.size();
                    boards.push_back(m);
                    weight.push_back(1);
                } else {
                    weight[it->second]++;
                }
                return;
            }
            for (int x = start; x < 52; ++x) { c[depth] = x; rec(depth + 1, x + 1); }
        };
        rec(0, 0);
    }
    void cards_of(int i, int* out) const {
        uint64_t m = boards[i];
        int k = 0;
        while (m) { out[k++] = pb_lobit64(m); m &= m - 1; }
    }
    // Lookup: board -> canonical id, and the suit permutation used.
    inline int lookup(const int* b, int* perm) const {
        uint64_t m = canon_board(b, n, perm);
        auto it = id.find(m);
        return it == id.end() ? -1 : it->second;
    }
};

// ---------------------------------------------------------------------------
// Configuration files: "key = value value ..." lines, '#' comments.
struct KV {
    std::map<std::string, std::vector<std::string>> v;
    std::string text;   // normalized text, used for hashing
    static std::string trim(const std::string& s) {
        size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
        return a == std::string::npos ? "" : s.substr(a, b - a + 1);
    }
    void parse(const std::string& src) {
        std::istringstream in(src);
        std::string line;
        while (std::getline(in, line)) {
            size_t h = line.find('#');
            if (h != std::string::npos) line = line.substr(0, h);
            size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string k = trim(line.substr(0, eq));
            std::istringstream vs(line.substr(eq + 1));
            std::vector<std::string> vals;
            std::string t;
            while (vs >> t) {
                for (char& ch : t) if (ch == ',') ch = ' ';
                std::istringstream ts(t);
                std::string u;
                while (ts >> u) vals.push_back(u);
            }
            if (!k.empty()) v[k] = vals;
        }
        text.clear();
        for (auto& kv : v) {
            text += kv.first + "=";
            for (auto& x : kv.second) text += x + " ";
            text += "\n";
        }
    }
    void load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot open config " + path);
        std::stringstream ss; ss << f.rdbuf();
        parse(ss.str());
    }
    bool has(const std::string& k) const { return v.count(k) > 0; }
    double num(const std::string& k, double def) const {
        auto it = v.find(k);
        return (it == v.end() || it->second.empty()) ? def : std::atof(it->second[0].c_str());
    }
    std::vector<double> list(const std::string& k, std::vector<double> def = {}) const {
        auto it = v.find(k);
        if (it == v.end()) return def;
        std::vector<double> out;
        for (auto& s : it->second) if (s != "none" && s != "-") out.push_back(std::atof(s.c_str()));
        return out;
    }
    // Per-table-size overrides: "key@6 = ..." wins over "key = ..." for 6 players.
    std::string pkey(const std::string& k, int players) const {
        std::string o = k + "@" + std::to_string(players);
        return v.count(o) ? o : k;
    }
    std::vector<double> list_p(const std::string& k, int players, std::vector<double> def = {}) const {
        return list(pkey(k, players), def);
    }
    double num_p(const std::string& k, int players, double def) const { return num(pkey(k, players), def); }
    std::string str(const std::string& k, const std::string& def) const {
        auto it = v.find(k);
        return (it == v.end() || it->second.empty()) ? def : it->second[0];
    }
};

inline uint64_t fnv1a(const std::string& s, uint64_t h = 1469598103934665603ull) {
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}

// ---------------------------------------------------------------------------
// ICM (Malmuth-Harville). stacks: chips of every player still in the
// tournament; start: stacks at the start of the hand (orders players who bust
// in the same hand: the smaller starting stack finishes lower, equal stacks
// share). payouts[k]: prize for place k+1 (missing places pay 0).
inline void icm_rec(const double* st, int n, uint32_t mask, double total, double prob,
                    int place, int paid, const double* pay, double* eq) {
    if (place >= paid || mask == 0) return;
    for (int i = 0; i < n; ++i) {
        if (!(mask & (1u << i)) || st[i] <= 0) continue;
        double p = prob * st[i] / total;
        eq[i] += p * pay[place];
        icm_rec(st, n, mask & ~(1u << i), total - st[i], p, place + 1, paid, pay, eq);
    }
}

inline void icm(const double* stacks, const double* start, int n,
                const std::vector<double>& payouts, double* eq) {
    double pay[MAX_PLAYERS] = {0};
    for (int k = 0; k < n && k < (int)payouts.size(); ++k) pay[k] = payouts[k];
    for (int i = 0; i < n; ++i) eq[i] = 0;
    int alive = 0;
    uint32_t mask = 0;
    double total = 0;
    for (int i = 0; i < n; ++i)
        if (stacks[i] > 0) { alive++; mask |= 1u << i; total += stacks[i]; }
    // Busted players take the bottom places, ordered by starting stack.
    std::vector<int> bust;
    for (int i = 0; i < n; ++i) if (stacks[i] <= 0) bust.push_back(i);
    std::sort(bust.begin(), bust.end(), [&](int a, int b) { return start[a] > start[b]; });
    size_t i = 0;
    int place = alive;
    while (i < bust.size()) {
        size_t j = i;
        while (j < bust.size() && start[bust[j]] == start[bust[i]]) ++j;
        double sum = 0;
        for (size_t k = i; k < j; ++k) sum += pay[place + (int)(k - i)];
        for (size_t k = i; k < j; ++k) eq[bust[k]] = sum / (double)(j - i);
        place += (int)(j - i);
        i = j;
    }
    int paid = 0;
    for (int k = 0; k < alive; ++k) if (pay[k] != 0) paid = k + 1;
    if (alive == 1) { for (int k = 0; k < n; ++k) if (mask & (1u << k)) eq[k] = pay[0]; return; }
    icm_rec(stacks, n, mask, total, 1.0, 0, paid, pay, eq);
}

}  // namespace pb
