"""Print a detailed summary of the strategy stored in an exported blueprint (.strat).

    python blueprint_summary.py blueprints/p6_s100.strat
    python blueprint_summary.py blueprints/p6_s100.strat --hands 5000 --out p6_s100.txt
    python blueprint_summary.py starter/blueprints/p3_s20.strat --no-grids --hands 0

Place this file in the poker_table_v5 folder, next to play.py. It works for
any player count and stack depth, including unequal-stack profiles.

Sections of the report:

1. File: training settings, iterations, bucket level, action abstraction,
   payouts, and the share of decision-point blocks the training reached.
2. Preflop statistics by position (exact): VPIP, PFR, limp, RFI, 3-bet,
   fold to 3-bet, 4-bet, fold to 4-bet and flops seen. The preflop betting
   tree is walked with every player's range, so these are the frequencies the
   blueprint plays when all players use it. Card removal between players is
   ignored, and branches reached less often than --min-reach are skipped (the
   covered probability is printed).
3. First-in play by position (folded to the player): fold, limp, each open
   size and all-in, then 13 x 13 grids of the raise and limp frequencies.
4. Responses to a single open (call and 3-bet, by position and opener), to
   limps, to 3-bets and 4-bets, with grids for the big blind's defense.
5. How hands end preflop: folds, all-ins, players and pot size at the flop.
6. Postflop (sampled): --hands deals are played out with the blueprint for
   every player, giving c-bet, donk-bet, check-raise, fold to bets, bet sizes
   by street and showdown rates. These are estimates; the standard error is
   about 1/sqrt(opportunities).

The blueprint is read through the engine's run-time library (native/), which
is built automatically, and the card abstraction the blueprint was trained
with must be present (it is found by its hash in abstraction/ or
starter/abstraction/, or give --abstraction DIR). The probabilities are the
blueprint's own, without search or the table's deep all-in limit.
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RANKS = "23456789TJQKA"


# ---------------------------------------------------------------------------
# Engine and files

def load_native():
    """Import pokerbot/mccfr/native.py alone (the pokerbot package imports torch)."""
    path = ROOT / "pokerbot" / "mccfr" / "native.py"
    if not path.exists():
        raise SystemExit(f"{path} not found. Put blueprint_summary.py in the poker_table_v5 folder.")
    spec = importlib.util.spec_from_file_location("pb_native_summary", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_header(path: Path) -> tuple[dict, int, int, int, int]:
    with open(path, "rb") as f:
        magic, n = struct.unpack("<4sI", f.read(8))
        if magic != b"PBS5":
            raise SystemExit(f"{path} is not a v5 strategy file (.strat)")
        text = f.read(n).decode()
        iters, chunks, data = struct.unpack("<QQQ", f.read(24))
    kv = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv, iters, chunks, data, 8 + n + 24


def block_coverage(path: Path, chunks: int, offset: int) -> float:
    """Share of decision-point blocks with stored data (the rest were never reached in training)."""
    table = np.memmap(path, dtype="<u4", mode="r", offset=offset, shape=(chunks,))
    reached, step = 0, 1 << 24
    for i in range(0, chunks, step):
        reached += int(np.count_nonzero(table[i:i + step] != 0xFFFFFFFF))
    return reached / chunks if chunks else 0.0


def config_dirs() -> list[Path]:
    out = []
    for cfg in (ROOT / "cfr_config.txt", ROOT / "starter" / "cfr_config.txt"):
        if not cfg.exists():
            continue
        for line in cfg.read_text().splitlines():
            line = line.split("#", 1)[0]
            if "=" in line and line.split("=", 1)[0].strip() == "abstraction_dir":
                out.append(cfg.parent / line.split("=", 1)[1].strip())
    return out


def find_abstraction(hash_: str, strat: Path, given: str | None) -> Path:
    cands = [Path(given)] if given else []
    cands += config_dirs() + [ROOT / "abstraction", ROOT / "starter" / "abstraction",
                              strat.parent.parent / "abstraction", strat.parent / "abstraction"]
    for d in cands:
        meta = d / "meta.txt"
        if meta.exists():
            for line in meta.read_text().splitlines():
                if line.split("=", 1)[0].strip() == "hash" and line.split("=", 1)[1].strip() == hash_:
                    return d.resolve()
    raise SystemExit(f"No card abstraction with hash {hash_} found. Searched: "
                     + ", ".join(str(c) for c in cands) + ". Give one with --abstraction DIR.")


def position_names(n: int) -> list[str]:
    if n == 2:
        return ["BTN/SB", "BB"]
    late = ["UTG", "HJ", "CO"][-(n - 3):] if n > 3 else []
    if n == 4:
        late = ["CO"]
    return ["BTN", "SB", "BB"] + late


def preflop_order(n: int) -> list[int]:
    return [0, 1] if n == 2 else list(range(3, n)) + [0, 1, 2]


# ---------------------------------------------------------------------------
# Hand classes: index = 13 * row + col of the usual grid (A first);
# suited above the diagonal, offsuit below.

def class_table():
    cards, combos, names = [], [], []
    for r in range(13):
        for c in range(13):
            a, b = 12 - r, 12 - c
            if r == c:
                cards.append((a * 4 + 3, a * 4 + 2)); combos.append(6); names.append(RANKS[a] * 2)
            elif r < c:
                cards.append((a * 4 + 3, b * 4 + 3)); combos.append(4); names.append(RANKS[a] + RANKS[b] + "s")
            else:
                cards.append((b * 4 + 3, a * 4 + 2)); combos.append(12); names.append(RANKS[b] + RANKS[a] + "o")
    return cards, np.array(combos, dtype=float), names


CLASS_CARDS, COMBOS, CLASS_NAMES = class_table()


# ---------------------------------------------------------------------------
# Betting state: a port of BetState in native/pb_game.h

class Bet:
    def __init__(self, n, stacks, sb, bb, ante):
        self.n, self.bb = n, bb
        self.stack = list(stacks)
        self.street_c, self.total_c = [0] * n, [0] * n
        self.acted, self.folded = [-1] * n, [False] * n
        if ante > 0:
            for i in range(n):
                a = min(ante, self.stack[i]); self.stack[i] -= a; self.total_c[i] += a
        sbp, bbp = (0, 1) if n == 2 else (1, 2)
        self._post(sbp, sb); self._post(bbp, bb)
        self.street, self.raises, self.last_aggr, self.prev_aggr = 0, 0, -1, -1
        self.last_raise, self.full_level = bb, self.cur_bet()
        self.done, self.to_act = False, -1
        self._begin_round(0 if n == 2 else (bbp + 1) % n)

    def copy(self):
        b = Bet.__new__(Bet)
        b.__dict__ = {k: (v[:] if isinstance(v, list) else v) for k, v in self.__dict__.items()}
        return b

    def _post(self, p, amt):
        paid = min(amt, self.stack[p])
        self.stack[p] -= paid; self.street_c[p] += paid; self.total_c[p] += paid

    def pot(self): return sum(self.total_c)
    def cur_bet(self): return max(self.street_c)
    def live(self): return sum(not f for f in self.folded)
    def with_chips(self): return sum(1 for i in range(self.n) if not self.folded[i] and self.stack[i] > 0)
    def to_call(self, p): return min(self.cur_bet() - self.street_c[p], self.stack[p])

    def _needs(self, p):
        if self.folded[p] or self.stack[p] == 0:
            return False
        return self.acted[p] < 0 or self.street_c[p] < self.cur_bet()

    def _find_actor(self, start):
        if self.live() < 2 or not any(self._needs(i) for i in range(self.n)):
            return -1
        if self.with_chips() == 1:
            for i in range(self.n):
                if not self.folded[i] and self.stack[i] > 0 and self.street_c[i] >= self.cur_bet():
                    return -1
        for k in range(self.n):
            p = (start + k) % self.n
            if self._needs(p):
                return p
        return -1

    def _begin_round(self, first):
        p = self._find_actor(first)
        if p >= 0:
            self.to_act = p
        else:
            self._end_round()

    def _end_round(self):
        if self.live() < 2 or self.street == 3:
            self.done, self.to_act = True, -1
            return
        self.street += 1
        self.street_c, self.acted = [0] * self.n, [-1] * self.n
        self.prev_aggr, self.last_aggr = self.last_aggr, -1
        self.raises, self.last_raise, self.full_level = 0, self.bb, 0
        self._begin_round(1 % self.n)

    def apply(self, kind, to):
        p = self.to_act
        tc = self.to_call(p)
        if kind == 0 and tc == 0:
            kind = 1
        if kind == 0:
            self.folded[p] = True
        elif kind == 1:
            self.stack[p] -= tc; self.street_c[p] += tc; self.total_c[p] += tc
        else:
            to = min(max(to, min(self.cur_bet() + self.last_raise, self.street_c[p] + self.stack[p])),
                     self.street_c[p] + self.stack[p])
            pay, inc = to - self.street_c[p], to - self.cur_bet()
            if inc >= self.last_raise:
                self.last_raise, self.full_level = inc, to
            self.stack[p] -= pay; self.street_c[p] = to; self.total_c[p] += pay
            self.raises += 1
            self.last_aggr = p
        self.acted[p] = self.full_level
        nxt = self._find_actor((p + 1) % self.n)
        if nxt >= 0:
            self.to_act = nxt
        else:
            self._end_round()


def kind_of(code: int) -> int:
    return 0 if code == 0 else 1 if code == 1 else 2


# ---------------------------------------------------------------------------

class Blueprint:
    def __init__(self, path: Path, abstraction: str | None):
        self.path = path
        self.kv, self.iters, self.chunks, self.data_bytes, self.table_offset = read_header(path)
        kv = self.kv
        self.n = int(kv["players"])
        if "stacks" in kv:
            self.stacks = [int(float(x)) for x in kv["stacks"].split()]
        else:
            self.stacks = [int(float(kv["stack"]))] * self.n
        self.ante = int(round(float(kv.get("ante_bb", "0.1")) * 100))
        pay = kv.get("payouts", "0.65 0.35").split()
        self.payouts = (pay + ["0"] * self.n)[:self.n]
        self.pos = position_names(self.n)
        native = load_native()
        self.engine_error = native.EngineError
        self.eng = native.Engine(find_abstraction(kv["abstraction"], path, abstraction))
        self.eng.load(path)
        self.calls = 0

    def new_bet(self) -> Bet:
        return Bet(self.n, self.stacks, 50, 100, self.ante)

    def request(self, acts, hero, hole, board=()):
        return "\n".join([
            f"players {self.n}", "stacks " + " ".join(map(str, self.stacks)), f"blinds 50 100 {self.ante}",
            "payouts " + " ".join(self.payouts), "actions " + " ".join(f"{p} {k} {t}" for p, k, t in acts),
            "board " + " ".join(map(str, board)), f"hero {hero}", f"hole {hole[0]} {hole[1]}",
            "mode blueprint", "equity_samples 0", "seed 1"]) + "\n"

    def decide(self, acts, hero, hole, board=()):
        self.calls += 1
        r = self.eng.decide(self.request(acts, hero, hole, board))
        opts = r["options"]
        p = np.array([max(0.0, o["blueprint"]) for o in opts])
        s = p.sum()
        p = p / s if s > 0 else np.full(len(p), 1 / len(p))
        return opts, p

    def node(self, acts, hero):
        """Options and a 169 x actions matrix of the blueprint strategy at a preflop node."""
        rows, opts = [], None
        for hole in CLASS_CARDS:
            o, p = self.decide(acts, hero, hole)
            opts = opts or o
            rows.append(p)
        return opts, np.array(rows)


def act_name(o: dict, to_call: int) -> str:
    c = o["code"]
    if c == 0:
        return "fold"
    if c == 1:
        return "call" if to_call > 0 else "check"
    if c == 15:
        return f"all-in ({o['to'] / 100:g} bb)"
    return f"raise to {o['to'] / 100:.2f} bb"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def ratio(a: float, b: float) -> str:
    return pct(a / b) if b > 1e-12 else "-"


# ---------------------------------------------------------------------------
# Exact preflop walk

class PreflopWalk:
    def __init__(self, bp: Blueprint, min_reach: float, verbose: bool):
        self.bp, self.min_reach, self.verbose = bp, min_reach, verbose
        n = bp.n
        self.vpip, self.pfr, self.limp, self.flop = (np.zeros(n) for _ in range(4))
        self.ctx = defaultdict(lambda: np.zeros(4))   # key -> [opportunity, fold, call/check, raise]
        self.rfi = {}                                  # position -> (reach, opts, matrix, to_call)
        self.covered = self.pruned = 0.0
        self.end_fold = self.end_allin = 0.0
        self.flop_players = defaultdict(float)
        self.flop_pot = 0.0
        self.nodes = 0
        self.t0 = time.time()

    def run(self):
        n = self.bp.n
        ranges = [COMBOS / COMBOS.sum() for _ in range(n)]
        info = {"vpip": [False] * n, "pfr": [False] * n, "limped": [False] * n, "raisers": [], "limpers": 0}
        self._walk(self.bp.new_bet(), [], ranges, info, 1.0)
        if self.verbose:
            print(f"  preflop walk: {self.nodes} nodes, {time.time() - self.t0:.0f} s" + " " * 30, file=sys.stderr)

    def _leaf(self, bet: Bet, reach: float):
        self.covered += reach
        if bet.done and bet.live() < 2:
            self.end_fold += reach
        elif bet.done:
            self.end_allin += reach
        else:
            live = [i for i in range(bet.n) if not bet.folded[i]]
            self.flop_players[len(live)] += reach
            self.flop_pot += reach * bet.pot() / 100
            for i in live:
                self.flop[i] += reach

    def _walk(self, bet: Bet, acts, ranges, info, reach):
        if bet.done or bet.street > 0:
            self._leaf(bet, reach)
            return
        self.nodes += 1
        if self.verbose and self.nodes % 50 == 0:
            print(f"  preflop walk: {self.nodes} nodes, {pct(self.covered)} of deals finished, "
                  f"{time.time() - self.t0:.0f} s   ", file=sys.stderr, end="\r")
        p = bet.to_act
        opts, P = self.bp.node(acts, p)
        r = ranges[p]
        m = r.sum()
        mass = (reach / m) * (r @ P)                  # probability of reaching each child
        tc = bet.to_call(p)
        codes = [o["code"] for o in opts]
        f = sum(mass[i] for i, c in enumerate(codes) if c == 0)
        c_ = sum(mass[i] for i, c in enumerate(codes) if c == 1)
        rz = sum(mass[i] for i, c in enumerate(codes) if c >= 2)
        vol = rz + (c_ if tc > 0 else 0.0)
        if not info["vpip"][p]:
            self.vpip[p] += vol
        if not info["pfr"][p]:
            self.pfr[p] += rz
        row = np.array([reach, f, c_, rz])
        if bet.raises == 0:
            if not info["vpip"][p] and tc > 0:
                self.limp[p] += c_
            if info["limpers"] == 0 and tc > 0:
                self.rfi[p] = (reach, opts, P, tc)
            elif info["limpers"] > 0:
                self.ctx[("vs_limp", p)] += row
        elif bet.raises == 1:
            if info["limped"][p]:
                self.ctx[("limp_vs_raise", p)] += row
            elif not info["vpip"][p]:
                self.ctx[("vs_open", p, info["raisers"][0])] += row
        elif bet.raises == 2 and info["raisers"][0] == p:
            self.ctx[("vs_3bet", p)] += row
        elif bet.raises == 3 and len(info["raisers"]) > 1 and info["raisers"][1] == p:
            self.ctx[("vs_4bet", p)] += row
        for i, o in enumerate(opts):
            if mass[i] < self.min_reach:
                self.pruned += mass[i]
                continue
            nb = bet.copy()
            k = kind_of(o["code"])
            nb.apply(k, o["to"])
            nr = list(ranges)
            nr[p] = r * P[:, i]
            ni = {"vpip": info["vpip"][:], "pfr": info["pfr"][:], "limped": info["limped"][:],
                  "raisers": info["raisers"][:], "limpers": info["limpers"]}
            if k == 2:
                ni["vpip"][p] = ni["pfr"][p] = True
                ni["raisers"].append(p)
            elif k == 1 and tc > 0:
                ni["vpip"][p] = True
                if bet.raises == 0:
                    ni["limped"][p] = True
                    ni["limpers"] += 1
            self._walk(nb, acts + [(p, k, o["to"])], nr, ni, mass[i])


# ---------------------------------------------------------------------------
# Postflop sampling

class PostflopSim:
    STREETS = ["flop", "turn", "river"]

    def __init__(self, bp: Blueprint, hands: int, seed: int, verbose: bool):
        self.bp, self.hands, self.rng, self.verbose = bp, hands, random.Random(seed), verbose
        self.c = defaultdict(lambda: np.zeros(4))      # key -> [opportunity, fold, call/check, bet/raise]
        self.sizes = defaultdict(lambda: defaultdict(int))
        self.saw_flop = self.showdown = 0
        self.player_flops = self.player_sd = 0
        self.errors = 0

    def run(self):
        t0 = time.time()
        for h in range(self.hands):
            try:
                self._hand()
            except self.bp.engine_error:
                self.errors += 1
            if self.verbose and (h + 1) % 100 == 0:
                print(f"  postflop sample: {h + 1}/{self.hands} hands, {time.time() - t0:.0f} s",
                      file=sys.stderr, end="\r")
        if self.verbose:
            print(f"  postflop sample: {self.hands} hands, {time.time() - t0:.0f} s" + " " * 20, file=sys.stderr)

    def _hand(self):
        bp, n = self.bp, self.bp.n
        deck = list(range(52))
        self.rng.shuffle(deck)
        holes = [(deck[2 * i], deck[2 * i + 1]) for i in range(n)]
        board = deck[2 * n:2 * n + 5]
        bet = bp.new_bet()
        acts = []
        pre_aggr = -1
        street_seen = 0
        checked = set()
        acted_street = set()
        while not bet.done:
            s = bet.street
            if s != street_seen:
                if s == 1:
                    pre_aggr = bet.prev_aggr
                    self.saw_flop += 1
                    self.player_flops += bet.live()
                street_seen, checked, acted_street = s, set(), set()
            p = bet.to_act
            vis = board[:0 if s == 0 else s + 2]
            opts, probs = bp.decide(acts, p, holes[p], vis)
            i = self.rng.choices(range(len(opts)), weights=probs)[0]
            o = opts[i]
            k = kind_of(o["code"])
            if s > 0:
                self._record(bet, s, p, o, k, pre_aggr, checked, acted_street)
            bet.apply(k, o["to"])
            acts.append((p, k, o["to"]))
        if street_seen >= 1 and bet.live() >= 2:
            self.showdown += 1
            self.player_sd += bet.live()

    def _record(self, bet, s, p, o, k, pre_aggr, checked, acted_street):
        st = self.STREETS[s - 1]
        multi = bet.live() > 2
        cat = 1 if k == 0 else 2 if k == 1 else 3
        row = np.zeros(4); row[0] = 1; row[cat] = 1
        if bet.raises == 0:
            self.c[(st, "first to bet", multi)] += row
            if s == 1 and p == pre_aggr:
                self.c[("cbet", multi)] += row
            elif s == 1 and pre_aggr >= 0 and not bet.folded[pre_aggr] and pre_aggr not in acted_street:
                self.c[("donk", multi)] += row
            if k == 1:
                checked.add(p)
        elif bet.raises == 1:
            self.c[(st, "facing bet", multi)] += row
            if s == 1 and bet.last_aggr == pre_aggr:
                self.c[("vs_cbet", multi)] += row
            if p in checked:
                self.c[(st, "check-raise", multi)] += row
        else:
            self.c[(st, "facing raise", multi)] += row
        if k == 2:
            label = "all-in" if o["code"] == 15 else o["label"]
            self.sizes[(st, "bet" if bet.raises == 0 else "raise")][label] += 1
        acted_street.add(p)


# ---------------------------------------------------------------------------
# Report

def grid(P_sel: np.ndarray, title: str) -> list[str]:
    out = [title, "       " + "".join(f"{r:>4}" for r in RANKS[::-1])]
    for r in range(13):
        cells = []
        for c in range(13):
            v = round(100 * P_sel[13 * r + c])
            cells.append(f"{'.' if v == 0 else v:>4}")
        out.append(f"    {RANKS[12 - r]}  " + "".join(cells))
    return out


def range_text(freq: np.ndarray, lo: float = 0.5) -> str:
    """Hands played at least `lo` of the time, strongest classes first (by grid order)."""
    names = [CLASS_NAMES[i] for i in range(169) if freq[i] >= lo]
    return ", ".join(names) if names else "(none)"


def report(bp: Blueprint, walk: PreflopWalk, sim: PostflopSim | None, args) -> str:
    kv, n, pos = bp.kv, bp.n, bp.pos
    L = []
    w = L.append

    # 1. File
    stacks_bb = [s / 100 for s in bp.stacks]
    stack_txt = f"{stacks_bb[0]:g} bb each" if len(set(stacks_bb)) == 1 else \
        ", ".join(f"{pos[i]} {stacks_bb[i]:g}" for i in range(n)) + " bb"
    w(f"BLUEPRINT SUMMARY: {bp.path.name}")
    w("=" * 78)
    w(f"Players {n}; stacks {stack_txt}; blinds 0.5/1 bb, ante {bp.ante / 100:g} bb per player.")
    w(f"Iterations trained: {bp.iters:,}.  File size: {bp.path.stat().st_size / 2**20:,.1f} MiB.")
    w(f"Buckets per street (preflop, flop, turn, river): {kv.get('buckets', '?')}; bucket level {kv.get('level', '?')}.")
    w(f"ICM payouts: {kv.get('payouts', '?')}.")
    w("Action abstraction (raise sizes as fractions of the pot after calling; all-in is always allowed):")
    for k, label in [("preflop_open", "preflop open"), ("preflop_3bet", "preflop 3-bet"),
                     ("preflop_later", "preflop 4-bet+"), ("postflop_bet", "postflop bet, heads-up"),
                     ("postflop_raise", "postflop raise, heads-up"), ("postflop_later", "postflop re-raise"),
                     ("multiway_bet", "postflop bet, multiway"), ("multiway_raise", "postflop raise, multiway")]:
        w(f"    {label:<26} {kv.get(k, '').strip() or 'all-in only'}")
    w(f"    max raises: preflop {kv.get('max_raises_preflop', '?')}, postflop {kv.get('max_raises_postflop', '?')}, "
      f"multiway {kv.get('max_raises_multiway', '?')}; sizes within {kv.get('allin_snap', '?')} of the stack become all-in")
    if not args.no_coverage:
        cov = block_coverage(bp.path, bp.chunks, bp.table_offset)
        w(f"Decision-point blocks reached in training: {pct(cov)} of {bp.chunks:,} "
          "(unreached blocks play uniformly at random; preflop is always stored).")
    w("")

    # 2. Preflop statistics
    order = preflop_order(n)
    w("PREFLOP STATISTICS BY POSITION (exact walk of the preflop tree)")
    w("-" * 78)
    w(f"Deals covered: {pct(walk.covered, 2)}; branches below {args.min_reach:g} skipped: {pct(walk.pruned, 2)}.")
    w("VPIP, PFR, limp and flops seen are shares of all hands dealt. The others are")
    w("conditional on the situation: RFI when folded to the player; 3-bet when facing a")
    w("single raise; F3bet and 4-bet after making the first raise and facing a 3-bet;")
    w("F4bet after 3-betting and facing a 4-bet. Flop: share of hands dealt in which the")
    w("player sees the flop.")
    w("")
    hdr = f"{'':<7}{'VPIP':>7}{'PFR':>7}{'Limp':>7}{'RFI':>7}{'3-bet':>7}{'F3bet':>7}{'4-bet':>7}{'F4bet':>7}{'Flop':>7}"
    w(hdr)
    for p in order:
        rfi = "-"
        if p in walk.rfi:
            reach, opts, P, tc = walk.rfi[p]
            dist = (COMBOS / COMBOS.sum()) @ P
            rfi = pct(sum(dist[i] for i, o in enumerate(opts) if o["code"] >= 2))
        vo = sum((v for k, v in walk.ctx.items() if k[0] == "vs_open" and k[1] == p), np.zeros(4))
        v3, v4 = walk.ctx[("vs_3bet", p)], walk.ctx[("vs_4bet", p)]
        w(f"{pos[p]:<7}{pct(walk.vpip[p]):>7}{pct(walk.pfr[p]):>7}{pct(walk.limp[p]):>7}{rfi:>7}"
          f"{ratio(vo[3], vo[0]):>7}{ratio(v3[1], v3[0]):>7}{ratio(v3[3], v3[0]):>7}{ratio(v4[1], v4[0]):>7}"
          f"{pct(walk.flop[p]):>7}")
    w("")

    # 3. First-in play
    w("FIRST-IN PLAY (all players before have folded)")
    w("-" * 78)
    for p in order:
        if p not in walk.rfi:
            continue
        reach, opts, P, tc = walk.rfi[p]
        dist = (COMBOS / COMBOS.sum()) @ P
        parts = [f"{act_name(o, tc).replace('call', 'limp' if pos[p] != 'SB' else 'complete')} {pct(dist[i])}"
                 for i, o in enumerate(opts) if dist[i] >= 0.0005]
        w(f"{pos[p]:<7}(folded to {pct(reach)} of deals)  " + ", ".join(parts))
    if not args.no_grids:
        for p in order:
            if p not in walk.rfi:
                continue
            reach, opts, P, tc = walk.rfi[p]
            raise_p = sum(P[:, i] for i, o in enumerate(opts) if o["code"] >= 2)
            call_p = sum((P[:, i] for i, o in enumerate(opts) if o["code"] == 1), np.zeros(169))
            w("")
            L.extend(grid(raise_p, f"{pos[p]} first in: raise (any size), % of the time; suited above the diagonal"))
            w(f"    Raised at least half the time: {range_text(raise_p)}")
            if (COMBOS / COMBOS.sum()) @ call_p >= 0.01:
                w("")
                L.extend(grid(call_p, f"{pos[p]} first in: {'complete' if pos[p] == 'SB' else 'limp'}, % of the time"))
            sizes = [(i, o) for i, o in enumerate(opts) if o["code"] >= 2]
            if args.size_grids and len(sizes) > 1:
                for i, o in sizes:
                    w("")
                    L.extend(grid(P[:, i], f"{pos[p]} first in: {act_name(o, tc)}"))
    w("")

    # 4. Responses
    w("RESPONSES TO A SINGLE OPEN RAISE (any size, any callers in between)")
    w("-" * 78)
    w("Each cell: call / 3-bet (incl. all-in), as % of the times the player faces that open.")
    openers = [p for p in order[:-1]]
    w(f"{'vs opener':<10}" + "".join(f"{pos[o]:>14}" for o in openers))
    for d in order:
        cells = []
        for o in openers:
            v = walk.ctx.get(("vs_open", d, o))
            cells.append(f"{ratio(v[2], v[0]) + ' / ' + ratio(v[3], v[0]):>14}" if v is not None and v[0] > 1e-9
                         else f"{'':>14}")
        if any(c.strip() for c in cells):
            w(f"{pos[d]:<10}" + "".join(cells))
    w("")
    w("Other situations: fold / call (or check) / raise, % of opportunities; 'reached' is")
    w("the share of all deals in which the situation arises.")
    w(f"{'':<7}{'situation':<28}{'reached':>9}{'fold':>9}{'call':>9}{'raise':>9}")
    for key, label in [("vs_limp", "facing limp(s)"), ("limp_vs_raise", "limped, facing raise"),
                       ("vs_3bet", "first raiser, facing 3-bet"), ("vs_4bet", "3-bet, facing 4-bet")]:
        for p in order:
            v = walk.ctx.get((key, p))
            if v is None or v[0] < 1e-5:
                continue
            w(f"{pos[p]:<7}{label:<28}{pct(v[0], 2):>9}{ratio(v[1], v[0]):>9}{ratio(v[2], v[0]):>9}{ratio(v[3], v[0]):>9}")
    w("")

    if not args.no_grids:
        bb = 1 if n == 2 else 2
        pairs = []
        if n == 2:
            pairs = [(0, 1)]
        else:
            first = order[0]
            pairs = [(first, bb)] + ([(0, bb)] if first != 0 else []) + [(1, bb)]
            if n >= 3 and first != 0:
                pairs.append((0, 1))
        for opener, defender in pairs:
            if opener not in walk.rfi:
                continue
            reach, opts, P, tc = walk.rfi[opener]
            dist = (COMBOS / COMBOS.sum()) @ P
            raises = [(dist[i], i, o) for i, o in enumerate(opts) if o["code"] >= 2 and o["code"] != 15]
            if not raises:
                raises = [(dist[i], i, o) for i, o in enumerate(opts) if o["code"] >= 2]
            if not raises:
                continue
            _, _, o = max(raises, key=lambda t: t[0])
            acts = [(q, 0, 0) for q in order[:order.index(opener)]] + [(opener, 2, o["to"])]
            acts += [(q, 0, 0) for q in order[order.index(opener) + 1:order.index(defender)]]
            dopts, DP = bp.node(acts, defender)
            ddist = (COMBOS / COMBOS.sum()) @ DP
            w(f"{pos[defender]} vs {pos[opener]} open to {o['to'] / 100:.2f} bb (others fold): " +
              ", ".join(f"{act_name(x, 1)} {pct(ddist[i])}" for i, x in enumerate(dopts) if ddist[i] >= 0.0005))
            call_p = sum((DP[:, i] for i, x in enumerate(dopts) if x["code"] == 1), np.zeros(169))
            raise_p = sum((DP[:, i] for i, x in enumerate(dopts) if x["code"] >= 2), np.zeros(169))
            L.extend(grid(call_p, f"{pos[defender]} vs {pos[opener]} open: call, % of the time"))
            w("")
            L.extend(grid(raise_p, f"{pos[defender]} vs {pos[opener]} open: 3-bet (any size, incl. all-in), % of the time"))
            w(f"    3-bet at least half the time: {range_text(raise_p)}")
            w("")

    # 5. How hands end preflop
    w("HOW HANDS LEAVE PREFLOP")
    w("-" * 78)
    tot = walk.covered
    flop = sum(walk.flop_players.values())
    w(f"Won without a flop (everyone folds): {ratio(walk.end_fold, tot)}")
    w(f"All-in preflop (no more decisions):  {ratio(walk.end_allin, tot)}")
    w(f"Flop seen:                           {ratio(flop, tot)}"
      + (f", average pot {walk.flop_pot / flop:.1f} bb" if flop > 0 else ""))
    for k in sorted(walk.flop_players):
        w(f"    {k} players see the flop: {ratio(walk.flop_players[k], tot)} of deals")
    w("")

    # 6. Postflop
    if sim is not None and sim.hands > 0:
        w(f"POSTFLOP (sampled from {sim.hands:,} deals played with the blueprint; seed {args.seed})")
        w("-" * 78)
        if sim.errors:
            w(f"({sim.errors} deals skipped after an engine error.)")
        w(f"Deals reaching the flop: {ratio(sim.saw_flop, sim.hands)}. Showdown, of deals that reach the flop: "
          f"{ratio(sim.showdown, sim.saw_flop)}.")
        w(f"Players who see the flop and reach showdown (WTSD): {ratio(sim.player_sd, sim.player_flops)}.")
        w("")
        w("Each row: opportunities in the sample, then fold / check-or-call / bet-or-raise.")
        w(f"{'situation':<36}{'count':>7}{'fold':>9}{'call':>9}{'bet/raise':>11}")

        def row(label, key):
            v = sim.c.get(key)
            if v is None or v[0] == 0:
                return
            w(f"{label:<36}{int(v[0]):>7}{ratio(v[1], v[0]):>9}{ratio(v[2], v[0]):>9}{ratio(v[3], v[0]):>11}")

        for multi, tag in ((False, "heads-up"), (True, "multiway")):
            row(f"flop c-bet, {tag}", ("cbet", multi))
            row(f"flop donk bet, {tag}", ("donk", multi))
            row(f"facing flop c-bet, {tag}", ("vs_cbet", multi))
        for st in PostflopSim.STREETS:
            for multi, tag in ((False, "heads-up"), (True, "multiway")):
                row(f"{st}: first to bet, {tag}", (st, "first to bet", multi))
                row(f"{st}: facing bet, {tag}", (st, "facing bet", multi))
                row(f"{st}: checked, facing bet, {tag}", (st, "check-raise", multi))
                row(f"{st}: facing raise, {tag}", (st, "facing raise", multi))
        w("")
        w("Bet and raise sizes chosen (share of that street's bets or raises):")
        for st in PostflopSim.STREETS:
            for kind in ("bet", "raise"):
                d = sim.sizes.get((st, kind))
                if not d:
                    continue
                t = sum(d.values())
                w(f"    {st} {kind}s ({t}): " + ", ".join(f"{k} {ratio(v, t)}" for k, v in
                                                       sorted(d.items(), key=lambda kv_: -kv_[1])))
        w("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strat", help="exported blueprint, e.g. blueprints/p6_s100.strat")
    ap.add_argument("--abstraction", default=None, help="card abstraction folder (default: found by hash)")
    ap.add_argument("--min-reach", type=float, default=5e-5,
                    help="skip preflop branches reached less often than this (default 0.00005)")
    ap.add_argument("--hands", type=int, default=20000, help="deals to sample for postflop statistics (0 = none)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-grids", action="store_true", help="leave out the 13 x 13 hand grids")
    ap.add_argument("--size-grids", action="store_true", help="also print one grid per open size")
    ap.add_argument("--no-coverage", action="store_true", help="skip the scan of reached blocks (large files)")
    ap.add_argument("--out", default=None, help="also write the report to this file")
    ap.add_argument("-q", "--quiet", action="store_true", help="no progress messages")
    args = ap.parse_args()

    path = Path(args.strat).resolve()
    if not path.exists():
        raise SystemExit(f"{path} not found")
    bp = Blueprint(path, args.abstraction)
    verbose = not args.quiet
    if verbose:
        print(f"Reading {path.name} ({bp.n} players)...", file=sys.stderr)
    walk = PreflopWalk(bp, args.min_reach, verbose)
    walk.run()
    sim = None
    if args.hands > 0:
        sim = PostflopSim(bp, args.hands, args.seed, verbose)
        sim.run()
    text = report(bp, walk, sim, args)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        if verbose:
            print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
