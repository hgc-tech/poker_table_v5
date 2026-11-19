"""Tests for the MCCFR bots.

    python tests/test_mccfr.py            # all tests
    python tests/test_mccfr.py hijack     # tests whose name contains "hijack"

The strategy tests use your trained blueprints (cfr_config.txt, or the
config named by the PB_CONFIG environment variable). Train for a while first:
an untrained blueprint fails the strategy tests. They check the complaints about the
earlier bots: too tight from the hijack, folding KQo on the button, and not
calling off a stack with enough equity.
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokerbot import Table                                     # noqa: E402
from pokerbot.cards import parse_card                          # noqa: E402
from pokerbot.mccfr import CfrConfig, SearchSettings, default_config, mccfr_bots  # noqa: E402
from pokerbot.mccfr.native import EXE, build                   # noqa: E402

CFG = CfrConfig(os.environ["PB_CONFIG"]) if os.environ.get("PB_CONFIG") else default_config()
ENGINE = None
# 6-handed positions: 0 BTN, 1 SB, 2 BB, 3 UTG, 4 HJ, 5 CO.
BTN, SB, BB, UTG, HJ, CO = range(6)
RANKS = "23456789TJQKA"


def engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = CFG.engine()
    return ENGINE


def cards(s: str) -> list[int]:
    return [parse_card(x) for x in s.split()]


def class_reps():
    """One representative combo per starting-hand class, with its combo count."""
    out = []
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i > j:
                out.append((cards(f"{r1}s {r2}s"), 4))          # suited
                out.append((cards(f"{r1}s {r2}h"), 12))         # offsuit
            elif i == j:
                out.append((cards(f"{r1}s {r1}h"), 6))
    return out


def request(players, stacks_bb, actions="", hero=0, hole="As Kd", board="", mode="blueprint",
            ms=400, extra="", payouts=None):
    if not isinstance(stacks_bb, (list, tuple)):
        stacks_bb = [stacks_bb] * players
    payouts = payouts or CFG.payouts
    h = cards(hole) if isinstance(hole, str) else hole
    lines = [f"players {players}", "stacks " + " ".join(str(int(s * 100)) for s in stacks_bb),
             "blinds 50 100 10", "payouts " + " ".join(map(str, payouts[:players])),
             f"actions {actions}", "board " + " ".join(str(c) for c in (cards(board) if board else [])),
             f"hero {hero}", f"hole {h[0]} {h[1]}", f"mode {mode}", f"search_ms {ms}", "threads 2",
             "seed 7", "equity_samples 500"]
    return "\n".join(lines) + "\n" + extra


def p_raise(res) -> float:
    return sum(o["prob"] for o in res["options"] if o["code"] >= 2)


def p_call(res) -> float:
    return sum(o["prob"] for o in res["options"] if o["code"] == 1)


def first_in_raise(players, stack, hero, folds, mode="blueprint") -> float:
    """Share of all starting hands that raise first in from `hero` after `folds` fold."""
    acts = " ".join(f"{p} 0 0" for p in folds)
    tot = w = 0.0
    for hole, n in class_reps():
        res = engine().decide(request(players, stack, acts, hero, hole, mode=mode))
        tot += n * p_raise(res)
        w += n
    return tot / w


# ---------------------------------------------------------------------------
def test_action_abstraction_has_fewer_sizes_after_first_raise():
    e = engine()
    open_ = e.decide(request(6, 20, "3 0 0", HJ, "As Kd"))
    sizes_open = [o for o in open_["options"] if 2 <= o["code"] < 15]
    three = e.decide(request(6, 20, "3 0 0 4 2 250", CO, "As Kd"))
    sizes_3bet = [o for o in three["options"] if 2 <= o["code"] < 15]
    assert len(sizes_open) > len(sizes_3bet), (sizes_open, sizes_3bet)
    assert any(o["code"] == 15 for o in three["options"]), "all-in must always be available"


def test_hijack_opens_a_reasonable_range():
    """Complaint: the old bots were far too tight from the hijack. Equilibrium
    six-handed charts open roughly 18 to 25 percent from the hijack at 20 to
    40 big blinds (a little less under ICM)."""
    for stack in (15, 20, 40):
        f = first_in_raise(6, stack, HJ, [UTG])
        print(f"  hijack first-in raise at {stack} bb: {f:.1%}")
        assert 0.15 <= f <= 0.40, (stack, f)


def test_positions_widen_toward_the_button():
    rates = [first_in_raise(6, 20, p, list(range(3, p)) if p >= 3 else [3, 4, 5])
             for p in (UTG, HJ, CO, BTN)]
    print("  first-in raise UTG/HJ/CO/BTN at 20 bb:", " ".join(f"{r:.0%}" for r in rates))
    assert rates[0] < rates[1] < rates[2] < rates[3], rates


def test_button_never_folds_kqo_first_in():
    """Complaint: the old bots folded KQo on the button."""
    for stack in (10, 15, 20, 40):
        res = engine().decide(request(6, stack, "3 0 0 4 0 0 5 0 0", BTN, "Kd Qc"))
        fold = sum(o["prob"] for o in res["options"] if o["code"] == 0)
        print(f"  KQo button first in at {stack} bb: fold {fold:.0%}, raise {p_raise(res):.0%}")
        assert fold < 0.05, (stack, res["options"])
        res = engine().decide(request(6, stack, "3 0 0 4 0 0 5 0 0", BTN, "Kd Qc", mode="search", ms=500))
        fold = sum(o["prob"] for o in res["options"] if o["code"] == 0)
        assert fold < 0.10, ("search", stack, res["options"])


def call_prob(players, stack, actions, hero, hole, ms=600, extra="", board=""):
    res = engine().decide(request(players, stack, actions, hero, hole, board=board, mode="search",
                                  ms=ms, extra=extra))
    return p_call(res), res


def test_calls_off_a_stack_with_enough_equity():
    """Complaint: the old bots would not call off their stacks with enough
    equity. Each spot is a clear call against the range that moves all-in."""
    spots = [
        # heads-up, 10 bb: button (small blind) jams, big blind calls
        (2, 10, "0 2 1000", 1, ["Ah Jd", "Ks Qs", "Tc Th", "As 8s", "Kc Tc"]),
        # six-handed, 20 bb: folds to the button, button jams, small blind folds
        (6, 20, "3 0 0 4 0 0 5 0 0 0 2 2000 1 0 0", BB, ["Ac Qd", "Jh Js", "As Kc", "Tc Td"]),
        # six-handed, 15 bb: small blind jams into the big blind
        (6, 15, "3 0 0 4 0 0 5 0 0 0 0 0 1 2 1500", BB, ["Ad Td", "Kd Qh", "9s 9h", "Ah 7c"]),
    ]
    for players, stack, acts, hero, hands in spots:
        for hole in hands:
            p, res = call_prob(players, stack, acts, hero, hole)
            print(f"  {players} players {stack} bb, {hole} facing all-in: call {p:.0%} "
                  f"(equity {res['equity']:.0%})")
            assert p >= 0.8, (players, stack, hole, res["options"])


def test_calls_a_river_shove_with_a_strong_hand():
    # Heads-up 20 bb, single-raised pot; villain shoves the river.
    acts = "0 2 250 1 1 0 1 1 0 0 1 0 1 1 0 0 1 0 1 1 0 0 2 1750"
    for hole in ["Js 9s", "Qs Qd"]:   # nut straight, top set
        p, res = call_prob(2, 20, acts, 1, hole, board="Qh 8d 3c 2s Td")
        print(f"  river shove, {hole} on Qh 8d 3c 2s Td: call {p:.0%} (equity {res['equity']:.0%})")
        assert p >= 0.9, (hole, res["options"])


FLOPS = ["Ks 7d 2c", "Qh Jh 4s", "9c 8c 5d", "As Td 6h", "7s 7c 3d", "Jd 6c 3h", "Kh Qd Td", "5s 4s 2h"]


def donk_rate(players, stack, preflop, hero, mode="blueprint", ms=300, sample=None):
    """Share of the caller's range that bets first into the preflop raiser."""
    rng = random.Random(1)
    tot = w = 0.0
    for flop in FLOPS:
        dead = set(cards(flop))
        reps = [(h, n) for h, n in class_reps() if not dead & set(h)]
        if sample:
            reps = rng.sample(reps, sample)
        for hole, n in reps:
            res = engine().decide(request(players, stack, preflop, hero, hole, board=flop, mode=mode, ms=ms))
            tot += n * p_raise(res)
            w += n
    return tot / w


def test_caller_rarely_donk_bets_into_the_raiser():
    """Complaint: the old bots led out of position far too often and almost
    never checked. Solvers check most of the big blind's range to the
    preflop raiser in single-raised pots."""
    spots = [(2, 20, "0 2 250 1 1 0", 1, "heads-up, button raised, big blind called"),
             (6, 20, "3 0 0 4 0 0 5 2 250 0 0 0 1 0 0 2 1 0", BB, "six-handed, cutoff raised, big blind called")]
    for players, stack, pre, hero, what in spots:
        r = donk_rate(players, stack, pre, hero)
        rs = donk_rate(players, stack, pre, hero, mode="search", sample=12)
        print(f"  {what}: big blind leads {r:.0%} of hands (blueprint), {rs:.0%} (search, sampled)")
        assert r <= 0.35 and rs <= 0.35, (what, r, rs)


def maniac_model(pos: int, n: int) -> str:
    """Model lines for a player who moved all-in every time it was first in."""
    return f"model {pos} 0 0 0 {n}\nmodel {pos} 2 0 0 {n}\n"


def test_exploits_a_maniac_by_calling_wider():
    """After watching a player jam every hand, the big blind calls wider."""
    acts = "3 0 0 4 0 0 5 0 0 0 0 0 1 2 2000"      # small blind jams 20 bb
    hands = ["Kd 8c", "Qh 9c", "Jd Td", "As 4c", "7s 7h", "Kh 5h", "Tc 8d"]
    base = model = 0.0
    for hole in hands:
        p0, _ = call_prob(6, 20, acts, BB, hole, ms=400)
        p1, _ = call_prob(6, 20, acts, BB, hole, ms=400, extra=maniac_model(SB, 80))
        base += p0 / len(hands)
        model += p1 / len(hands)
    print(f"  big blind call rate vs small-blind jam: {base:.0%} unknown player, {model:.0%} vs maniac")
    assert model >= base + 0.15, (base, model)


def test_restricted_nash_response_limits_exploitation():
    """A tiny sample (2 hands) moves the bot much less than a large sample."""
    acts = "3 0 0 4 0 0 5 0 0 0 0 0 1 2 2000"
    hands = ["Kd 8c", "Qh 9c", "As 4c", "Kh 5h"]
    few = many = 0.0
    for hole in hands:
        few += call_prob(6, 20, acts, BB, hole, ms=400, extra=maniac_model(SB, 2))[0] / len(hands)
        many += call_prob(6, 20, acts, BB, hole, ms=400, extra=maniac_model(SB, 200))[0] / len(hands)
    print(f"  call rate after 2 observed jams {few:.0%}, after 200: {many:.0%}")
    assert many > few + 0.1


def test_no_deep_all_ins():
    """No 100 bb jams into small pots (the complaint about the trained six-handed bots)."""
    spots = [
        ("UTG first in", "", 3, "", ["As Ad", "Ks Kh", "7c 7d", "As Kd"]),
        ("big blind vs open and call", "3 2 250 4 0 0 5 0 0 0 0 0 1 1 250", 2, "", ["Qs Qh", "Ah 5h", "Kc Kd"]),
        ("flop, facing a bet three-way", "3 2 250 4 0 0 5 0 0 0 0 0 1 1 250 2 1 250 1 1 0 2 1 0 3 2 350", 1,
         "Ks 7d 2c", ["Kd Kh", "7s 7h", "Ac Kc", "2s 2h"]),
    ]
    for name, acts, hero, board, hands in spots:
        for hole in hands:
            res = engine().decide(request(6, 100, acts, hero, hole, board=board, mode="search", ms=300))
            jam = sum(o["prob"] for o in res["options"] if o["code"] == 15)
            print(f"  100 bb, {name}, {hole}: all-in {jam:.0%} (pot {res['pot'] / 100:.1f} bb)")
            assert jam == 0, (name, hole, res["options"])


def test_contexts_classify_actions():
    # UTG opens, HJ 3-bets, UTG folds (fold to 3-bet), everyone else folds.
    req = request(6, 20, "3 2 250 4 2 700 5 0 0 0 0 0 1 0 0 2 0 0 3 0 0", 0, "As Kd")
    obs = engine().contexts(req)
    assert obs[0] == [3, 0, 2]            # UTG, preflop unopened, raise
    assert obs[1] == [4, 2, 2]            # HJ, facing open, raise
    assert obs[-1] == [3, 3, 0]           # UTG, facing 3-bet, fold


def test_tournament_runs_and_conserves_chips():
    bots = mccfr_bots(CFG, k=6, settings=SearchSettings(mode="blueprint"), seed=3)
    t = Table(bots, starting_stack=400, seed=11)
    total = sum(t.stacks)
    for _ in range(60):
        if len(t.alive()) < 2:
            break
        st = t.play_hand()
        assert sum(t.stacks) == total, st.summary()
        assert sum(st.net) == 0


def test_training_stops_and_resumes_from_checkpoint():
    build(quiet=True)
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "hu.ckpt"
        common = [str(EXE), "train", "--config", str(CFG.path), "--abstraction", str(CFG.abstraction_dir),
                  "--players", "2", "--stack", "10", "--out", str(ck), "--threads", "2", "--memory", "1"]
        out1 = subprocess.run(common + ["--iterations", "20000"], capture_output=True, text=True)
        assert out1.returncode == 0, out1.stderr
        out2 = subprocess.run(common + ["--iterations", "50000", "--export", str(Path(d) / "hu.strat")],
                              capture_output=True, text=True)
        assert "Resuming" in out2.stdout and "iteration 20000" in out2.stdout, out2.stdout
        assert "Stopped at iteration 50000" in out2.stdout, out2.stdout
        assert (Path(d) / "hu.strat").exists()


def test_memory_estimate_scales_with_buckets():
    build(quiet=True)
    out = subprocess.run([str(EXE), "memory", "--config", str(CFG.path), "--players", "3", "--stacks", "20"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    row = [ln for ln in out.stdout.splitlines() if ln.startswith("3 ")][0].split()
    gib = [float(x) for x in row[3:3 + 2]]
    assert gib[0] >= gib[1], row               # coarser levels never need more memory


def test_search_offers_more_sizes_than_the_blueprint():
    menus = CFG.search_menus()
    if not menus.get("search_bet"):
        print("  (no search_bet menu in the config; skipped)")
        return
    acts = "3 2 250 4 0 0 5 0 0 0 0 0 1 0 0 2 1 250"   # UTG opens, BB calls, heads-up flop
    base = request(6, 40, acts, BB, "Ah Kh", board="Kd 7s 2c", mode="search", ms=100)
    extra = "".join(f"{k} {' '.join(map(str, v))}\n" for k, v in menus.items())
    bp = engine().decide(request(6, 40, acts, BB, "Ah Kh", board="Kd 7s 2c", mode="blueprint"))
    se = engine().decide(base + extra)
    nb = sum(1 for o in bp["options"] if o["code"] >= 2)
    ns = sum(1 for o in se["options"] if o["code"] >= 2)
    assert ns > nb, (bp["options"], se["options"])
    assert abs(sum(o["prob"] for o in se["options"]) - 1) < 1e-3


def test_exploitability_estimate_runs():
    build(quiet=True)
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "hu.ckpt"
        common = [str(EXE), "train", "--config", str(CFG.path), "--abstraction", str(CFG.abstraction_dir),
                  "--players", "2", "--stack", "10", "--out", str(ck), "--threads", "2", "--memory", "1"]
        assert subprocess.run(common + ["--iterations", "20000"], capture_output=True).returncode == 0
        out = subprocess.run([str(EXE), "lbr", "--in", str(ck), "--hands", "2000", "--abstraction",
                              str(CFG.abstraction_dir), "--threads", "2"], capture_output=True, text=True)
        assert out.returncode == 0 and "RESULT" in out.stdout, out.stdout + out.stderr


if __name__ == "__main__":
    pick = sys.argv[1] if len(sys.argv) > 1 else ""
    random.seed(0)
    print(f"Blueprints: {CFG.blueprint_dir}")
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and pick in name:
            print(name)
            try:
                fn()
            except AssertionError as e:
                failed += 1
                print(f"  FAILED: {e}")
    print("all tests passed" if not failed else f"{failed} test(s) failed")
    sys.exit(1 if failed else 0)
