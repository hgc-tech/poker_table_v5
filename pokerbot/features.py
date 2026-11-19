"""Turn (history, current) tensors into a compact decision feature vector.

The features combine hand strength measured against each opponent's modeled
range, pot odds, stack sizes, position, fold equity read from the opponent
model, and tournament standing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import handstrength as hs
from .encoding import CONTEXT_LAYOUT, NUM_SEATS
from .opponent import BET, CALL, FOLD, POSTS, RAISE, OpponentModel, parse_actions

_C = {k: v.start for k, v in CONTEXT_LAYOUT.items()}

FEATURE_NAMES = [
    "bias", "equity", "raw_strength", "pot_odds", "edge", "call_frac",
    "log_stack", "log_spr", "preflop", "flop", "turn", "river", "behind",
    "live_opps", "fold_equity", "raises_this_street", "facing_all_in",
    "players_left", "stack_vs_avg", "stack_rank", "opp_aggression",
    "initiative", "hand_pct", "log_m", "facing_bet", "unopened",
    "equity_vs_bettors", "aggressor_bluff_rate", "aggressor_strength",
    "callers_strength", "shown_evidence",
]
NUM_FEATURES = len(FEATURE_NAMES)

# Action indices produced by policies: fold, check/call, eight raise sizes
# (as a fraction of the pot after calling), and all-in.
RAISE_FRACS = [0.25, 1 / 3, 0.5, 2 / 3, 1.0, 2.0, 2.5, 3.0]
RAISE_NAMES = ["1/4 pot", "1/3 pot", "1/2 pot", "2/3 pot", "pot", "2x pot",
               "2.5x pot", "3x pot"]
A_FOLD, A_CALL = 0, 1
(A_QUARTER, A_THIRD, A_HALF, A_TWO_THIRDS, A_POT, A_2X, A_25X,
 A_3X) = range(2, 2 + len(RAISE_FRACS))
A_ALLIN = 2 + len(RAISE_FRACS)
NUM_ACTIONS = A_ALLIN + 1
RAISE_ACTIONS = list(range(2, 2 + len(RAISE_FRACS)))
AGGRESSIVE_ACTIONS = RAISE_ACTIONS + [A_ALLIN]
RAISE_POT_FRAC = dict(zip(RAISE_ACTIONS, RAISE_FRACS))
ACTION_NAMES = ["fold", "check/call"] + [f"raise {n}" for n in RAISE_NAMES] + ["all-in"]
A_SMALL, A_BIG = A_HALF, A_POT          # older names


@dataclass
class Situation:
    """Decoded current decision, in big blinds."""
    hole: list
    board: list
    street: int
    pot: float
    to_call: float
    can_raise: bool
    min_to: float
    max_to: float
    my_bet: float
    current_bet: float
    stack: float
    bb_chips: float
    widths: dict           # live opponent rel seat -> modeled range width


def _vec(row, name, n):
    s = _C[name]
    return row[s:s + n]


def decode(row: np.ndarray, model: OpponentModel, width_mult: float = 1.0):
    hole = list(np.nonzero(_vec(row, "hole_cards", 52))[0])
    board = list(np.nonzero(_vec(row, "board", 52))[0])
    street = int(np.argmax(_vec(row, "street", 4)))
    stacks = _vec(row, "stacks_bb", NUM_SEATS)
    committed = _vec(row, "committed_street_bb", NUM_SEATS)
    in_hand = _vec(row, "in_hand", NUM_SEATS) > 0.5
    all_in = _vec(row, "all_in", NUM_SEATS) > 0.5
    acts = parse_actions(row)

    widths = {}
    for s in range(1, NUM_SEATS):
        if not in_hand[s]:
            continue
        r = model.rates(s)
        w = 1.0
        for street_a, actor, kind, a_all_in, raises_before, *_ in acts:
            if actor != s or street_a != 0 or kind in POSTS:
                continue
            if kind in (BET, RAISE):
                # Shown preflop raises calibrate how wide the raising range
                # really is (twice the mean percentile of shown raising hands).
                quality = 2.0 * model.rate(s, "pf_raise_pct") / max(r["pfr"], 1e-3)
                cand = r["pfr"] if raises_before == 0 else r["reraise"]
                cand *= min(max(quality, 0.25), 4.0)
                if a_all_in:
                    cand = max(min(cand, r["shove"] * 1.5), r["shove"])
                w = min(w, cand)
            elif kind == CALL:
                cand = r["vpip"] if raises_before == 0 else min(
                    r["vpip"], 1.0 - r["fold_to_raise"])
                w = min(w, cand)
        widths[s] = float(min(max(w * width_mult, 0.01), 1.0))

    return Situation(
        hole=hole, board=board, street=street,
        pot=float(row[_C["pot_bb"]]), to_call=float(row[_C["to_call_bb"]]),
        can_raise=row[_C["can_raise"]] > 0.5,
        min_to=float(row[_C["min_raise_to_bb"]]),
        max_to=float(row[_C["max_raise_to_bb"]]),
        my_bet=float(committed[0]), current_bet=float(committed.max()),
        stack=float(stacks[0]), bb_chips=float(row[_C["big_blind_chips"]]) * 1000,
        widths=widths), acts, stacks, in_hand, all_in


def features(row: np.ndarray, model: OpponentModel, width_mult: float = 1.0):
    sit, acts, stacks, in_hand, all_in = decode(row, model, width_mult)
    f = np.zeros(NUM_FEATURES, dtype=np.float32)
    opps = list(sit.widths)
    hole, board = sit.hole, sit.board

    eqs = [hs.equity(hole, board, sit.widths[s]) for s in opps]
    raw = hs.equity(hole, board, 1.0)
    eq = float(np.prod(eqs)) if eqs else 1.0
    # Equity against only the opponents who have put money in voluntarily
    # (everyone still live after the flop).
    if sit.street == 0:
        contest = {a[1] for a in acts if a[0] == 0 and a[2] in (CALL, BET, RAISE)}
        ceqs = [e for s, e in zip(opps, eqs) if s in contest]
        eq_bet = float(np.prod(ceqs)) if ceqs else raw
    else:
        eq_bet = eq
    pot_odds = sit.to_call / (sit.pot + sit.to_call) if sit.to_call > 0 else 0.0

    # Players to act after me this street.
    dealt = _vec(row, "dealt_in", NUM_SEATS) > 0.5
    if sit.street == 0:
        last = int(np.argmax(_vec(row, "bb_seat", NUM_SEATS)))
    else:
        last = int(np.argmax(_vec(row, "button", NUM_SEATS)))
    behind = sum(1 for s in range(1, last + 1) if in_hand[s] and not all_in[s]) \
        if last != 0 else 0

    street_acts = [a for a in acts if a[0] == sit.street and a[2] not in POSTS]
    raises = sum(1 for a in street_acts if a[2] in (BET, RAISE))
    last_aggr = None
    for a in acts:
        if a[2] in (BET, RAISE):
            last_aggr = a[1]

    if sit.street == 0:
        fold_eq = float(np.prod([model.rate(s, "fold_to_raise") for s in opps])) \
            if opps else 1.0
    else:
        fold_eq = float(np.prod([model.rate(s, "fold_to_bet") for s in opps])) \
            if opps else 1.0
    aggr = max((model.rate(s, "bet") for s in opps), default=0.0)

    alive = [s for s in range(NUM_SEATS) if dealt[s] or stacks[s] > 0]
    tot = [stacks[s] + _vec(row, "committed_total_bb", NUM_SEATS)[s]
           for s in range(NUM_SEATS)]
    alive_tot = [tot[s] for s in alive] or [1.0]
    my_tot = tot[0]
    rank = sum(1 for v in alive_tot if v > my_tot) / max(len(alive_tot) - 1, 1)
    ante = float(row[_C["ante_bb"]])
    m = (sit.stack + sit.my_bet) / (1.5 + ante * len(alive_tot))
    eff = min(sit.stack, max((stacks[s] for s in opps), default=0.0))

    f[0] = 1.0
    f[1] = eq
    f[2] = raw
    f[3] = pot_odds
    f[4] = eq - pot_odds
    f[5] = min(sit.to_call / max(sit.stack, 1e-6), 1.0) if sit.to_call else 0.0
    f[6] = math.log1p(sit.stack) / 5.0
    f[7] = math.log1p(eff / max(sit.pot, 1e-6)) / 3.0
    f[8 + sit.street] = 1.0
    f[12] = behind / 5.0
    f[13] = len(opps) / 5.0
    f[14] = fold_eq
    f[15] = min(raises, 3) / 3.0
    f[16] = float(any(all_in[s] for s in opps))
    f[17] = len(alive_tot) / 6.0
    f[18] = math.log(max(my_tot, 1e-3) / (sum(alive_tot) / len(alive_tot))) / 2.0
    f[19] = rank
    f[20] = aggr
    f[21] = float(last_aggr == 0)
    f[22] = 1.0 - hs.percentile(hole[0], hole[1])
    f[23] = math.log1p(m) / 4.0
    f[24] = float(sit.to_call > 0)
    f[25] = float(raises == 0)
    f[26] = eq_bet
    # Learned from shown cards: how often the player betting into us bluffs,
    # and how strong opponents are when they call.
    aggressor = None
    for a in acts:
        if a[0] == sit.street and a[2] in (BET, RAISE) and a[1] != 0 and a[1] in opps:
            aggressor = a
    if aggressor is not None and sit.street > 0:
        s_ag = aggressor[1]
        big = aggressor[3] or aggressor[6] >= 0.75
        f[27] = model.rate(s_ag, "big_bluff" if big else "bluff")
        f[28] = model.rate(s_ag, "agg_strength")
        f[30] = math.log1p(model.shown_count(s_ag, "big_shown" if big else "agg_shown")) / 3.0
    elif opps:
        f[27] = float(np.mean([model.rate(s, "bluff") for s in opps]))
        f[28] = float(np.mean([model.rate(s, "agg_strength") for s in opps]))
        f[30] = math.log1p(sum(model.shown_count(s) for s in opps)) / 3.0
    f[29] = float(np.mean([model.rate(s, "call_strength") for s in opps])) if opps else 0.62
    return f, sit


def action_to_decision(action: int, sit: Situation):
    """Map a policy action index onto a (kind, chips) decision."""
    from .state import Decision
    if action == A_FOLD:
        return Decision.fold() if sit.to_call > 0 else Decision.check()
    if action == A_CALL or not sit.can_raise:
        return Decision.call() if sit.to_call > 0 else Decision.check()
    if action == A_ALLIN:
        to_bb = sit.max_to
    else:
        frac = RAISE_POT_FRAC[action]
        to_bb = sit.current_bet + frac * (sit.pot + sit.to_call)
        to_bb = max(to_bb, sit.min_to)
        # Commit fully if the raise uses most of the stack.
        if to_bb >= 0.6 * sit.max_to:
            to_bb = sit.max_to
    return Decision.raise_to(int(round(min(to_bb, sit.max_to) * sit.bb_chips)))


def legal_mask(sit: Situation) -> np.ndarray:
    m = np.ones(NUM_ACTIONS, dtype=bool)
    if sit.to_call <= 0:
        m[A_FOLD] = False
    if not sit.can_raise:
        m[AGGRESSIVE_ACTIONS] = False
    return m
