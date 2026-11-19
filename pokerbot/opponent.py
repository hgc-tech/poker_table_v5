"""Opponent modeling from the full hand history tensor.

Two kinds of evidence are used:

* Action frequencies from every hand (how often a player enters the pot,
  raises, shoves, folds to a raise, bets after the flop, folds to a bet).
* Shown cards. When a hand is tabled at showdown, every action that player
  took is paired with the strength of their hand at that moment (preflop
  percentile, or equity against a random hand on the board at that street).
  This teaches how often their bets are bluffs, how strong their big bets and
  calls are, and how wide their preflop raises really are. Mucked hands teach
  nothing about cards.

`OpponentModel.update(history)` reads every past hand in the history tensor
(only rows it has not seen yet, so cost per hand stays constant) and keeps
recency-weighted counts of each opponent's tendencies. Rates are posterior
means under a prior that describes a solid player, so with no data a bot
assumes solid opponents and moves toward the observed behavior as hands come
in. The prior strength and the decay are evolvable genes.
"""
from __future__ import annotations

import numpy as np

from .encoding import (ACTION_DIM, ACTION_LAYOUT, CONTEXT_DIM, CONTEXT_LAYOUT,
                       MAX_ACTIONS, NUM_SEATS)
from .state import ActionType

_A = {k: v.start for k, v in ACTION_LAYOUT.items()}
_C = {k: v.start for k, v in CONTEXT_LAYOUT.items()}

FOLD, CHECK, CALL, BET, RAISE = (int(ActionType.FOLD), int(ActionType.CHECK),
                                 int(ActionType.CALL), int(ActionType.BET),
                                 int(ActionType.RAISE))
POSTS = (int(ActionType.POST_SB), int(ActionType.POST_BB))

# name: (numerator count, denominator count, prior rate of a solid player)
STATS = {
    "vpip": ("vpip", "dealt", 0.22),
    "pfr": ("pfr", "dealt", 0.16),
    "shove": ("shove", "dealt", 0.03),
    "fold_to_raise": ("pf_fold", "pf_faced", 0.65),
    "reraise": ("pf_reraise", "pf_faced", 0.08),
    "fold_to_bet": ("post_fold", "post_faced", 0.45),
    "raise_vs_bet": ("post_raise", "post_faced", 0.10),
    "bet": ("post_bet", "post_open", 0.45),
}
# Learned from shown hands only: (numerator, denominator, prior).
SHOWN_STATS = {
    "bluff": ("agg_weak", "agg_shown", 0.30),        # postflop bets/raises with < 50% equity
    "big_bluff": ("big_weak", "big_shown", 0.30),    # same, for pot-size-or-larger bets and jams
    "agg_strength": ("agg_eq", "agg_shown", 0.68),   # average equity when betting/raising
    "call_strength": ("call_eq", "call_shown", 0.62),  # average equity when calling postflop
    "pf_raise_pct": ("pf_raise_pct", "pf_raise_shown", None),  # mean percentile of shown raises
}
SHOW_PRIOR_N = 4.0      # shown hands are worth this many "solid player" hands of prior
WEAK_EQUITY = 0.5
BIG_BET = 0.75          # pot fraction that counts as a big bet

COUNTS = sorted({c for v in list(STATS.values()) + list(SHOWN_STATS.values())
                 for c in v[:2]})
CI = {c: i for i, c in enumerate(COUNTS)}


def parse_actions(row: np.ndarray):
    """Return (street, actor_rel, kind, all_in, raises_before_on_street,
    to_call_before_bb, pot_fraction) for each action in an encoded hand row."""
    acts = row[CONTEXT_DIM:].reshape(MAX_ACTIONS, ACTION_DIM)
    n = int(row[_C["num_actions"]])
    raises = [0, 0, 0, 0]
    out = []
    for i in range(n):
        a = acts[i]
        street = int(np.argmax(a[_A["street"]:_A["street"] + 4]))
        actor = int(np.argmax(a[_A["actor"]:_A["actor"] + NUM_SEATS]))
        kind = int(np.argmax(a[_A["kind"]:_A["kind"] + len(ActionType)]))
        all_in = a[_A["all_in"]] > 0.5
        out.append((street, actor, kind, all_in, raises[street],
                    float(a[_A["to_call_before_bb"]]), float(a[_A["pot_fraction"]])))
        if kind in (BET, RAISE):
            raises[street] += 1
    return out


class OpponentModel:
    def __init__(self, prior_n: float = 8.0, decay: float = 0.99):
        self.prior_n = prior_n
        self.decay = decay
        self.reset()

    def reset(self):
        shown_keys = {c for v in SHOWN_STATS.values() for c in v[:2]}
        self._decay_vec = np.array([self.decay ** (0.2 if c in shown_keys else 1.0)
                                    for c in COUNTS])
        self.counts = np.zeros((NUM_SEATS, len(COUNTS)))
        self.seen = 0

    def update(self, history) -> None:
        """Process any hands in `history` (N, HAND_DIM) not seen yet."""
        n = history.shape[0]
        if n < self.seen:          # a new tournament started
            self.reset()
        if n == self.seen:
            return
        rows = history[self.seen:n]
        rows = rows.numpy() if hasattr(rows, "numpy") else rows
        for row in rows:
            self._add_hand(row)
        self.seen = n

    def _add_hand(self, row: np.ndarray) -> None:
        # Shown hands are rare, so their evidence fades five times slower.
        self.counts *= self._decay_vec
        c = self.counts
        dealt = row[_C["dealt_in"]:_C["dealt_in"] + NUM_SEATS] > 0.5
        hand = np.zeros((NUM_SEATS, len(COUNTS)))
        flags = [dict() for _ in range(NUM_SEATS)]
        acts = parse_actions(row)
        self._learn_from_shown(row, acts, hand)
        for street, actor, kind, all_in, raises_before, to_call, _ in acts:
            if kind in POSTS:
                continue
            f, h = flags[actor], hand[actor]
            if street == 0:
                if kind in (CALL, BET, RAISE):
                    f["vpip"] = 1
                if kind in (BET, RAISE):
                    f["pfr"] = 1
                    if all_in:
                        f["shove"] = 1
                if raises_before > 0:
                    h[CI["pf_faced"]] += 1
                    if kind == FOLD:
                        h[CI["pf_fold"]] += 1
                    elif kind in (BET, RAISE):
                        h[CI["pf_reraise"]] += 1
            else:
                if to_call > 0:
                    h[CI["post_faced"]] += 1
                    if kind == FOLD:
                        h[CI["post_fold"]] += 1
                    elif kind in (BET, RAISE):
                        h[CI["post_raise"]] += 1
                else:
                    h[CI["post_open"]] += 1
                    if kind in (BET, RAISE):
                        h[CI["post_bet"]] += 1
        for s in range(NUM_SEATS):
            if not dealt[s]:
                continue
            hand[s, CI["dealt"]] += 1
            for k in ("vpip", "pfr", "shove"):
                hand[s, CI[k]] += flags[s].get(k, 0)
        c += hand

    def _learn_from_shown(self, row, acts, hand) -> None:
        from . import handstrength as hs
        holes = row[_C["hole_cards"]:_C["hole_cards"] + NUM_SEATS * 52].reshape(NUM_SEATS, 52)
        order = [int(v) - 1 for v in row[_C["board_order"]:_C["board_order"] + 5] if v > 0]
        for s in range(1, NUM_SEATS):
            cards = np.nonzero(holes[s])[0]
            if len(cards) != 2:
                continue                       # not shown (or mucked)
            hole = [int(cards[0]), int(cards[1])]
            h = hand[s]
            eq_cache = {}
            for street, actor, kind, all_in, _, _, frac in acts:
                if actor != s or kind in POSTS or kind == FOLD:
                    continue
                if street == 0:
                    if kind in (BET, RAISE):
                        h[CI["pf_raise_shown"]] += 1
                        h[CI["pf_raise_pct"]] += hs.percentile(*hole)
                    continue
                n_board = street + 2
                if len(order) < n_board or kind == CHECK:
                    continue
                if street not in eq_cache:
                    eq_cache[street] = hs.equity(hole, order[:n_board], 1.0, iters=200)
                eq = eq_cache[street]
                if kind in (BET, RAISE):
                    h[CI["agg_shown"]] += 1
                    h[CI["agg_eq"]] += eq
                    h[CI["agg_weak"]] += eq < WEAK_EQUITY
                    if all_in or frac >= BIG_BET:
                        h[CI["big_shown"]] += 1
                        h[CI["big_weak"]] += eq < WEAK_EQUITY
                elif kind == CALL:
                    h[CI["call_shown"]] += 1
                    h[CI["call_eq"]] += eq

    def rate(self, seat: int, stat: str) -> float:
        c = self.counts[seat]
        if stat in SHOWN_STATS:
            num, den, prior = SHOWN_STATS[stat]
            if prior is None:      # pf_raise_pct: prior = half the raise frequency
                prior = self.rate(seat, "pfr") / 2
            return (c[CI[num]] + SHOW_PRIOR_N * prior) / (c[CI[den]] + SHOW_PRIOR_N)
        num, den, prior = STATS[stat]
        return (c[CI[num]] + self.prior_n * prior) / (c[CI[den]] + self.prior_n)

    def shown_count(self, seat: int, den: str = "agg_shown") -> float:
        return float(self.counts[seat, CI[den]])

    def rates(self, seat: int) -> dict[str, float]:
        return {k: self.rate(seat, k) for k in STATS}

    def hands_observed(self, seat: int) -> float:
        return float(self.counts[seat, CI["dealt"]])
