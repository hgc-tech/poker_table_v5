"""Fast hand strength: eval7-backed scoring, preflop ranking, range equity.

A "range of width w" is the strongest w fraction of all 1326 starting hands,
ordered by preflop equity against a random hand. Opponent models describe each
opponent's likely holding as such a width.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import eval7

from .cards import RANKS, SUITS, rank_of, suit_of

E7 = [eval7.Card(RANKS[c // 4] + SUITS[c % 4]) for c in range(52)]
DATA = Path(__file__).resolve().parent / "data"
WIDTHS = np.geomspace(0.01, 1.0, 40)       # width buckets for preflop tables


def score(cards) -> int:
    """Higher is better; compares 5-7 card hands."""
    return eval7.evaluate([E7[c] for c in cards])


def hand_class(c1: int, c2: int) -> int:
    """Index 0..168 of the starting-hand class (pairs, suited, offsuit)."""
    r1, r2 = rank_of(c1), rank_of(c2)
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo:
        return hi                          # 0..12 pairs
    suited = suit_of(c1) == suit_of(c2)
    idx = hi * (hi - 1) // 2 + lo          # 0..77 for hi > lo
    return 13 + idx * 2 + (0 if suited else 1)


ALL_COMBOS = [(a, b) for a in range(52) for b in range(a + 1, 52)]


def _compute_class_equity(iters: int = 20000) -> list[float]:
    rand = [((E7[a], E7[b]), 1.0) for a, b in ALL_COMBOS]
    eq = [0.0] * 169
    done = set()
    for a, b in ALL_COMBOS:
        k = hand_class(a, b)
        if k in done:
            continue
        done.add(k)
        eq[k] = eval7.py_hand_vs_range_monte_carlo([E7[a], E7[b]], rand, [], iters)
    return eq


def _load_or_build():
    DATA.mkdir(exist_ok=True)
    path = DATA / "preflop_tables.json"
    if path.exists():
        d = json.loads(path.read_text())
        return d["class_equity"], np.array(d["eq_vs_width"])
    class_eq = _compute_class_equity()
    order = sorted(ALL_COMBOS, key=lambda c: -class_eq[hand_class(*c)])
    table = np.zeros((169, len(WIDTHS)))
    reps = {}
    for a, b in ALL_COMBOS:
        reps.setdefault(hand_class(a, b), (a, b))
    for j, w in enumerate(WIDTHS):
        k = max(1, int(round(w * len(order))))
        rng = [((E7[a], E7[b]), 1.0) for a, b in order[:k]]
        for cls, (a, b) in reps.items():
            table[cls, j] = eval7.py_hand_vs_range_monte_carlo(
                [E7[a], E7[b]], rng, [], 3000)
    path.write_text(json.dumps({"class_equity": class_eq,
                                "eq_vs_width": table.tolist()}))
    return class_eq, table


CLASS_EQUITY, EQ_VS_WIDTH = _load_or_build()
ORDER = sorted(ALL_COMBOS, key=lambda c: -CLASS_EQUITY[hand_class(*c)])
_ORDER_E7 = [((E7[a], E7[b]), 1.0) for a, b in ORDER]
# Percentile of each class: fraction of combos at least as strong.
_CLASS_PCT = {}
for i, (a, b) in enumerate(ORDER):
    _CLASS_PCT.setdefault(hand_class(a, b), 0.0)
    _CLASS_PCT[hand_class(a, b)] = (i + 1) / len(ORDER)


def percentile(c1: int, c2: int) -> float:
    """0 (best) .. 1 (worst): where this starting hand ranks among all."""
    return _CLASS_PCT[hand_class(c1, c2)]


@lru_cache(maxsize=2048)
def _range(k: int):
    return _ORDER_E7[:k]


def range_of_width(w: float):
    k = int(round(min(max(w, 0.005), 1.0) * len(ORDER)))
    # Quantize to keep the cache small.
    k = max(6, min(len(ORDER), (k // 6) * 6 or 6))
    return _range(k)


def preflop_equity(c1: int, c2: int, width: float) -> float:
    """Heads-up equity of the hand vs a range of the given width."""
    row = EQ_VS_WIDTH[hand_class(c1, c2)]
    x = np.log(min(max(width, WIDTHS[0]), 1.0))
    return float(np.interp(x, np.log(WIDTHS), row))


def equity(hole, board, width: float, iters: int = 250) -> float:
    """Heads-up equity vs a range of the given width (Monte Carlo)."""
    if not board:
        return preflop_equity(hole[0], hole[1], width)
    return eval7.py_hand_vs_range_monte_carlo(
        [E7[c] for c in hole], range_of_width(width), [E7[c] for c in board],
        iters)
