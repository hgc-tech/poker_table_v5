"""Cards, deck, and a 7-card hand evaluator.

A card is an int in [0, 52): rank = card // 4 (0 = deuce ... 12 = ace),
suit = card % 4 (0 = clubs, 1 = diamonds, 2 = hearts, 3 = spades).
"""
from __future__ import annotations

import random
from collections import Counter
from itertools import combinations

RANKS = "23456789TJQKA"
SUITS = "cdhs"
SUIT_SYMBOLS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}

HAND_CLASSES = [
    "High card", "Pair", "Two pair", "Three of a kind", "Straight",
    "Flush", "Full house", "Four of a kind", "Straight flush",
]


def rank_of(card: int) -> int:
    return card // 4


def suit_of(card: int) -> int:
    return card % 4


def card_str(card: int, pretty: bool = True) -> str:
    r, s = RANKS[rank_of(card)], SUITS[suit_of(card)]
    return r + (SUIT_SYMBOLS[s] if pretty else s)


def cards_str(cards, pretty: bool = True) -> str:
    return " ".join(card_str(c, pretty) for c in cards) if cards else "--"


def parse_card(text: str) -> int:
    """Parse e.g. 'As' or 'Td' into a card int."""
    return RANKS.index(text[0].upper()) * 4 + SUITS.index(text[1].lower())


class Deck:
    def __init__(self, rng: random.Random):
        self.cards = list(range(52))
        rng.shuffle(self.cards)

    def deal(self, n: int = 1) -> list[int]:
        out, self.cards = self.cards[:n], self.cards[n:]
        return out


def _rank5(cards) -> tuple:
    """Score a 5-card hand; larger tuples are better hands."""
    ranks = sorted((rank_of(c) for c in cards), reverse=True)
    counts = Counter(ranks)
    # Sort ranks by (multiplicity, rank) so kickers are ordered correctly.
    grouped = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [cnt for _, cnt in grouped]
    ordered = [r for r, _ in grouped]
    flush = len({suit_of(c) for c in cards}) == 1
    unique = sorted(set(ranks), reverse=True)
    straight_high = None
    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            straight_high = unique[0]
        elif unique == [12, 3, 2, 1, 0]:  # wheel A-2-3-4-5
            straight_high = 3
    if straight_high is not None and flush:
        return (8, straight_high)
    if shape == [4, 1]:
        return (7, *ordered)
    if shape == [3, 2]:
        return (6, *ordered)
    if flush:
        return (5, *ranks)
    if straight_high is not None:
        return (4, straight_high)
    if shape == [3, 1, 1]:
        return (3, *ordered)
    if shape == [2, 2, 1]:
        return (2, *ordered)
    if shape == [2, 1, 1, 1]:
        return (1, *ordered)
    return (0, *ranks)


def best_hand(cards) -> tuple:
    """Best 5-card score from 5 to 7 cards."""
    return max(_rank5(combo) for combo in combinations(cards, 5))


def hand_name(score: tuple) -> str:
    return HAND_CLASSES[score[0]]
