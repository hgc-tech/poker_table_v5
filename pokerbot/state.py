"""Plain-Python representation of one poker hand.

HandState is the single source of truth for a hand. The table mutates it as
play proceeds; encoding.py turns it into tensors for bots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from .cards import cards_str

STREETS = ["preflop", "flop", "turn", "river"]


class ActionType(IntEnum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4
    POST_SB = 5
    POST_BB = 6


@dataclass(frozen=True)
class Decision:
    """What an agent wants to do.

    kind is one of "fold", "check", "call", "raise". For "raise", amount is the
    total the player wants to have committed on this street ("raise to"),
    which also covers an opening bet.
    """
    kind: str
    amount: int = 0

    @staticmethod
    def fold() -> "Decision":
        return Decision("fold")

    @staticmethod
    def check() -> "Decision":
        return Decision("check")

    @staticmethod
    def call() -> "Decision":
        return Decision("call")

    @staticmethod
    def raise_to(amount: int) -> "Decision":
        return Decision("raise", int(amount))


@dataclass(frozen=True)
class Action:
    seat: int
    street: int
    kind: ActionType
    amount: int          # chips added to the pot by this action
    to: int              # player's total committed on this street afterwards
    pot_before: int      # total pot before this action
    to_call_before: int  # chips the player needed to call before acting
    all_in: bool = False


@dataclass(frozen=True)
class LegalActions:
    """Options available to the player to act."""
    to_call: int        # chips needed to call (0 means check is legal)
    can_raise: bool
    min_raise_to: int   # smallest legal raise-to total (all-in may be smaller)
    max_raise_to: int   # all-in raise-to total

    def describe(self) -> str:
        opts = ["fold"]
        opts.append("check" if self.to_call == 0 else f"call {self.to_call}")
        if self.can_raise:
            word = "bet" if self.to_call == 0 else "raise to"
            if self.min_raise_to >= self.max_raise_to:
                opts.append(f"{word} {self.max_raise_to} (all-in)")
            else:
                opts.append(f"{word} {self.min_raise_to}-{self.max_raise_to}")
        return ", ".join(opts)


@dataclass
class HandState:
    hand_id: int
    num_seats: int
    names: list[str]
    button: int
    small_blind: int
    big_blind: int
    ante: int
    level: int
    starting_stacks: list[int]
    stacks: list[int] = field(default_factory=list)       # chips behind
    dealt_in: list[bool] = field(default_factory=list)
    hole_cards: list[list[int]] = field(default_factory=list)
    board: list[int] = field(default_factory=list)
    street: int = 0
    actions: list[Action] = field(default_factory=list)
    folded: list[bool] = field(default_factory=list)
    committed_street: list[int] = field(default_factory=list)
    committed_total: list[int] = field(default_factory=list)
    antes_paid: list[int] = field(default_factory=list)
    pot_at_street_start: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    street_reached: list[bool] = field(default_factory=lambda: [True, False, False, False])
    shown: list[bool] = field(default_factory=list)       # tabled at showdown
    mucked: list[bool] = field(default_factory=list)      # mucked at showdown
    net: list[int] = field(default_factory=list)          # filled at hand end
    finished: bool = False
    sb_seat: int = -1
    bb_seat: int = -1

    def __post_init__(self):
        n = self.num_seats
        if not self.stacks:
            self.stacks = list(self.starting_stacks)
        if not self.dealt_in:
            self.dealt_in = [s > 0 for s in self.starting_stacks]
        for name in ("folded", "shown", "mucked"):
            if not getattr(self, name):
                setattr(self, name, [False] * n)
        for name in ("committed_street", "committed_total", "antes_paid", "net"):
            if not getattr(self, name):
                setattr(self, name, [0] * n)
        if not self.hole_cards:
            self.hole_cards = [[] for _ in range(n)]

    # Derived quantities -------------------------------------------------
    @property
    def pot(self) -> int:
        return sum(self.committed_total)

    @property
    def current_bet(self) -> int:
        return max(self.committed_street)

    def in_hand(self, seat: int) -> bool:
        return self.dealt_in[seat] and not self.folded[seat]

    def is_all_in(self, seat: int) -> bool:
        return self.in_hand(seat) and self.stacks[seat] == 0

    def live_seats(self) -> list[int]:
        return [s for s in range(self.num_seats) if self.in_hand(s)]

    def actions_on(self, street: int) -> list[Action]:
        return [a for a in self.actions if a.street == street]

    # Display ------------------------------------------------------------
    def summary(self) -> str:
        lines = [f"Hand #{self.hand_id}  level {self.level}  blinds "
                 f"{self.small_blind}/{self.big_blind}"
                 + (f" ante {self.ante}" if self.ante else "")]
        lines.append(f"Board: {cards_str(self.board)}   Pot: {self.pot}")
        for st in range(4):
            acts = self.actions_on(st)
            if acts:
                txt = ", ".join(describe_action(self, a) for a in acts)
                lines.append(f"  {STREETS[st]}: {txt}")
        return "\n".join(lines)


def describe_action(state: HandState, a: Action) -> str:
    name = state.names[a.seat]
    k = a.kind
    if k == ActionType.FOLD:
        s = "folds"
    elif k == ActionType.CHECK:
        s = "checks"
    elif k == ActionType.CALL:
        s = f"calls {a.amount}"
    elif k == ActionType.BET:
        s = f"bets {a.to}"
    elif k == ActionType.RAISE:
        s = f"raises to {a.to}"
    elif k == ActionType.POST_SB:
        s = f"posts SB {a.amount}"
    else:
        s = f"posts BB {a.amount}"
    if a.all_in:
        s += " (all-in)"
    if name.lower() == "you":          # "You call", not "You calls"
        verb, _, rest = s.partition(" ")
        s = verb[:-1] + (" " + rest if rest else "")
    return f"{name} {s}"
