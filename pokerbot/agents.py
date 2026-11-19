"""Agents: the bot shell and the human CLI player."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import torch

from .cards import cards_str
from .encoding import context_fields
from .state import Decision, HandState, LegalActions

DecisionLike = Union[Decision, str, tuple]
DecisionFn = Callable[[torch.Tensor, torch.Tensor], DecisionLike]


@dataclass
class Observation:
    seat: int
    history: torch.Tensor   # (N, HAND_DIM): past hands from this seat's view
    current: torch.Tensor   # (HAND_DIM,): the hand in progress, masked
    legal: LegalActions
    state: HandState        # full state; only the human agent may read it


class Agent:
    name: str = "agent"
    is_human: bool = False

    def act(self, obs: Observation) -> Decision:
        raise NotImplementedError

    def hand_finished(self, state: HandState, seat: int) -> None:
        """Called after every hand. Default: do nothing."""

    def show_or_muck(self, state: HandState, seat: int) -> bool:
        """Asked at showdown when this player cannot win.
        Return True to show, False to muck. Default: muck."""
        return False


def to_decision(out: DecisionLike) -> Decision:
    if isinstance(out, Decision):
        return out
    if isinstance(out, str):
        return Decision(out.lower())
    kind, amount = out
    return Decision(str(kind).lower(), int(amount))


def sanitize(d: Decision, legal: LegalActions) -> Decision:
    """Map any decision onto a legal one.

    Check when facing a bet becomes fold; call with nothing to call becomes
    check; an illegal raise becomes call; raise amounts are clamped to
    [min_raise_to, max_raise_to].
    """
    kind = d.kind
    if kind == "fold":
        return Decision.check() if legal.to_call == 0 else d
    if kind == "check":
        return d if legal.to_call == 0 else Decision.fold()
    if kind == "call":
        return Decision.check() if legal.to_call == 0 else d
    if kind in ("raise", "bet"):
        if not legal.can_raise:
            return Decision.check() if legal.to_call == 0 else Decision.call()
        amt = min(max(d.amount, legal.min_raise_to), legal.max_raise_to)
        return Decision.raise_to(amt)
    raise ValueError(f"unknown decision kind {d.kind!r}")


class Bot(Agent):
    """Shell around a decision function f(history, current) -> decision.

    The function sees only tensors (see encoding.py), so it cannot peek at
    hidden cards. It may return a Decision, a string ("fold", "check",
    "call"), or a tuple ("raise", raise_to_amount_in_chips). Illegal outputs
    are mapped to legal ones by `sanitize`.

    history_window limits how many past hands are passed in (None = all).
    """

    def __init__(self, name: str, decide: DecisionFn,
                 history_window: int | None = None):
        self.name = name
        self.decide = decide
        self.history_window = history_window

    def act(self, obs: Observation) -> Decision:
        with torch.no_grad():
            out = self.decide(obs.history, obs.current)
        return sanitize(to_decision(out), obs.legal)


# Decision functions ----------------------------------------------------
def check_call(history: torch.Tensor, current: torch.Tensor) -> Decision:
    """Never folds, never raises."""
    to_call = context_fields(current)["to_call_bb"].item()
    return Decision.call() if to_call > 0 else Decision.check()


# Human player ----------------------------------------------------------
class HumanCLI(Agent):
    """Text interface. Commands: f, k (check), c (call), r <amount>, a."""

    is_human = True

    def __init__(self, name: str = "You", auto_muck: bool = False):
        self.name = name
        self.auto_muck = auto_muck

    def show_or_muck(self, state: HandState, seat: int) -> bool:
        if self.auto_muck:
            return False
        try:
            raw = input("Your hand can't win. Show or muck? (s/m) ").strip().lower()
        except EOFError:
            return False
        return raw in ("s", "show")

    def act(self, obs: Observation) -> Decision:
        s, seat, legal = obs.state, obs.seat, obs.legal
        print()
        print(f"Board: {cards_str(s.board)}   Pot: {s.pot}")
        print(_seat_table(s, seat))
        print(f"Your cards: {cards_str(s.hole_cards[seat])}")
        print(f"Options: {legal.describe()}")
        while True:
            try:
                raw = input("> ").strip().lower()
            except EOFError:
                raise SystemExit(0)
            d = _parse(raw, legal)
            if d is not None:
                return d
            print("Enter f (fold), k (check), c (call), r <amount> "
                  "(bet/raise to), a (all-in), or q (quit).")

    def hand_finished(self, state: HandState, seat: int) -> None:
        print()
        print("-" * 60)
        print(state.summary())
        for p in range(state.num_seats):
            if state.shown[p]:
                print(f"  {state.names[p]} shows {cards_str(state.hole_cards[p])}")
        for p in range(state.num_seats):
            if state.net[p]:
                sign = "+" if state.net[p] > 0 else ""
                print(f"  {state.names[p]}: {sign}{state.net[p]}")
        print("-" * 60)


def _parse(raw: str, legal: LegalActions) -> Decision | None:
    if not raw:
        return None
    if raw in ("q", "quit", "exit"):
        raise SystemExit(0)
    cmd, *rest = raw.split()
    if cmd in ("f", "fold"):
        return Decision.fold()
    if cmd in ("k", "check", "x"):
        return Decision.check() if legal.to_call == 0 else None
    if cmd in ("c", "call"):
        return Decision.call() if legal.to_call > 0 else Decision.check()
    if cmd in ("a", "allin", "all-in", "shove"):
        if legal.can_raise:
            return Decision.raise_to(legal.max_raise_to)
        return Decision.call() if legal.to_call else Decision.check()
    if cmd in ("r", "b", "raise", "bet") and rest and rest[0].isdigit():
        if not legal.can_raise:
            print("Raising is not allowed here.")
            return None
        amt = int(rest[0])
        if amt > legal.max_raise_to:
            print(f"You can raise to at most {legal.max_raise_to}.")
            return None
        if amt < legal.min_raise_to and amt != legal.max_raise_to:
            print(f"Minimum raise is to {legal.min_raise_to}.")
            return None
        return Decision.raise_to(amt)
    return None


def _seat_table(s: HandState, me: int) -> str:
    rows = []
    for p in range(s.num_seats):
        if not s.dealt_in[p]:
            continue
        tags = []
        if p == s.button:
            tags.append("BTN")
        if p == s.sb_seat:
            tags.append("SB")
        if p == s.bb_seat:
            tags.append("BB")
        status = "folded" if s.folded[p] else ("all-in" if s.stacks[p] == 0 else "")
        mark = "*" if p == me else " "
        rows.append(f" {mark}{s.names[p]:<8} stack {s.stacks[p]:>6}  "
                    f"bet {s.committed_street[p]:>5}  {' '.join(tags):<7} {status}")
    return "\n".join(rows)
