"""Tensor encoding of hands.

Every hand, current or past, encodes to one 1-D float tensor of length
HAND_DIM, always from one observer's point of view:

    [ context block (CONTEXT_DIM) | action block (MAX_ACTIONS * ACTION_DIM) ]

Seats are relative to the observer: relative seat 0 is the observer, 1 is the
next seat clockwise, and so on. Seats are fixed for the whole tournament, so a
relative seat always refers to the same opponent, which lets a model track
each opponent across hands. Chip amounts are in big blinds.

Hidden information is masked: the observer sees its own hole cards, and sees
opponents' cards only if they were shown at showdown (past hands only).

The history tensor is a stack of past hands, shape (N, HAND_DIM), oldest first.
Use `split` / `context_fields` to recover named pieces.
"""
from __future__ import annotations

import numpy as np
import torch

from .state import ActionType, HandState, LegalActions

NUM_SEATS = 6
MAX_ACTIONS = 96
NUM_ACTION_TYPES = len(ActionType)


def _build_layout(spec):
    layout, offset = {}, 0
    for name, size in spec:
        layout[name] = slice(offset, offset + size)
        offset += size
    return layout, offset


CONTEXT_SPEC = [
    ("is_current", 1),            # 1 for the hand being played, 0 for history
    ("big_blind_chips", 1),       # raw chips / 1000, for blind-level awareness
    ("ante_bb", 1),
    ("level", 1),
    ("button", NUM_SEATS),        # one-hot relative seat
    ("sb_seat", NUM_SEATS),
    ("bb_seat", NUM_SEATS),
    ("street", 4),                # current street (or last reached, for history)
    ("board", 52),                # multi-hot
    ("hole_cards", NUM_SEATS * 52),  # per relative seat, multi-hot, masked
    ("starting_stacks_bb", NUM_SEATS),
    ("stacks_bb", NUM_SEATS),     # chips behind now (end of hand for history)
    ("committed_street_bb", NUM_SEATS),
    ("committed_total_bb", NUM_SEATS),
    ("dealt_in", NUM_SEATS),
    ("in_hand", NUM_SEATS),       # dealt in and not folded
    ("all_in", NUM_SEATS),
    ("pot_at_street_bb", 4),      # pot at the start of each street
    ("street_reached", 4),
    ("pot_bb", 1),
    ("to_call_bb", 1),            # legal-action info for the observer
    ("can_raise", 1),
    ("min_raise_to_bb", 1),
    ("max_raise_to_bb", 1),
    ("net_bb", NUM_SEATS),        # result of the hand (history only)
    ("num_actions", 1),
    ("board_order", 5),           # board cards in deal order, as card + 1 (0 = none)
    ("mucked", NUM_SEATS),        # mucked at showdown (history only)
]

ACTION_SPEC = [
    ("valid", 1),
    ("street", 4),
    ("actor", NUM_SEATS),         # relative seat one-hot
    ("kind", NUM_ACTION_TYPES),
    ("all_in", 1),
    ("amount_bb", 1),             # chips added
    ("to_bb", 1),                 # street total after the action
    ("pot_before_bb", 1),
    ("to_call_before_bb", 1),
    ("pot_fraction", 1),          # amount added beyond the call / (pot + call)
]

CONTEXT_LAYOUT, CONTEXT_DIM = _build_layout(CONTEXT_SPEC)
ACTION_LAYOUT, ACTION_DIM = _build_layout(ACTION_SPEC)
HAND_DIM = CONTEXT_DIM + MAX_ACTIONS * ACTION_DIM


def _rel(seat: int, observer: int, n: int) -> int:
    return (seat - observer) % n


_C = {k: v.start for k, v in CONTEXT_LAYOUT.items()}
_A = {k: v.start for k, v in ACTION_LAYOUT.items()}


def encode_hand_np(state: HandState, observer: int, *, current: bool,
                   legal: LegalActions | None = None) -> np.ndarray:
    """NumPy version of encode_hand (same layout)."""
    n = state.num_seats
    assert n <= NUM_SEATS
    bb = float(state.big_blind)
    out = np.zeros(HAND_DIM, dtype=np.float32)
    C = _C
    rel = [(s - observer) % n for s in range(n)]

    out[C["is_current"]] = 1.0 if current else 0.0
    out[C["big_blind_chips"]] = state.big_blind / 1000.0
    out[C["ante_bb"]] = state.ante / bb
    out[C["level"]] = float(state.level)
    out[C["button"] + rel[state.button]] = 1.0
    if state.sb_seat >= 0:
        out[C["sb_seat"] + rel[state.sb_seat]] = 1.0
    if state.bb_seat >= 0:
        out[C["bb_seat"] + rel[state.bb_seat]] = 1.0
    out[C["street"] + state.street] = 1.0
    for i, c in enumerate(state.board):
        out[C["board"] + c] = 1.0
        out[C["board_order"] + i] = float(c + 1)
    for seat in range(n):
        r = rel[seat]
        if seat == observer or (not current and state.shown[seat]):
            for c in state.hole_cards[seat]:
                out[C["hole_cards"] + r * 52 + c] = 1.0
        out[C["starting_stacks_bb"] + r] = state.starting_stacks[seat] / bb
        out[C["stacks_bb"] + r] = state.stacks[seat] / bb
        out[C["committed_street_bb"] + r] = state.committed_street[seat] / bb
        out[C["committed_total_bb"] + r] = state.committed_total[seat] / bb
        out[C["dealt_in"] + r] = float(state.dealt_in[seat])
        live = state.dealt_in[seat] and not state.folded[seat]
        out[C["in_hand"] + r] = float(live)
        out[C["all_in"] + r] = float(live and state.stacks[seat] == 0)
        if not current:
            out[C["net_bb"] + r] = state.net[seat] / bb
            out[C["mucked"] + r] = float(state.mucked[seat])
    for st in range(4):
        out[C["pot_at_street_bb"] + st] = state.pot_at_street_start[st] / bb
        out[C["street_reached"] + st] = float(state.street_reached[st])
    out[C["pot_bb"]] = state.pot / bb
    if legal is not None:
        out[C["to_call_bb"]] = legal.to_call / bb
        out[C["can_raise"]] = float(legal.can_raise)
        out[C["min_raise_to_bb"]] = legal.min_raise_to / bb
        out[C["max_raise_to_bb"]] = legal.max_raise_to / bb
    out[C["num_actions"]] = float(min(len(state.actions), MAX_ACTIONS))

    A = _A
    for i, a in enumerate(state.actions[:MAX_ACTIONS]):
        b = CONTEXT_DIM + i * ACTION_DIM
        out[b + A["valid"]] = 1.0
        out[b + A["street"] + a.street] = 1.0
        out[b + A["actor"] + rel[a.seat]] = 1.0
        out[b + A["kind"] + int(a.kind)] = 1.0
        out[b + A["all_in"]] = float(a.all_in)
        out[b + A["amount_bb"]] = a.amount / bb
        out[b + A["to_bb"]] = a.to / bb
        out[b + A["pot_before_bb"]] = a.pot_before / bb
        out[b + A["to_call_before_bb"]] = a.to_call_before / bb
        extra = max(a.amount - a.to_call_before, 0)
        denom = a.pot_before + a.to_call_before
        out[b + A["pot_fraction"]] = extra / denom if denom else 0.0
    return out


def encode_hand(state: HandState, observer: int, *, current: bool,
                legal: LegalActions | None = None) -> torch.Tensor:
    """Encode `state` from `observer`'s seat into a (HAND_DIM,) tensor."""
    return torch.from_numpy(encode_hand_np(state, observer, current=current,
                                           legal=legal))


def split(hand_tensor: torch.Tensor):
    """Split (..., HAND_DIM) into context (..., CONTEXT_DIM) and actions
    (..., MAX_ACTIONS, ACTION_DIM)."""
    ctx = hand_tensor[..., :CONTEXT_DIM]
    acts = hand_tensor[..., CONTEXT_DIM:].reshape(
        *hand_tensor.shape[:-1], MAX_ACTIONS, ACTION_DIM)
    return ctx, acts


def context_fields(hand_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Named view of the context block, e.g. fields['to_call_bb']."""
    ctx, _ = split(hand_tensor)
    return {k: ctx[..., s] for k, s in CONTEXT_LAYOUT.items()}


def action_fields(hand_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Named view of the action block; each value has shape
    (..., MAX_ACTIONS, width)."""
    _, acts = split(hand_tensor)
    return {k: acts[..., s] for k, s in ACTION_LAYOUT.items()}


class HistoryBuffer:
    """Keeps every finished hand, encoded once per observer seat.

    Rows live in a preallocated tensor per observer, so `tensor()` returns a
    view of all past hands without copying.
    """

    def __init__(self, num_seats: int, capacity: int = 256):
        self.num_seats = num_seats
        self.records: list[HandState] = []
        self._buf = [torch.zeros(capacity, HAND_DIM) for _ in range(num_seats)]
        self.size = 0

    def add(self, state: HandState) -> None:
        self.records.append(state)
        if self.size == self._buf[0].shape[0]:
            self._buf = [torch.cat([b, torch.zeros_like(b)]) for b in self._buf]
        for obs in range(self.num_seats):
            self._buf[obs][self.size] = torch.from_numpy(
                encode_hand_np(state, obs, current=False))
        self.size += 1

    def tensor(self, observer: int, last: int | None = None) -> torch.Tensor:
        start = 0 if last is None else max(0, self.size - last)
        return self._buf[observer][start:self.size]
