"""Fixed-strategy bots used as training opponents and exploit tests."""
from __future__ import annotations

import numpy as np

from .agents import check_call
from .cards import rank_of
from .encoding import context_fields
from .state import Decision


def _fields(current):
    f = context_fields(current)
    bb = f["big_blind_chips"].item() * 1000
    return f, bb


def all_in_bot(history, current):
    """Goes all-in every hand it plays (never folds)."""
    f, bb = _fields(current)
    if f["can_raise"].item():
        return Decision.raise_to(int(round(f["max_raise_to_bb"].item() * bb)))
    return Decision.call()


def _hole(current):
    cards = np.nonzero(context_fields(current)["hole_cards"][:52].numpy())[0]
    return [int(c) for c in cards]


def premium_only_bot(history, current):
    """Folds everything except AA, KK, QQ and AK; plays those aggressively."""
    f, bb = _fields(current)
    a, b = _hole(current)
    ra, rb = sorted((rank_of(a), rank_of(b)), reverse=True)
    premium = (ra == rb and ra >= 10) or (ra == 12 and rb == 11)
    to_call = f["to_call_bb"].item()
    if not premium:
        return Decision.check() if to_call == 0 else Decision.fold()
    if f["can_raise"].item():
        pot = f["pot_bb"].item()
        target = max(f["min_raise_to_bb"].item(), to_call + pot) + \
            f["committed_street_bb"][0].item()
        return Decision.raise_to(int(round(target * bb)))
    return Decision.call()


def make_random_aggro(seed=None):
    rng = np.random.default_rng(seed)

    def random_aggro(history, current):
        f, bb = _fields(current)
        r = rng.random()
        if r < 0.25 and f["to_call_bb"].item() > 0:
            return Decision.fold()
        if r < 0.6 and f["can_raise"].item():
            pot = f["pot_bb"].item()
            return Decision.raise_to(int(round(
                (f["min_raise_to_bb"].item() + pot * rng.random()) * bb)))
        return Decision.call()
    return random_aggro


class RiverJammer:
    """Plays like the teacher preflop, check-calls the flop and turn, then
    moves all-in on the river. With bluff=True it jams both its strongest
    hands and its busted hands (a polarized bluffer); with bluff=False it
    jams only strong hands and otherwise checks and folds to big bets."""

    def __init__(self, bluff: bool, seed=None):
        from .neural import Teacher
        self.teacher = Teacher(seed)
        self.bluff = bluff

    def __call__(self, history, current):
        from .features import A_CALL, A_FOLD, action_to_decision, features, legal_mask
        t = self.teacher
        t.model.update(history)
        f, sit = features(current.numpy(), t.model, 1.0)
        if sit.street == 0:
            a = t.choose(f, sit)
            if not legal_mask(sit)[a]:
                a = A_CALL
            return action_to_decision(a, sit)
        raw = f[2]                       # equity vs a random hand
        if sit.street < 3:
            if sit.to_call == 0 or sit.to_call <= sit.pot or raw > 0.6:
                return action_to_decision(A_CALL, sit)
            return action_to_decision(A_FOLD, sit)
        jam = raw >= 0.8 or (self.bluff and raw < 0.4)
        if jam and sit.can_raise:
            return Decision.raise_to(int(round(sit.max_to * sit.bb_chips)))
        if sit.to_call == 0:
            return Decision.check()
        return Decision.call() if raw > 0.65 else Decision.fold()


SPARRING = {
    "all_in": lambda seed=None: all_in_bot,
    "premium_only": lambda seed=None: premium_only_bot,
    "check_call": lambda seed=None: check_call,
    "random_aggro": make_random_aggro,
    "river_bluffer": lambda seed=None: RiverJammer(True, seed),
    "value_jammer": lambda seed=None: RiverJammer(False, seed),
}


# Test-only bots (not used as training opponents) --------------------------
def make_min_raiser(freq=0.7, seed=None):
    """Plays like the teacher, but when facing a bet or raise it min-raises
    `freq` of the time."""
    from .neural import Teacher
    teacher = Teacher(seed)
    rng = np.random.default_rng(seed)

    def min_raiser(history, current):
        f, bb = _fields(current)
        if f["to_call_bb"].item() > 0 and f["can_raise"].item() and rng.random() < freq:
            return Decision.raise_to(int(round(f["min_raise_to_bb"].item() * bb)))
        return teacher(history, current)
    return min_raiser


def make_jam_fold(width, seed=None):
    """Moves all-in with the strongest `width` fraction of starting hands and
    folds everything else (checks when checking is free)."""
    from .handstrength import percentile

    def jam_fold(history, current):
        f, bb = _fields(current)
        a, b = _hole(current)
        if percentile(a, b) <= width:
            if f["can_raise"].item():
                return Decision.raise_to(int(round(f["max_raise_to_bb"].item() * bb)))
            return Decision.call()
        return Decision.check() if f["to_call_bb"].item() == 0 else Decision.fold()
    return jam_fold


TEST_BOTS = {
    "min_raiser": lambda seed=None: make_min_raiser(0.7, seed),
    "jam_fold_wide": lambda seed=None: make_jam_fold(0.50, seed),
    "jam_fold_narrow": lambda seed=None: make_jam_fold(0.12, seed),
}
ALL_FIXED_BOTS = {**SPARRING, **TEST_BOTS}
