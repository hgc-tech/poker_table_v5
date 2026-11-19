"""MCCFR bots: blueprint plus depth-limited search, with opponent modeling.

A bot sees only public information (blinds, stacks, the board, every action)
and its own hole cards. It reads the live HandState for those fields and
nothing else. After each hand it updates its action counts for every other
player from the public action list.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from ..agents import Agent, Observation, sanitize
from ..state import ActionType, Decision, HandState
from .config import CfrConfig
from .native import Engine

NCTX = 16
CTX_NAMES = [
    "preflop unopened", "-", "preflop facing open", "preflop facing 3-bet+",
    "flop first to bet", "flop as preflop aggressor", "flop facing bet", "flop facing raise",
    "turn first to bet", "turn as flop aggressor", "turn facing bet", "turn facing raise",
    "river first to bet", "river as turn aggressor", "river facing bet", "river facing raise"]


@dataclass
class SearchSettings:
    mode: str = "search"          # "search" or "blueprint"
    search_ms: float = 800        # time budget per decision
    threads: int = 2
    depth: int = 6                # actions searched ahead (stops at the end of the street)
    # Preflop: 0 = always play the blueprint, 1 = always search, 2 = search only after a bet size
    # far from the blueprint's sizes (as Pluribus did). Full preflop search overrates jams.
    search_preflop: int = 2
    rnr_max: float = 0.75         # largest weight on the opponent model
    rnr_n0: float = 60            # observed decisions at which the weight reaches half of rnr_max
    prior: float = 20             # Dirichlet prior strength (pseudo-observations at blueprint rates)
    equity_samples: int = 2000
    # No all-in when the effective stack exceeds max_jam_spr times the pot (after calling), except
    # preflop at jam_free_bb big blinds or less. 0 turns the limit off.
    max_jam_spr: float = 4
    jam_free_bb: float = 30
    # Search size menus (pot fractions), e.g. {"search_bet": [0.33, 0.75]}; empty = the
    # blueprint's sizes. mccfr_bots() fills these from cfr_config.txt.
    menus: dict = field(default_factory=dict)


def seat_order(st: HandState) -> list[int]:
    """Seats of the players dealt in, from the button clockwise (position order)."""
    n = st.num_seats
    return [s for s in ((st.button + k) % n for k in range(n)) if st.dealt_in[s]]


_KIND = {ActionType.FOLD: 0, ActionType.CHECK: 1, ActionType.CALL: 1,
         ActionType.BET: 2, ActionType.RAISE: 2}


def build_request(st: HandState, hero_seat: int | None, n_actions: int | None = None,
                  board: list[int] | None = None, payouts=(0.65, 0.35),
                  counts: dict | None = None, settings: SearchSettings | None = None,
                  seed: int = 1) -> tuple[str, list[int]]:
    """Text request for the engine. Uses public information only, plus the
    hero's hole cards. Returns (request, seats in position order)."""
    order = seat_order(st)
    pos = {s: i for i, s in enumerate(order)}
    acts = st.actions if n_actions is None else st.actions[:n_actions]
    lines = [f"players {len(order)}",
             "stacks " + " ".join(str(st.starting_stacks[s]) for s in order),
             f"blinds {st.small_blind} {st.big_blind} {st.ante}",
             "payouts " + " ".join(str(p) for p in list(payouts)[:len(order)])]
    items = []
    for a in acts:
        k = _KIND.get(a.kind)
        if k is not None:
            items.append(f"{pos[a.seat]} {k} {a.to}")
    lines.append("actions " + " ".join(items))
    b = st.board if board is None else board
    lines.append("board " + " ".join(str(c) for c in b))
    if hero_seat is not None:
        lines.append(f"hero {pos[hero_seat]}")
        h = st.hole_cards[hero_seat]
        if h:
            lines.append(f"hole {h[0]} {h[1]}")
    if settings is not None:
        s = settings
        lines += [f"mode {s.mode}", f"search_ms {s.search_ms}", f"threads {s.threads}",
                  f"depth {s.depth}", f"search_preflop {int(s.search_preflop)}",
                  f"rnr_max {s.rnr_max}", f"rnr_n0 {s.rnr_n0}", f"prior {s.prior}",
                  f"max_jam_spr {s.max_jam_spr}", f"jam_free_bb {s.jam_free_bb}",
                  f"equity_samples {s.equity_samples}"]
        for k, v in s.menus.items():
            if v:
                lines.append(k + " " + " ".join(f"{x:g}" for x in v))
    lines.append(f"seed {seed}")
    if counts:
        for seat, c in counts.items():
            if seat in pos and seat != hero_seat:
                for ctx in range(NCTX):
                    if c[ctx].sum() > 0:
                        lines.append(f"model {pos[seat]} {ctx} {c[ctx, 0]:g} {c[ctx, 1]:g} {c[ctx, 2]:g}")
    return "\n".join(lines) + "\n", order


class OpponentStats:
    """Counts of each player's fold / check-call / bet-raise per context.

    The bots combine these counts with a Dirichlet prior centred on the
    blueprint's own frequencies (a Bayesian update); see pb_runtime.cpp."""

    def __init__(self, engine: Engine, num_seats: int, payouts=(0.65, 0.35)):
        self.engine = engine
        self.payouts = payouts
        self.counts = {s: np.zeros((NCTX, 3)) for s in range(num_seats)}
        self.seen: set[int] = set()

    def update(self, st: HandState) -> None:
        if st.hand_id in self.seen:
            return
        self.seen.add(st.hand_id)
        if len(seat_order(st)) < 2:
            return
        req, order = build_request(st, None, payouts=self.payouts)
        for p, ctx, cat in self.engine.contexts(req):
            self.counts[order[p]][ctx, cat] += 1

    def rates(self, seat: int) -> dict:
        """Raw observed rates for the named statistics."""
        c = self.counts[seat]

        def rate(ctx, cat):
            n = c[ctx].sum()
            return (c[ctx, cat] / n if n else float("nan")), int(n)
        return {"open (raise first in)": rate(0, 2), "fold to 3-bet+": rate(3, 0),
                "flop c-bet": rate(5, 2), "river call vs bet": rate(14, 1)}


class MCCFRBot(Agent):
    """Blueprint + depth-limited search + restricted Nash response."""

    def __init__(self, name: str, engine: Engine, settings: SearchSettings | None = None,
                 payouts=(0.65, 0.35), seed: int | None = None,
                 stats: OpponentStats | None = None, num_seats: int = 6, log=None):
        self.name = name
        self.engine = engine
        self.settings = settings or SearchSettings()
        self.payouts = payouts
        self.rng = random.Random(seed)
        self.stats = stats or OpponentStats(engine, num_seats, payouts)
        self.log = log
        self.last: dict | None = None

    def evaluate(self, st: HandState, seat: int) -> dict:
        req, _ = build_request(st, seat, payouts=self.payouts, counts=self.stats.counts,
                               settings=self.settings, seed=self.rng.randrange(1 << 30))
        return self.engine.decide(req)

    def act(self, obs: Observation) -> Decision:
        res = self.evaluate(obs.state, obs.seat)
        self.last = res
        opts = res["options"]
        probs = np.array([o["prob"] for o in opts], dtype=float)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(opts)) / len(opts)
        i = self.rng.choices(range(len(opts)), weights=probs)[0]
        o = opts[i]
        if self.log is not None:
            plan = ", ".join(f"{p['label']} {p['prob']:.0%}" for p in opts if p["prob"] >= 0.005)
            how = (f"search {res['search_iterations']} it" if res["searched"] else "blueprint")
            self.log(f"{self.name} plan: {plan} (equity {res['equity']:.0%}, {how})")
        if o["code"] == 0:
            d = Decision.fold()
        elif o["code"] == 1:
            d = Decision.call() if obs.legal.to_call else Decision.check()
        else:
            d = Decision.raise_to(int(o["to"]))
        return sanitize(d, obs.legal)

    def hand_finished(self, state: HandState, seat: int) -> None:
        self.stats.update(state)


def mccfr_bots(cfg: CfrConfig, k: int = 5, settings: SearchSettings | None = None,
               seed: int | None = None, num_seats: int = 6, log=None,
               names: list[str] | None = None) -> list[MCCFRBot]:
    engine = cfg.engine()
    settings = settings or SearchSettings()
    if not settings.menus:
        settings.menus = cfg.search_menus()
    stats = OpponentStats(engine, num_seats, cfg.payouts)
    names = names or [f"Bot{i + 1}" for i in range(k)]
    base = random.Random(seed)
    return [MCCFRBot(n, engine, settings, cfg.payouts, base.randrange(1 << 30), stats,
                     num_seats, log) for n in names]
