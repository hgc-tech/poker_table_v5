"""Tournament table: deals hands, runs betting, pays pots, raises blinds."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable

from .agents import Agent, Observation, sanitize
from .cards import Deck, best_hand, cards_str, hand_name
from .handstrength import score as fast_score
from .encoding import HistoryBuffer, encode_hand
from .state import (STREETS, Action, ActionType, Decision, HandState,
                    LegalActions, describe_action)


@dataclass(frozen=True)
class BlindLevel:
    small_blind: int
    big_blind: int
    ante: int = 0
    hands: int = 8          # hands played at this level


# Tournament structure: 2,000 chips (100 big blinds), levels measured in hands.
# After the last level the blinds stay there.
DEFAULT_SCHEDULE = [
    BlindLevel(10, 20, 2, 8), BlindLevel(15, 30, 3, 8),
    BlindLevel(20, 40, 4, 8), BlindLevel(25, 50, 5, 8),
    BlindLevel(30, 60, 6, 8), BlindLevel(40, 80, 8, 8),
    BlindLevel(50, 100, 10, 8), BlindLevel(60, 120, 12, 8),
    BlindLevel(75, 150, 15, 8), BlindLevel(100, 200, 20, 6),
    BlindLevel(125, 250, 25, 6), BlindLevel(150, 300, 30, 6),
    BlindLevel(200, 400, 40, 6), BlindLevel(250, 500, 50, 6),
    BlindLevel(300, 600, 60, 6), BlindLevel(350, 700, 70, 6),
    BlindLevel(400, 800, 80, 6), BlindLevel(500, 1000, 100, 6),
]
STARTING_STACK = 2000


class Table:
    def __init__(self, agents: list[Agent], starting_stack: int = STARTING_STACK,
                 schedule: list[BlindLevel] | None = None,
                 seed: int | None = None,
                 log: Callable[[str], None] | None = None,
                 bot_delay: float = 0.0,
                 observer: Callable[[HandState, str, str], None] | None = None):
        """bot_delay: seconds to pause after each bot action (for human play).
        observer: called as observer(state, message, kind) on every event;
        kind is "log", "deal", "action", "street", "showdown" or "hand_end"."""
        self.agents = agents
        self.n = len(agents)
        self.names = [a.name for a in agents]
        self.stacks = [starting_stack] * self.n
        self.schedule = schedule or DEFAULT_SCHEDULE
        self.bot_delay = bot_delay
        self.rng = random.Random(seed)
        self.log = log or (lambda msg: None)
        self.log_enabled = log is not None
        self.observer = observer
        self.history = HistoryBuffer(self.n)
        self.hand_count = 0
        self.button = self.rng.randrange(self.n)
        self.finish_order: list[int] = []   # seats, first eliminated first

    # Tournament loop ----------------------------------------------------
    def alive(self) -> list[int]:
        return [s for s in range(self.n) if self.stacks[s] > 0]

    def level_index(self) -> int:
        remaining = self.hand_count
        for i, lvl in enumerate(self.schedule):
            if remaining < lvl.hands:
                return i
            remaining -= lvl.hands
        return len(self.schedule) - 1

    def run(self, max_hands: int | None = None,
            stop_when_busted: int | None = None) -> list[int]:
        """Play until one player remains. Returns seats ranked 1st..last.

        stop_when_busted: stop early once this seat is eliminated.
        """
        while len(self.alive()) > 1:
            if max_hands is not None and self.hand_count >= max_hands:
                break
            self.play_hand()
            if stop_when_busted is not None and self.stacks[stop_when_busted] == 0:
                break
        remaining = sorted(self.alive(), key=lambda s: -self.stacks[s])
        return remaining + self.finish_order[::-1]

    # One hand -----------------------------------------------------------
    def play_hand(self) -> HandState:
        alive = self.alive()
        lvl = self.schedule[self.level_index()]
        if self.hand_count > 0:
            self.button = self._next(self.button, alive)
        elif self.button not in alive:
            self.button = self._next(self.button, alive)
        st = HandState(
            hand_id=self.hand_count + 1, num_seats=self.n, names=self.names,
            button=self.button, small_blind=lvl.small_blind,
            big_blind=lvl.big_blind, ante=lvl.ante,
            level=self.level_index() + 1, starting_stacks=list(self.stacks))
        self._emit(st, f"\n=== Hand #{st.hand_id}  blinds {lvl.small_blind}/"
                 f"{lvl.big_blind}" + (f" ante {lvl.ante}" if lvl.ante else "")
                 + f"  button: {self.names[self.button]} ===")

        deck = Deck(self.rng)
        if len(alive) == 2:
            sb, bb = self.button, self._next(self.button, alive)
        else:
            sb = self._next(self.button, alive)
            bb = self._next(sb, alive)
        st.sb_seat, st.bb_seat = sb, bb

        for s in alive:
            if st.ante:
                paid = min(st.ante, st.stacks[s])
                st.stacks[s] -= paid
                st.committed_total[s] += paid
                st.antes_paid[s] = paid
        if st.ante:
            self._emit(st, f"Everyone antes {st.ante} (pot {st.pot})")
        self._post(st, sb, st.small_blind, ActionType.POST_SB)
        self._post(st, bb, st.big_blind, ActionType.POST_BB)
        for s in alive:
            st.hole_cards[s] = deck.deal(2)
        self._emit(st, "Cards are dealt.", "deal")

        for street in range(4):
            if street > 0:
                if len(st.live_seats()) < 2:
                    break
                st.street = street
                st.street_reached[street] = True
                st.pot_at_street_start[street] = st.pot
                st.committed_street = [0] * self.n
                st.board += deck.deal(3 if street == 1 else 1)
                self._emit(st, f"--- {STREETS[street]}: {cards_str(st.board)}  "
                           f"(pot {st.pot})", "street")
            if len(st.live_seats()) < 2:
                break
            if street == 0:
                first = (self.button if len(alive) == 2
                         else self._next(bb, alive))
            else:
                first = self._next(self.button, st.live_seats())
            self._betting_round(st, first)

        self._settle(st)
        self.stacks = list(st.stacks)
        for s in range(self.n):
            st.net[s] = st.stacks[s] - st.starting_stacks[s]
        busted = [s for s in alive if self.stacks[s] == 0]
        # Players busted in the same hand: smaller starting stack finishes lower.
        for s in sorted(busted, key=lambda s: st.starting_stacks[s]):
            self.finish_order.append(s)
            self._emit(st, f"{self.names[s]} is eliminated.")
        st.finished = True
        self._emit(st, "", "hand_end")
        self.hand_count += 1
        self.history.add(st)
        for s, agent in enumerate(self.agents):
            agent.hand_finished(st, s)
        return st

    # Helpers ------------------------------------------------------------
    def _emit(self, st: HandState, msg: str, kind: str = "log") -> None:
        if self.log_enabled and msg:
            self.log(msg)
        if self.observer is not None:
            self.observer(st, msg, kind)

    def _next(self, seat: int, among: list[int]) -> int:
        for k in range(1, self.n + 1):
            cand = (seat + k) % self.n
            if cand in among:
                return cand
        raise RuntimeError("no seat available")

    def _post(self, st: HandState, seat: int, amount: int, kind: ActionType):
        paid = min(amount, st.stacks[seat])
        pot_before = st.pot
        st.stacks[seat] -= paid
        st.committed_street[seat] += paid
        st.committed_total[seat] += paid
        a = Action(seat, 0, kind, paid, st.committed_street[seat], pot_before,
                   0, all_in=st.stacks[seat] == 0)
        st.actions.append(a)
        self._emit(st, describe_action(st, a), "action")

    def _betting_round(self, st: HandState, first: int) -> None:
        n = self.n
        last_raise = st.big_blind            # minimum raise increment
        full_level = st.current_bet          # bet level after last full raise
        acted: dict[int, int] = {}           # seat -> full_level when it acted
        seat = first
        while True:
            live = st.live_seats()
            if len(live) < 2:
                return
            # Stop if nobody can still act.
            def needs(p):
                if st.stacks[p] == 0:
                    return False
                return p not in acted or st.committed_street[p] < st.current_bet
            pending = [p for p in live if needs(p)]
            if not pending:
                return
            # Everyone else all-in and this player already matched: done.
            with_chips = [p for p in live if st.stacks[p] > 0]
            if (len(with_chips) == 1 and
                    st.committed_street[with_chips[0]] >= st.current_bet):
                return
            if seat not in pending:
                seat = (seat + 1) % n
                continue

            to_call = min(st.current_bet - st.committed_street[seat], st.stacks[seat])
            others_with_chips = [p for p in live if p != seat and st.stacks[p] > 0]
            max_to = st.committed_street[seat] + st.stacks[seat]
            reopened = seat not in acted or acted[seat] < full_level
            can_raise = (st.stacks[seat] > to_call and reopened
                         and bool(others_with_chips))
            min_to = min(st.current_bet + last_raise, max_to)
            legal = LegalActions(to_call, can_raise, min_to, max_to)

            d = self._ask(st, seat, legal)
            pot_before = st.pot
            if d.kind == "fold":
                st.folded[seat] = True
                a = Action(seat, st.street, ActionType.FOLD, 0,
                           st.committed_street[seat], pot_before, to_call)
            elif d.kind in ("check", "call"):
                pay = to_call
                st.stacks[seat] -= pay
                st.committed_street[seat] += pay
                st.committed_total[seat] += pay
                kind = ActionType.CALL if pay > 0 else ActionType.CHECK
                a = Action(seat, st.street, kind, pay, st.committed_street[seat],
                           pot_before, to_call, all_in=st.stacks[seat] == 0)
            else:  # raise
                target = d.amount
                pay = target - st.committed_street[seat]
                increment = target - st.current_bet
                kind = ActionType.BET if st.current_bet == 0 else ActionType.RAISE
                if increment >= last_raise:
                    last_raise = increment
                    full_level = target
                st.stacks[seat] -= pay
                st.committed_street[seat] = target
                st.committed_total[seat] += pay
                a = Action(seat, st.street, kind, pay, target, pot_before,
                           to_call, all_in=st.stacks[seat] == 0)
            acted[seat] = full_level
            st.actions.append(a)
            self._emit(st, describe_action(st, a), "action")
            if self.bot_delay and not getattr(self.agents[seat], "is_human", False):
                time.sleep(self.bot_delay)
            seat = (seat + 1) % n

    def _ask(self, st: HandState, seat: int, legal: LegalActions) -> Decision:
        agent = self.agents[seat]
        st.to_act = seat
        window = getattr(agent, "history_window", None)
        obs = Observation(
            seat=seat,
            history=self.history.tensor(seat, last=window),
            current=encode_hand(st, seat, current=True, legal=legal),
            legal=legal, state=st)
        d = agent.act(obs)
        st.to_act = -1
        return sanitize(d, legal)

    def _showdown(self, st: HandState, live: list[int], scores: dict) -> None:
        """Reveal hands in showdown order.

        If anyone is all-in, every hand is tabled (standard tournament rule).
        Otherwise the last bettor or raiser on the final street shows first
        (or, with no bet, the first player left of the button). Each later
        player shows if their hand beats or ties the best hand shown so far,
        and otherwise chooses to show or muck. Mucked hands stay hidden.
        """
        all_in = any(st.stacks[p] == 0 for p in live)
        aggr = [a.seat for a in st.actions
                if a.street == st.street and a.kind in (ActionType.BET, ActionType.RAISE)
                and a.seat in live]
        if aggr:
            first = aggr[-1]
        else:
            first = min(live, key=lambda p: (p - st.button - 1) % self.n)
        order = sorted(live, key=lambda p: (p - first) % self.n)
        best = None
        for p in order:
            if all_in or best is None or scores[p] >= best:
                show = True
            else:
                agent = self.agents[p]
                show = bool(getattr(agent, "show_or_muck", lambda *a: False)(st, p))
            if show:
                st.shown[p] = True
                best = scores[p] if best is None else max(best, scores[p])
                if self.log_enabled or self.observer is not None:
                    name = hand_name(best_hand(st.hole_cards[p] + st.board))
                    self._emit(st, f"{self.names[p]} shows "
                               f"{cards_str(st.hole_cards[p])} ({name})",
                               "showdown")
            else:
                st.mucked[p] = True
                self._emit(st, f"{self.names[p]} mucks", "showdown")

    def _settle(self, st: HandState) -> None:
        live = st.live_seats()
        showdown = len(live) > 1
        scores = {p: fast_score(st.hole_cards[p] + st.board) for p in live} \
            if showdown else {}
        if showdown:
            self._showdown(st, live, scores)
        contrib = list(st.committed_total)
        won = [0] * self.n
        returned = [0] * self.n
        levels = sorted({c for c in contrib if c > 0})
        prev = 0
        for level in levels:
            portion = sum(min(c, level) - min(c, prev) for c in contrib)
            prev_level, prev = prev, level
            if portion == 0:
                continue
            eligible = [p for p in live if contrib[p] >= level]
            if not eligible:
                # Everyone who reached this level folded; give it to the
                # remaining live player(s) with the largest contribution.
                eligible = [p for p in live if contrib[p] > prev_level] or live
            if len(eligible) == 1:
                winners = eligible
            else:
                best = max(scores[p] for p in eligible)
                winners = [p for p in eligible if scores[p] == best]
            share, odd = divmod(portion, len(winners))
            # Odd chips go to winners closest to the left of the button.
            ordered = sorted(winners, key=lambda p: (p - st.button - 1) % self.n)
            for i, p in enumerate(ordered):
                st.stacks[p] += share + (1 if i < odd else 0)
            uncalled = len(eligible) == 1 and all(
                min(c, level) - min(c, prev_level) == 0
                for q, c in enumerate(contrib) if q != eligible[0])
            for i, p in enumerate(ordered):
                amt = share + (1 if i < odd else 0)
                (returned if uncalled else won)[p] += amt
        for p in range(self.n):
            if won[p]:
                extra = f" (and {returned[p]} uncalled returned)" if returned[p] else ""
                self._emit(st, f"{self.names[p]} wins {won[p]}{extra}", "win")
            elif returned[p]:
                self._emit(st, f"Uncalled {returned[p]} returned to {self.names[p]}")
