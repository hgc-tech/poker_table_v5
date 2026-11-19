"""Review your decisions against the trained bots.

ReviewRecorder wraps your player, and recorder.watch(bot) wraps each other
player, so every decision is recorded with exactly the information the acting
player had: the history of all previous hands from that seat and the current
hand with only that player's cards. After the tournament, write_reviews()
asks each of the top six bots what it would have done in every spot and
writes one Markdown file per hand, walking through the action step by step,
plus a summary of your own decisions.
"""
from __future__ import annotations

import csv
import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .agents import Agent, Observation, sanitize
from .cards import best_hand, card_str, cards_str, hand_name
from .features import (A_ALLIN, A_CALL, A_FOLD, ACTION_NAMES, NUM_ACTIONS, RAISE_ACTIONS,
                       RAISE_NAMES, action_to_decision,
                       features, legal_mask)
from .neural import NeuralBot
from .state import STREETS, ActionType, Decision, HandState, LegalActions

ACTION_LABELS = (["Fold", "Check/call"] + [f"Raise {n}" for n in RAISE_NAMES]
                 + ["All-in"])


@dataclass
class Spot:
    hand_id: int
    seat: int
    history_len: int
    current: np.ndarray
    legal: LegalActions
    decision: Decision
    street: int
    board: list
    pot: int
    actions_before: int
    stack: int = 0


@dataclass
class HandRecord:
    state: HandState | None = None
    spots: list = field(default_factory=list)


class ReviewRecorder(Agent):
    """Wraps an agent (normally you) and records its decisions.

    `watch(agent)` wraps another player so that its decisions are recorded in
    the same log; the review then shows the bots' view of every decision."""

    def __init__(self, inner: Agent, hands: dict | None = None):
        self.inner = inner
        self.name = inner.name
        self.is_human = getattr(inner, "is_human", False)
        self.hands: dict[int, HandRecord] = {} if hands is None else hands
        self.table = None

    def watch(self, agent: Agent) -> "ReviewRecorder":
        return ReviewRecorder(agent, self.hands)

    def act(self, obs: Observation) -> Decision:
        d = sanitize(self.inner.act(obs), obs.legal)
        st = obs.state
        rec = self.hands.setdefault(st.hand_id, HandRecord())
        rec.spots.append(Spot(st.hand_id, obs.seat, int(obs.history.shape[0]),
                              obs.current.numpy().copy(), obs.legal, d, st.street,
                              list(st.board), st.pot, len(st.actions),
                              st.stacks[obs.seat]))
        return d

    def show_or_muck(self, state: HandState, seat: int) -> bool:
        return self.inner.show_or_muck(state, seat)

    def hand_finished(self, state: HandState, seat: int) -> None:
        self.hands.setdefault(state.hand_id, HandRecord()).state = state
        self.inner.hand_finished(state, seat)


def _policy_probs(bot: NeuralBot, history, row):
    bot.model.update(history)
    f, sit = features(row, bot.model, bot.genes["width_mult"])
    mask = legal_mask(sit)
    z = np.where(mask, bot.logits(f), -1e9) / bot.genes["temp"]
    p = np.exp(z - z.max())
    p /= p.sum()
    return p, f, sit, mask


def _classify(d: Decision, sit, legal: LegalActions) -> int:
    """Map your decision onto the bots' five action buckets."""
    if d.kind == "fold":
        return A_FOLD
    if d.kind in ("check", "call"):
        return A_CALL
    if d.amount >= legal.max_raise_to:
        return A_ALLIN
    sizes = {a: action_to_decision(a, sit).amount for a in RAISE_ACTIONS}
    return min(sizes, key=lambda a: abs(d.amount - sizes[a]))


def _same_actions(d: Decision, sit, legal: LegalActions, mine: int) -> list[int]:
    """Every policy action that produces the decision actually taken. A raise
    size that would commit most of the stack is converted to all-in, so
    several sizes can lead to the same all-in."""
    if mine != A_ALLIN:
        return [mine]
    same = [a for a in RAISE_ACTIONS
            if action_to_decision(a, sit).amount >= legal.max_raise_to]
    return same + [A_ALLIN]


COL_NAMES = ["Fold", "Check/call", "1/4 pot", "1/3 pot", "1/2 pot", "2/3 pot", "Pot",
             "2x pot", "2.5x pot", "3x pot", "All-in"]


def positions(st: HandState) -> dict[int, str]:
    """Position names for the players dealt into this hand."""
    n = st.num_seats
    order = [(st.button + k) % n for k in range(n)]
    live = [s for s in order if st.dealt_in[s]]
    if len(live) == 2:
        return {st.sb_seat: "BTN/SB", st.bb_seat: "BB"}
    names = {live[0]: "BTN", st.sb_seat: "SB", st.bb_seat: "BB"}
    rest = [s for s in live if s not in names]
    later = ["UTG", "HJ", "CO"][-len(rest):] if rest else []
    names.update(zip(rest, later))
    return names


def _action_text(a) -> str:
    k = a.kind
    if k == ActionType.FOLD:
        t = "folds"
    elif k == ActionType.CHECK:
        t = "checks"
    elif k == ActionType.CALL:
        t = f"calls {a.amount:,}"
    elif k == ActionType.BET:
        t = f"bets {a.to:,}"
    elif k == ActionType.RAISE:
        t = f"raises to {a.to:,}"
    elif k == ActionType.POST_SB:
        t = f"posts SB {a.amount:,}"
    else:
        t = f"posts BB {a.amount:,}"
    return t + (" **(all-in)**" if a.all_in else "")


def _signed(x: int) -> str:
    return f"+{x:,}" if x > 0 else f"{x:,}"


class _Judges:
    """The top bots, with one opponent model per (bot, seat) so each
    decision is judged from the acting player's own history."""

    def __init__(self, genomes, history_buffer):
        self.genomes = [g if isinstance(g, torch.Tensor) else torch.tensor(g)
                        for g in genomes]
        self.history = history_buffer
        self.by_seat: dict[int, list[NeuralBot]] = {}
        self.tensors: dict[int, torch.Tensor] = {}

    def __len__(self):
        return len(self.genomes)

    def judge(self, spot: Spot):
        if spot.seat not in self.by_seat:
            self.by_seat[spot.seat] = [NeuralBot(g) for g in self.genomes]
            self.tensors[spot.seat] = self.history.tensor(spot.seat)
        history = self.tensors[spot.seat][:spot.history_len]
        res = [_policy_probs(b, history, spot.current) for b in self.by_seat[spot.seat]]
        probs = np.array([r[0] for r in res])
        _, f, sit, mask = res[0]
        return probs, f, sit, mask


def _table(rows: list[list[str]], right: set[int]) -> list[str]:
    """Fixed-width text table. Columns in `right` are right-aligned."""
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    out = []
    for n, r in enumerate(rows):
        cells = [r[c].rjust(widths[c]) if c in right else r[c].ljust(widths[c])
                 for c in range(len(r))]
        out.append("  ".join(cells).rstrip())
        if n == 0:
            out.append("  ".join("-" * w for w in widths))
    return out


def _cards(cards) -> str:
    return " ".join(card_str(c, pretty=False) for c in cards) if cards else "--"


SHORT_COLS = ["Fold", "Chk/Call", "1/4", "1/3", "1/2", "2/3", "1x", "2x", "2.5x",
              "3x", "All-in"]


def write_reviews(recorder: ReviewRecorder, history_buffer, genomes,
                  out_dir: Path | str | None = None,
                  reveal_all: bool = False, hero_seat: int = 0) -> Path:
    """Write review files. Returns the output directory.

    hand_NNN.txt walks through each hand action by action in aligned text
    columns. Every decision shows the acting player's cards and stack and the
    average probability the top bots give each action, judged with only what
    that player could see. hand_NNN.csv holds the same tables for a
    spreadsheet, and summary.txt reviews your own decisions."""
    if out_dir is None:
        stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = Path("reviews") / stamp
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    judges = _Judges(genomes, history_buffer)
    nb = len(judges)

    summary_rows = []
    total, agree_top, agree_any = 0, 0, 0
    flagged = []
    header = (["Player", "Pos", "Cards", "Stack", "Pot", "To call", "Action", "Equity"]
              + SHORT_COLS)
    right = {3, 4, 5} | set(range(7, len(header)))
    for hand_id in sorted(recorder.hands):
        rec = recorder.hands[hand_id]
        st = rec.state
        if st is None:
            continue
        pos = positions(st)
        spots = {sp.actions_before: sp for sp in rec.spots}
        hero_dec, hand_flags = 0, 0
        csv_rows = [[f"Hand {hand_id}", f"Level {st.level}",
                     f"Blinds {st.small_blind}/{st.big_blind}", f"Ante {st.ante}"], []]
        lines = [f"HAND {hand_id}", "=" * 60,
                 f"Level {st.level}, blinds {st.small_blind:,}/{st.big_blind:,}"
                 + (f", ante {st.ante:,}" if st.ante else ""),
                 f"You hold {_cards(st.hole_cards[hero_seat])} in the "
                 f"{pos.get(hero_seat, '?')} and finish {_signed(st.net[hero_seat])}.", "",
                 f"Percentages are the average probability the top {nb} bots give",
                 "each action in that player's spot, seeing only what that player saw.",
                 "The action taken is in [brackets]; '-' means not allowed.",
                 "Raise sizes are fractions of the pot after calling. A 1/2 raise",
                 "raises to the current bet plus 1/2 x (Pot + To call); a 1/2 bet",
                 "is 1/2 x Pot. Sizes below the minimum raise become the minimum,",
                 "and sizes reaching 60% of the most the player can bet become all-in;",
                 "every size that leads to the action taken is in [brackets].",
                 "Equity is against the opponents' modeled ranges. Stack and Pot are",
                 "before the action. Cards are shown for every player, tabled or not."]
        if reveal_all:
            lines.append("Diagnostic mode: all hole cards were visible during play.")
        lines.append("")
        for s in range(4):
            acts = [(k, a) for k, a in enumerate(st.actions) if a.street == s]
            n_board = [0, 3, 4, 5][s]
            if s > 0:
                if len(st.board) < n_board:
                    break
                new = st.board[[0, 0, 3, 4][s]:n_board]
                title = f"{STREETS[s].upper()}: {_cards(new)}"
                if s > 1:
                    title += f"   (board {_cards(st.board[:n_board])})"
            else:
                title = "PREFLOP"
            lines += [title, "-" * len(title)]
            csv_rows.append([title])
            if not acts:
                lines += ["No betting (players all-in).", ""]
                continue
            rows = [header]
            for k, a in acts:
                name = st.names[a.seat]
                is_hero = a.seat == hero_seat
                cells = [name + (" *" if is_hero else ""), pos.get(a.seat, ""),
                         _cards(st.hole_cards[a.seat])]
                spot = spots.get(k)
                act = _action_text(a).replace("**", "")
                if spot is None:          # blinds
                    before = st.starting_stacks[a.seat] - st.antes_paid[a.seat]
                    rows.append(cells + [f"{before:,}", f"{a.pot_before:,}", "", act, ""]
                                + [""] * NUM_ACTIONS)
                    continue
                probs, f0, sit0, mask0 = judges.judge(spot)
                avg = probs.mean(0)
                mine = _classify(spot.decision, sit0, spot.legal)
                same = _same_actions(spot.decision, sit0, spot.legal, mine)
                p_mine = float(avg[same].sum())
                pc = []
                for c in range(NUM_ACTIONS):
                    if not mask0[c]:
                        pc.append("-")
                    elif c in same:
                        pc.append(f"[{avg[c]:.0%}]")
                    else:
                        pc.append(f"{avg[c]:.0%}")
                rows.append(cells + [f"{spot.stack:,}", f"{a.pot_before:,}",
                                     f"{a.to_call_before:,}", act, f"{f0[26]:.0%}"] + pc)
                if is_hero:
                    hero_dec += 1
                    total += 1
                    tops = probs.argmax(1)
                    agree_top += int(np.isin(tops, same).sum())
                    agree_any += int(p_mine >= 0.10)
                    if p_mine < 0.10:
                        hand_flags += 1
                        best = int(avg.argmax())
                        flagged.append((hand_id, hero_dec, STREETS[s],
                                        ACTION_LABELS[mine], ACTION_LABELS[best], p_mine))
            lines += _table(rows, right) + [""]
            csv_rows += rows + [[]]

        lines += ["RESULT", "------"]
        res_start = len(lines)
        for p in range(st.num_seats):
            if st.shown[p]:
                lines.append(f"{st.names[p]} shows {_cards(st.hole_cards[p])}: "
                             f"{hand_name(best_hand(st.hole_cards[p] + st.board))}")
            elif st.mucked[p]:
                lines.append(f"{st.names[p]} mucks {_cards(st.hole_cards[p])}")
        for p in range(st.num_seats):
            if st.net[p] > 0:
                how = (f" with {hand_name(best_hand(st.hole_cards[p] + st.board))}"
                       if st.shown[p] else "")
                lines.append(f"Winner: {st.names[p]} ({_signed(st.net[p])}){how}")
        result_text = lines[res_start:]
        rows = [["Player", "Start", "End", "Net"]]
        for p in range(st.num_seats):
            if st.dealt_in[p]:
                rows.append([st.names[p], f"{st.starting_stacks[p]:,}",
                             f"{st.stacks[p]:,}", _signed(st.net[p])])
        lines += [""] + _table(rows, {1, 2, 3}) + [""]
        csv_rows += [["RESULT"]] + [[x] for x in result_text]
        csv_rows += [[]] + rows
        fname = f"hand_{hand_id:03d}.txt"
        (out / fname).write_text("\n".join(lines), encoding="utf-8")
        with open(out / f"hand_{hand_id:03d}.csv", "w", newline="",
                  encoding="utf-8") as fh:
            csv.writer(fh).writerows(csv_rows)
        summary_rows.append((hand_id, _cards(st.hole_cards[hero_seat]),
                             pos.get(hero_seat, ""), st.net[hero_seat], hero_dec,
                             hand_flags, fname))

    head = ["TOURNAMENT REVIEW" + (" (diagnostic mode)" if reveal_all else ""),
            "=" * 60,
            f"Every decision in every hand was shown to the top {nb} trained bots.",
            "Each bot saw exactly what the acting player saw: every previous hand",
            "from that seat and the current hand with only that player's cards.",
            "hand_NNN.txt walks through each hand in aligned text columns;",
            "hand_NNN.csv holds the same tables for a spreadsheet.", ""]
    if total:
        head += [f"Your decisions reviewed:                       {total}",
                 f"Bots whose top choice matched yours (average): "
                 f"{agree_top / total:.1f} of {nb}",
                 f"Your decisions the bots give at least 10%:      "
                 f"{agree_any / total:.0%}", ""]
    if flagged:
        head += ["YOUR DECISIONS TO LOOK AT (bots give your play under 10%)",
                 "-" * 57]
        rows = [["Hand", "Decision", "Street", "You", "Bots preferred", "Bots' prob."]]
        rows += [[str(h), str(k), s, mine, pref, f"{p:.0%}"]
                 for h, k, s, mine, pref, p in flagged]
        head += _table(rows, {0, 1, 5}) + [""]
    head += ["HANDS", "-----"]
    rows = [["Hand", "Your cards", "Pos", "Result", "Decisions", "Flagged", "File"]]
    rows += [[str(h), c, p, _signed(net), str(n), str(fl or ""), f]
             for h, c, p, net, n, fl, f in summary_rows]
    head += _table(rows, {0, 3, 4, 5})
    (out / "summary.txt").write_text("\n".join(head) + "\n", encoding="utf-8")
    return out
