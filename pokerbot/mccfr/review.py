"""Decision review against the MCCFR bot.

Same files and layout as pokerbot/review.py (hand_NNN.txt, hand_NNN.csv,
summary.txt). Each decision is shown to a fresh MCCFR bot that knows only
what the acting player knew: the public actions of every earlier hand (for
its opponent model) and the current hand with only that player's cards. The
percentages are the bot's final strategy after its search.
"""
from __future__ import annotations

import csv
import datetime as _dt
import sys
from pathlib import Path

from ..cards import best_hand, hand_name
from ..review import _action_text, _cards, _signed, _table, positions
from ..state import STREETS, ActionType
from .bot import OpponentStats, SearchSettings, build_request


def _size_label(f: float) -> str:
    return f"{round(f * 100):d}%"


class MCCFRJudge:
    def __init__(self, cfg, settings: SearchSettings):
        self.cfg = cfg
        self.engine = cfg.engine()
        self.settings = settings
        sizes = set()
        for bp in self.engine.info()["blueprints"]:
            sizes.update(round(s, 4) for s in bp["sizes"])
        self.sizes = sorted(sizes)
        self.columns = ["Fold", "Chk/Call"] + [_size_label(s) for s in self.sizes] + ["All-in"]

    def col_of(self, opt) -> int:
        if opt["code"] == 0:
            return 0
        if opt["code"] == 1:
            return 1
        if opt["code"] == 15:
            return len(self.columns) - 1
        return 2 + min(range(len(self.sizes)), key=lambda i: abs(self.sizes[i] - opt["pot_frac"]))

    def judge(self, st, spot, counts, seed):
        req, _ = build_request(st, spot.seat, n_actions=spot.actions_before, board=spot.board,
                               payouts=self.cfg.payouts, counts=counts, settings=self.settings,
                               seed=seed)
        res = self.engine.decide(req)
        ncol = len(self.columns)
        probs = [None] * ncol          # None = not available
        for o in res["options"]:
            c = self.col_of(o)
            probs[c] = (probs[c] or 0.0) + o["prob"]
        snapped = {2 + min(range(len(self.sizes)), key=lambda i: abs(self.sizes[i] - f))
                   for f in res["snapped"]}
        # Columns of the decision actually taken.
        d = spot.decision
        if d.kind == "fold":
            same = [0]
        elif d.kind in ("check", "call"):
            same = [1]
        elif d.amount >= res["max_to"]:
            same = [ncol - 1] + sorted(snapped)
        else:
            raises = [o for o in res["options"] if o["code"] >= 2]
            best = min(raises, key=lambda o: abs(o["to"] - d.amount)) if raises else None
            same = [self.col_of(best)] if best else [1]
        return res, probs, snapped, same


def write_mccfr_reviews(recorder, cfg, settings: SearchSettings | None = None,
                        out_dir: Path | str | None = None, reveal_all: bool = False,
                        hero_seat: int = 0, progress=True) -> Path:
    settings = settings or SearchSettings(search_ms=300, threads=2)
    if out_dir is None:
        out_dir = Path("reviews") / _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    judge = MCCFRJudge(cfg, settings)
    hands = [h for h in sorted(recorder.hands) if recorder.hands[h].state is not None]
    n_spots = sum(len(recorder.hands[h].spots) for h in hands)
    num_seats = recorder.hands[hands[0]].state.num_seats if hands else 6
    stats = OpponentStats(judge.engine, num_seats, cfg.payouts)
    header = (["Player", "Pos", "Cards", "Stack", "Pot", "To call", "Action", "Equity"] + judge.columns)
    right = {3, 4, 5} | set(range(7, len(header)))
    how = ("blueprint only" if settings.mode != "search"
           else f"blueprint plus {settings.search_ms:g} ms of search per decision")
    summary_rows, flagged = [], []
    total = agree_top = agree_any = 0
    done = 0
    for hand_id in hands:
        rec = recorder.hands[hand_id]
        st = rec.state
        counts = {s: c.copy() for s, c in stats.counts.items()}
        pos = positions(st)
        spots = {sp.actions_before: sp for sp in rec.spots}
        hero_dec = hand_flags = 0
        lines = [f"HAND {hand_id}", "=" * 60,
                 f"Level {st.level}, blinds {st.small_blind:,}/{st.big_blind:,}"
                 + (f", ante {st.ante:,}" if st.ante else ""),
                 f"You hold {_cards(st.hole_cards[hero_seat])} in the "
                 f"{pos.get(hero_seat, '?')} and finish {_signed(st.net[hero_seat])}.", "",
                 "Percentages are the MCCFR bot's strategy in that player's spot",
                 f"({how}),",
                 "seeing only what that player saw: the public actions of every",
                 "earlier hand and that player's own cards.",
                 "The action taken is in [brackets]; '-' means not available.",
                 "Raise sizes are fractions of the pot after calling. A 75% raise",
                 "raises to the current bet plus 0.75 x (Pot + To call); a 75% bet",
                 f"is 0.75 x Pot. Sizes that reach {cfg.num('allin_snap', 0.6):.0%} of the most the",
                 "player can bet become all-in and are marked =AI; when the action taken was",
                 "all-in, those sizes are in [brackets] too.",
                 "Equity is against the opponents' ranges as the bot estimates them.",
                 "Stack and Pot are before the action. Cards are shown for every",
                 "player, tabled or not."]
        if reveal_all:
            lines.append("Diagnostic mode: all hole cards were visible during play.")
        lines.append("")
        csv_rows = [[f"Hand {hand_id}", f"Level {st.level}",
                     f"Blinds {st.small_blind}/{st.big_blind}", f"Ante {st.ante}"], []]
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
                is_hero = a.seat == hero_seat
                cells = [st.names[a.seat] + (" *" if is_hero else ""), pos.get(a.seat, ""),
                         _cards(st.hole_cards[a.seat])]
                act = _action_text(a).replace("**", "")
                spot = spots.get(k)
                if spot is None or a.kind in (ActionType.POST_SB, ActionType.POST_BB):
                    before = st.starting_stacks[a.seat] - st.antes_paid[a.seat]
                    rows.append(cells + [f"{before:,}", f"{a.pot_before:,}", "", act, ""]
                                + [""] * len(judge.columns))
                    continue
                res, probs, snapped, same = judge.judge(st, spot, counts, seed=hand_id * 1000 + k)
                done += 1
                if progress and done % 20 == 0:
                    print(f"  reviewed {done} of {n_spots} decisions", file=sys.stderr)
                pc = []
                for c, p in enumerate(probs):
                    if p is None:
                        txt = "=AI" if c in snapped else "-"
                    else:
                        txt = f"{p:.0%}"
                    pc.append(f"[{txt}]" if c in same else txt)
                p_mine = sum(probs[c] or 0.0 for c in same)
                eq = res.get("equity", -1)
                rows.append(cells + [f"{spot.stack:,}", f"{a.pot_before:,}", f"{a.to_call_before:,}",
                                     act, f"{eq:.0%}" if eq >= 0 else ""] + pc)
                if is_hero:
                    hero_dec += 1
                    total += 1
                    top = max(range(len(probs)), key=lambda c: probs[c] or 0.0)
                    agree_top += int(top in same)
                    agree_any += int(p_mine >= 0.10)
                    if p_mine < 0.10:
                        hand_flags += 1
                        flagged.append((hand_id, hero_dec, STREETS[s], judge.columns[same[0]],
                                        judge.columns[top], p_mine))
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
                w = (f" with {hand_name(best_hand(st.hole_cards[p] + st.board))}"
                     if st.shown[p] else "")
                lines.append(f"Winner: {st.names[p]} ({_signed(st.net[p])}){w}")
        result_text = lines[res_start:]
        rows = [["Player", "Start", "End", "Net"]]
        for p in range(st.num_seats):
            if st.dealt_in[p]:
                rows.append([st.names[p], f"{st.starting_stacks[p]:,}", f"{st.stacks[p]:,}",
                             _signed(st.net[p])])
        lines += [""] + _table(rows, {1, 2, 3}) + [""]
        csv_rows += [["RESULT"]] + [[x] for x in result_text] + [[]] + rows
        fname = f"hand_{hand_id:03d}.txt"
        (out / fname).write_text("\n".join(lines), encoding="utf-8")
        with open(out / f"hand_{hand_id:03d}.csv", "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(csv_rows)
        summary_rows.append((hand_id, _cards(st.hole_cards[hero_seat]), pos.get(hero_seat, ""),
                             st.net[hero_seat], hero_dec, hand_flags, fname))
        stats.update(st)

    head = ["TOURNAMENT REVIEW" + (" (diagnostic mode)" if reveal_all else ""), "=" * 60,
            "Every decision in every hand was shown to the MCCFR bot",
            f"({how}).",
            "The bot saw exactly what the acting player saw:",
            "the public actions of every earlier hand and the current hand",
            "with only that player's cards.",
            "hand_NNN.txt walks through each hand in aligned text columns;",
            "hand_NNN.csv holds the same tables for a spreadsheet.", ""]
    if total:
        head += [f"Your decisions reviewed:                      {total}",
                 f"Bot's most likely action matched yours:       {agree_top / total:.0%}",
                 f"Your decisions the bot gives at least 10%:    {agree_any / total:.0%}", ""]
    if flagged:
        head += ["YOUR DECISIONS TO LOOK AT (the bot gives your play under 10%)", "-" * 61]
        rows = [["Hand", "Decision", "Street", "You", "Bot preferred", "Bot's prob."]]
        rows += [[str(h), str(k), s, mine, pref, f"{p:.0%}"] for h, k, s, mine, pref, p in flagged]
        head += _table(rows, {0, 1, 5}) + [""]
    # How the bots read you at the end.
    head += ["HOW THE BOTS READ YOU", "---------------------",
             "Observed rates over the whole tournament. The bots blend these with",
             "the blueprint's own rates (a Bayesian prior), and the weight they put",
             "on the resulting model of you grows with the number of observations",
             f"(restricted Nash response, at most {settings.rnr_max:.0%}).", ""]
    rows = [["Statistic", "Your rate", "Observations"]]
    for name, (rate, n) in stats.rates(hero_seat).items():
        rows.append([name, "-" if n == 0 else f"{rate:.0%}", str(n)])
    n_obs = int(stats.counts[hero_seat].sum())
    w = settings.rnr_max * n_obs / (n_obs + settings.rnr_n0)
    head += _table(rows, {1, 2}) + ["", f"Decisions observed: {n_obs}; weight on the model of you: {w:.0%}", ""]
    head += ["HANDS", "-----"]
    rows = [["Hand", "Your cards", "Pos", "Result", "Decisions", "Flagged", "File"]]
    rows += [[str(h), c, p, _signed(net), str(n), str(fl or ""), f]
             for h, c, p, net, n, fl, f in summary_rows]
    head += _table(rows, {0, 3, 4, 5})
    (out / "summary.txt").write_text("\n".join(head) + "\n", encoding="utf-8")
    return out
