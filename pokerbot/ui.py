"""Rich terminal interface: players seated around an oval table.

TableView renders the table after every event. RichHuman is a text-command
player that uses the view. Both work in a normal terminal window; make it at
least 100 columns wide and 40 rows tall.
"""
from __future__ import annotations

import time
from collections import deque

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from .agents import Agent, Observation
import re

from .cards import RANKS, SUITS, best_hand, hand_name, rank_of, suit_of
from .state import STREETS, ActionType, Decision, HandState, LegalActions

W, H = 100, 25
CX, CY, RX, RY = 50, 12, 31, 7

# Screen anchors (left x, top y) for seat boxes, by position relative to the
# viewer, clockwise: 0 = bottom (viewer), 1 = bottom-left, 2 = top-left,
# 3 = top, 4 = top-right, 5 = bottom-right.
SEAT_BOX = {0: (41, 21), 1: (1, 15), 2: (1, 3), 3: (41, 0), 4: (80, 3), 5: (80, 15)}
# Where each seat's chips in front of it are drawn (centre x, y).
BET_POS = {0: (50, 18), 1: (28, 15), 2: (28, 8), 3: (50, 6), 4: (72, 8), 5: (72, 15)}
BOX_W = 19

FELT = "on #0b5d2a"
RAIL = "on #5a3a1a"
SUIT_CH = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}


class Canvas:
    def __init__(self, w=W, h=H):
        self.w, self.h = w, h
        self.ch = [[" "] * w for _ in range(h)]
        self.st = [[""] * w for _ in range(h)]

    def put(self, x, y, text, style=""):
        has_bg = style.startswith("on ") or " on " in style
        for i, c in enumerate(text):
            if 0 <= x + i < self.w and 0 <= y < self.h:
                self.ch[y][x + i] = c
                if has_bg:
                    self.st[y][x + i] = style
                else:
                    self.st[y][x + i] = f"{style} {_bg(self.st[y][x + i])}".strip()

    def put_center(self, cx, y, text, style=""):
        self.put(cx - len(text) // 2, y, text, style)

    def put_segments(self, cx, y, segs):
        """segs: list of (text, style); centred on cx."""
        total = sum(len(t) for t, _ in segs)
        x = cx - total // 2
        for t, s in segs:
            self.put(x, y, t, s)
            x += len(t)

    def fill(self, x, y, w, style):
        for i in range(w):
            if 0 <= x + i < self.w and 0 <= y < self.h:
                self.st[y][x + i] = style

    def to_text(self) -> Text:
        out = Text()
        for y in range(self.h):
            run, run_style = "", None
            for x in range(self.w):
                s = self.st[y][x].strip()
                if s != run_style and run:
                    out.append(run, style=run_style or None)
                    run = ""
                run_style = s
                run += self.ch[y][x]
            if run:
                out.append(run, style=run_style or None)
            out.append("\n")
        return out


def _bg(style: str) -> str:
    parts = style.split()
    for i, p in enumerate(parts):
        if p == "on" and i + 1 < len(parts):
            return f"on {parts[i + 1]}"
    return ""


_VERB = re.compile(r"^(You) (fold|check|call|bet|raise|post|show|win)s\b")


def _grammar(line: str) -> str:
    return _VERB.sub(r"\1 \2", line)


def card_segs(card: int | None, hidden=False):
    if hidden:
        return [("▒▒", "bold white on #2244aa")]
    if card is None:
        return [("  ", "on #0f7a38")]
    r, s = RANKS[rank_of(card)], SUITS[suit_of(card)]
    color = "#cc0000" if s in "hd" else "black"
    return [(r + SUIT_CH[s], f"bold {color} on white")]


def fmt(n: float) -> str:
    return f"{int(n):,}"


def last_action_text(st: HandState, seat: int) -> str:
    acts = [a for a in st.actions if a.seat == seat and a.street == st.street]
    if not acts:
        return ""
    a = acts[-1]
    k = a.kind
    if k == ActionType.FOLD:
        t = "fold"
    elif k == ActionType.CHECK:
        t = "check"
    elif k == ActionType.CALL:
        t = "call"
    elif k == ActionType.BET:
        t = f"bet {fmt(a.to)}"
    elif k == ActionType.RAISE:
        t = f"raise to {fmt(a.to)}"
    elif k == ActionType.POST_SB:
        t = "small blind"
    else:
        t = "big blind"
    return t + (" ALL-IN" if a.all_in else "")


class TableView:
    def __init__(self, hero: int | None = 0, show_all: bool = False,
                 delay: float = 0.0, log_lines: int = 9,
                 console: Console | None = None):
        self.hero = hero
        self.view_seat = hero if hero is not None else 0
        self.show_all = show_all
        self.delay = delay
        self.console = console or Console()
        self.log = deque(maxlen=log_lines)
        self.state: HandState | None = None
        self.message = ""

    # Table observer --------------------------------------------------------
    def __call__(self, st: HandState, msg: str, kind: str) -> None:
        self.state = st
        if self.show_all and kind == "action" and st.actions:
            a = st.actions[-1]
            name = st.names[a.seat]
            if msg.startswith(name) and st.hole_cards[a.seat]:
                from .cards import cards_str
                msg = f"{name} [{cards_str(st.hole_cards[a.seat])}]" + msg[len(name):]
        if msg.strip():
            for line in msg.strip().splitlines():
                self.log.append((line, kind))
        if kind == "hand_end":
            return
        if kind in ("action", "street", "showdown", "deal", "win"):
            self.render()

    # Drawing ----------------------------------------------------------------
    def rel(self, seat: int) -> int:
        return (seat - self.view_seat) % self.state.num_seats

    def visible(self, st: HandState, seat: int) -> bool:
        return (self.show_all or seat == self.hero or st.shown[seat])

    def draw_table(self, st: HandState) -> Text:
        c = Canvas()
        for y in range(H):
            for x in range(W):
                d = ((x - CX) / RX) ** 2 + ((y - CY) / RY) ** 2
                if d <= 0.80:
                    c.st[y][x] = FELT
                elif d <= 1.0 and abs(x - CX) > 1:
                    c.st[y][x] = RAIL
        # Centre: hand info, board, pot.
        lvl = (f"Hand {st.hand_id}  ·  Level {st.level}  ·  "
               f"Blinds {fmt(st.small_blind)}/{fmt(st.big_blind)}"
               + (f"  ante {fmt(st.ante)}" if st.ante else ""))
        c.put_center(CX, CY - 3, lvl, f"bold #e8e8e8 {FELT}")
        segs = []
        for i in range(5):
            card = st.board[i] if i < len(st.board) else None
            segs += card_segs(card) + [("  ", FELT)]
        c.put_segments(CX + 1, CY - 1, segs[:-1])
        street = STREETS[st.street].capitalize() if not st.finished else "Hand over"
        c.put_center(CX, CY + 1, street, f"italic #bfe6c8 {FELT}")
        c.put_center(CX, CY + 2, f"Pot {fmt(st.pot)}", f"bold yellow {FELT}")
        if self.show_all and self.hero is not None:
            c.put_center(CX, CY + 4, "DIAGNOSTIC MODE: all cards visible",
                         f"bold #ff9d5c {FELT}")

        for seat in range(st.num_seats):
            self.draw_seat(c, st, seat)
        return c.to_text()

    def draw_seat(self, c: Canvas, st: HandState, seat: int):
        r = self.rel(seat)
        x, y = SEAT_BOX[r]
        busted = not st.dealt_in[seat]
        to_act = getattr(st, "to_act", -1) == seat
        won = st.finished and st.net[seat] > 0
        box = "on #3a3a3a"
        if to_act:
            box = "on #8a5a00"
        elif won:
            box = "on #1f6f3f"
        elif busted or st.folded[seat]:
            box = "on #262626"
        for dy in range(4):
            c.fill(x, y + dy, BOX_W, box)
        name = st.names[seat]
        if seat == self.hero and name.lower() != "you":
            name += " (you)"
        tags = []
        if seat == st.button:
            tags.append("D")
        if seat == st.sb_seat:
            tags.append("SB")
        if seat == st.bb_seat:
            tags.append("BB")
        dim = busted or st.folded[seat]
        name_style = "bold white" if not dim else "#8a8a8a"
        c.put(x + 1, y, name[:BOX_W - 2], f"{name_style} {box}")
        if tags:
            tag = " ".join(tags)
            c.put(x + BOX_W - len(tag) - 1, y, tag, f"bold black on #e0c050")
        if busted:
            c.put(x + 1, y + 1, "out", f"#8a8a8a {box}")
            return
        c.put(x + 1, y + 1, fmt(st.stacks[seat]), f"bold #ffd866 {box}")
        if st.finished and st.net[seat]:
            sign = "+" if st.net[seat] > 0 else ""
            txt = f"{sign}{fmt(st.net[seat])}"
            c.put(x + BOX_W - len(txt) - 1, y + 1, txt,
                  f"bold {'#7CFC00' if st.net[seat] > 0 else '#ff7b7b'} {box}")
        # Cards.
        hole = st.hole_cards[seat]
        if st.folded[seat]:
            c.put(x + 1, y + 2, "folded", f"italic #8a8a8a {box}")
        elif st.mucked[seat] and not self.visible(st, seat):
            c.put(x + 1, y + 2, "mucked", f"italic #8a8a8a {box}")
        elif hole:
            show = self.visible(st, seat)
            xx = x + 1
            for card in hole:
                for t, s in card_segs(card, hidden=not show):
                    c.put(xx, y + 2, t, s)
                    xx += len(t)
                xx += 1
        act = last_action_text(st, seat)
        if to_act and seat != self.hero:
            act = "thinking..."
        if act:
            c.put(x + 1, y + 3, act[:BOX_W - 2],
                  f"{'bold #ff9d5c' if 'ALL-IN' in act else '#d0d0d0'} {box}")
        # Chips in front of the seat.
        bet = st.committed_street[seat]
        if bet and not st.finished:
            bx, by = BET_POS[r]
            c.put_segments(bx, by, [("● ", f"bold #ff5555 {FELT}"),
                                    (fmt(bet), f"bold white {FELT}")])

    def render(self, legal: LegalActions | None = None, prompt_help: bool = False):
        st = self.state
        if st is None:
            return
        parts = [self.draw_table(st)]
        lines = Text()
        for i, (line, kind) in enumerate(self.log):
            style = {"street": "bold cyan", "win": "bold green",
                     "showdown": "magenta", "deal": "dim"}.get(kind, "")
            if line.startswith("==="):
                style = "bold"
            lines.append(_grammar(line.strip("=\n ").strip()) + ("\n" if i < len(self.log) - 1 else ""),
                         style=style)
        parts.append(Panel(lines, title="Action", title_align="left",
                           border_style="#555555", width=W))
        if legal is not None and self.hero is not None:
            parts.append(self.options_text(st, legal))
        if self.message:
            parts.append(Text(self.message, style="bold red"))
        self.console.clear()
        self.console.print(Group(*parts))

    def options_text(self, st: HandState, legal: LegalActions) -> Text:
        t = Text()
        t.append("Your turn.  ", style="bold #ffb000")
        if legal.to_call:
            t.append(f"To call: {fmt(legal.to_call)}   ", style="bold")
            pot_odds = legal.to_call / (st.pot + legal.to_call)
            t.append(f"(pot odds {pot_odds:.0%})   ", style="dim")
        if legal.can_raise:
            word = "raise" if st.current_bet else "bet"
            if legal.min_raise_to >= legal.max_raise_to:
                t.append(f"{word}: all-in {fmt(legal.max_raise_to)}", style="bold")
            else:
                t.append(f"{word} to {fmt(legal.min_raise_to)}"
                         f"–{fmt(legal.max_raise_to)}", style="bold")
        hole = st.hole_cards[self.hero]
        if len(st.board) >= 3 and hole:
            name = hand_name(best_hand(hole + st.board))
            t.append(f"\nYou have: {name.lower()}", style="bold #9ad")
        t.append("\nType: ", style="dim")
        opts = ["fold", "check" if legal.to_call == 0 else "call"]
        if legal.can_raise:
            opts += ["raise <amount>", "raise 1/3 | 1/2 | 2/3 | pot | 2x pot",
                     "raise 3x (the bet)", "all in"]
        t.append(" | ".join(opts), style="cyan")
        t.append("   (help, quit)", style="dim")
        return t


HELP = ("Commands: fold (f), check (k), call (c), raise <amount> (r 300), "
        "bet <amount>, raise 1/4 | 1/3 | half | 2/3 | pot | 2x pot | 2.5x pot "
        "(sizes relative to the pot), raise 3x (3 times the current bet), "
        "all in (a), quit (q). Amounts are the total you put in this street.")


class RichHuman(Agent):
    is_human = True

    def __init__(self, view: TableView, name: str = "You", pause_after_hand=True,
                 auto_muck: bool = False):
        self.view = view
        self.name = name
        self.pause_after_hand = pause_after_hand
        self.auto_muck = auto_muck

    def show_or_muck(self, state: HandState, seat: int) -> bool:
        if self.auto_muck:
            return False
        self.view.state = state
        self.view.render()
        while True:
            try:
                raw = self.view.console.input(
                    "[bold #ffb000]Your hand can't win. Show or muck?[/] "
                    "[dim](s / m)[/dim] ").strip().lower()
            except EOFError:
                return False
            if raw in ("s", "show"):
                return True
            if raw in ("m", "muck", ""):
                return False

    def act(self, obs: Observation) -> Decision:
        legal = obs.legal
        self.view.state = obs.state
        while True:
            self.view.render(legal)
            self.view.message = ""
            try:
                raw = self.view.console.input("[bold]> [/bold]").strip().lower()
            except EOFError:
                raise SystemExit(0)
            d, err = parse_command(raw, obs.state, legal)
            if d is not None:
                return d
            self.view.message = err

    def hand_finished(self, state: HandState, seat: int) -> None:
        self.view.state = state
        self.view.render()
        if self.pause_after_hand:
            try:
                raw = self.view.console.input(
                    "[dim]Press Enter for the next hand (q to quit) [/dim]")
            except EOFError:
                raise SystemExit(0)
            if raw.strip().lower() in ("q", "quit", "exit"):
                raise SystemExit(0)


def parse_command(raw: str, st: HandState, legal: LegalActions):
    if not raw:
        return None, ""
    if raw in ("q", "quit", "exit"):
        raise SystemExit(0)
    if raw in ("h", "help", "?"):
        return None, HELP
    words = raw.replace(",", "").split()
    cmd, rest = words[0], words[1:]
    if cmd in ("f", "fold"):
        if legal.to_call == 0:
            return Decision.check(), ""
        return Decision.fold(), ""
    if cmd in ("k", "x", "check"):
        if legal.to_call:
            return None, f"You can't check; it costs {fmt(legal.to_call)} to call."
        return Decision.check(), ""
    if cmd in ("c", "call"):
        return (Decision.call() if legal.to_call else Decision.check()), ""
    if cmd in ("a", "allin", "all-in", "shove", "jam") or raw in ("all in", "all-in"):
        if legal.can_raise:
            return Decision.raise_to(legal.max_raise_to), ""
        return (Decision.call() if legal.to_call else Decision.check()), ""
    if cmd in ("r", "raise", "b", "bet"):
        if not legal.can_raise:
            return None, "Raising is not allowed here."
        if not rest:
            return None, "Say how much, e.g. 'raise 300', 'raise pot' or 'raise 3x'."
        arg = rest[-1] if rest[0] == "to" and len(rest) > 1 else rest[0]
        my_bet = st.committed_street[getattr(st, "to_act", 0)] \
            if getattr(st, "to_act", -1) >= 0 else 0
        tail = " ".join(rest[1:] if rest[0] == "to" else rest)
        frac = _pot_fraction(tail)
        if frac is not None:
            amt = int(st.current_bet + frac * (st.pot + legal.to_call))
        elif arg in ("all", "allin", "max"):
            amt = legal.max_raise_to
        elif arg.endswith("x") and _is_num(arg[:-1]):
            amt = int(float(arg[:-1]) * max(st.current_bet, st.big_blind))
        elif _is_num(arg):
            amt = int(float(arg))
        else:
            return None, f"I don't understand '{raw}'. Type help."
        if amt >= legal.max_raise_to:
            return Decision.raise_to(legal.max_raise_to), ""
        if amt < legal.min_raise_to:
            if frac is not None:
                amt = legal.min_raise_to
            else:
                return None, (f"The minimum raise is to {fmt(legal.min_raise_to)} "
                              "(amounts are your total for this street).")
        if amt >= legal.max_raise_to:
            return Decision.raise_to(legal.max_raise_to), ""
        return Decision.raise_to(amt), ""
    return None, f"I don't understand '{raw}'. Type help."


def _pot_fraction(text: str):
    """'pot' -> 1, 'half' -> 0.5, '1/3' or '1/3 pot' -> 0.333, '2x pot' -> 2.
    A bare multiple such as '3x' is not a pot fraction (it means 3 times the bet)."""
    t = text.replace("pot", " pot").split()
    if not t:
        return None
    if t == ["pot"]:
        return 1.0
    if t[0] == "half":
        return 0.5
    if "/" in t[0]:
        a, _, b = t[0].partition("/")
        if _is_num(a) and _is_num(b) and float(b):
            return float(a) / float(b)
    if len(t) >= 2 and t[1] == "pot":
        x = t[0].rstrip("x")
        if _is_num(x):
            return float(x)
    return None


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


class WatchPacer:
    """Pause between hands in watch mode."""

    def __init__(self, view: TableView, pause: float):
        self.view, self.pause = view, pause

    def __call__(self, st, msg, kind):
        self.view(st, msg, kind)
        if kind == "hand_end":
            self.view.render()
            time.sleep(self.pause)
