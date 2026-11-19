"""Adapter for the equity-based player from neuron_poker.

Source: https://github.com/dickreuter/neuron_poker (MIT License, copyright
2019 Nicolas Dickreuter; see LICENSE_neuron_poker.txt). The `Action` enum and
the `Player` class below are copied from gym_env/enums.py and
agents/agent_consider_equity.py without changes to their decision logic.

The adapter supplies what neuron_poker's environment would supply:
* info["player_data"]["equity_to_river_alive"]: Monte Carlo probability of
  winning against random hands of every player still in the hand (neuron_poker
  computes this with its own simulator; here eval7 does the same simulation);
* the action space, using neuron_poker's legality rules (at most two raises per
  player per street; raise sizes of half pot, pot, 2x pot and all-in);
* the raise amounts, using neuron_poker's definition (a half-pot raise adds
  half of the pot to the player's chips in front).
"""
from __future__ import annotations

from enum import Enum

import numpy as np

import eval7

from ..encoding import context_fields
from ..handstrength import E7
from ..opponent import BET, RAISE, parse_actions
from ..state import Decision


# ---- copied from neuron_poker gym_env/enums.py ---------------------------
class Action(Enum):
    """Allowed actions"""

    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE_3BB = 3
    RAISE_HALF_POT = 4
    RAISE_POT = 5
    RAISE_2POT = 6
    ALL_IN = 7
    SMALL_BLIND = 8
    BIG_BLIND = 9


# ---- copied from neuron_poker agents/agent_consider_equity.py -------------
class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name="Random", min_call_equity=None, min_bet_equity=None):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.name = name

        self.min_call_equity = min_call_equity
        self.min_bet_equity = min_bet_equity

        self.autoplay = True

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation
        equity_alive = info["player_data"]["equity_to_river_alive"]

        incremen1 = 0.1
        increment2 = 0.2

        if (
            equity_alive > self.min_bet_equity + increment2
            and Action.ALL_IN in action_space
        ):
            action = Action.ALL_IN

        elif (
            equity_alive > self.min_bet_equity + incremen1
            and Action.RAISE_2POT in action_space
        ):
            action = Action.RAISE_2POT

        elif equity_alive > self.min_bet_equity and Action.RAISE_POT in action_space:
            action = Action.RAISE_POT

        elif (
            equity_alive > self.min_bet_equity - incremen1
            and Action.RAISE_HALF_POT in action_space
        ):
            action = Action.RAISE_HALF_POT

        elif equity_alive > self.min_call_equity and Action.CALL in action_space:
            action = Action.CALL

        elif Action.CHECK in action_space:
            action = Action.CHECK

        else:
            action = Action.FOLD

        return action


# ---- adapter --------------------------------------------------------------
def equity_vs_random(hole, board, n_opponents, runs=600, rng=None):
    """Probability of winning (ties count half) against n random hands."""
    rng = rng or np.random.default_rng()
    if n_opponents <= 0:
        return 1.0
    dead = set(hole) | set(board)
    deck = np.array([c for c in range(52) if c not in dead])
    need = 5 - len(board)
    k = need + 2 * n_opponents
    wins = 0.0
    me_cards = [E7[c] for c in hole]
    for _ in range(runs):
        draw = rng.choice(deck, size=k, replace=False)
        full = [E7[c] for c in board] + [E7[c] for c in draw[:need]]
        mine = eval7.evaluate(me_cards + full)
        best_opp = max(eval7.evaluate([E7[draw[need + 2 * i]], E7[draw[need + 2 * i + 1]]] + full)
                       for i in range(n_opponents))
        wins += 1.0 if mine > best_opp else (0.5 if mine == best_opp else 0.0)
    return wins / runs


class NeuronPokerEquityBot:
    """Decision function f(history, current) wrapping neuron_poker's player."""

    def __init__(self, min_call_equity=0.5, min_bet_equity=0.6, seed=None):
        self.player = Player(f"Equity/{min_call_equity}/{min_bet_equity}",
                             min_call_equity, min_bet_equity)
        self.rng = np.random.default_rng(seed)

    def __call__(self, history, current):
        f = context_fields(current)
        bb = f["big_blind_chips"].item() * 1000
        hole = [int(c) for c in np.nonzero(f["hole_cards"][:52].numpy())[0]]
        board = [int(c) for c in np.nonzero(f["board"].numpy())[0]]
        in_hand = f["in_hand"].numpy() > 0.5
        street = int(np.argmax(f["street"].numpy()))
        to_call = f["to_call_bb"].item() * bb
        stack = f["stacks_bb"][0].item() * bb
        my_bet = f["committed_street_bb"][0].item() * bb
        pot = f["pot_bb"].item() * bb
        can_raise = f["can_raise"].item() > 0.5
        max_to = f["max_raise_to_bb"].item() * bb

        my_raises = sum(1 for a in parse_actions(current.numpy())
                        if a[0] == street and a[1] == 0 and a[2] in (BET, RAISE))
        space = [Action.CHECK] if to_call == 0 else [Action.CALL, Action.FOLD]
        if can_raise and my_raises < 2:
            if stack >= 3 * bb - my_bet:
                space.append(Action.RAISE_3BB)
            for act, mult in ((Action.RAISE_HALF_POT, 0.5), (Action.RAISE_POT, 1.0),
                              (Action.RAISE_2POT, 2.0)):
                if stack >= pot * mult >= to_call:
                    space.append(act)
            if stack > 0:
                space.append(Action.ALL_IN)

        equity = equity_vs_random(hole, board, int(in_hand.sum()) - 1, rng=self.rng)
        act = self.player.action(space, None,
                                 {"player_data": {"equity_to_river_alive": equity}})
        if act == Action.FOLD:
            return Decision.fold()
        if act == Action.CHECK:
            return Decision.check()
        if act == Action.CALL:
            return Decision.call()
        if act == Action.ALL_IN:
            return Decision.raise_to(int(round(max_to)))
        mult = {Action.RAISE_HALF_POT: 0.5, Action.RAISE_POT: 1.0,
                Action.RAISE_2POT: 2.0}.get(act)
        add = pot * mult if mult is not None else 3 * bb - my_bet
        return Decision.raise_to(int(round(my_bet + add)))
