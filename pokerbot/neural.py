"""Evolvable neural bots.

A genome is a flat torch vector: the weights of a small policy network plus a
few "genes" that control the opponent model and play style:

    prior_n    how many hands of evidence the solid-player prior is worth
    decay      per-hand recency weight on observed counts (adapts to changes)
    width_mult scales every modeled opponent range
    temp       softmax temperature (mixing between actions)

NeuralBot(genome) is a decision function f(history, current) usable with
pokerbot.Bot.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .features import (A_2X, A_ALLIN, A_BIG, A_CALL, A_FOLD, A_HALF, A_POT,
                       A_SMALL, A_THIRD, A_TWO_THIRDS, NUM_ACTIONS, RAISE_ACTIONS,
                       NUM_FEATURES, action_to_decision, features, legal_mask)
from .opponent import OpponentModel

HIDDEN = 32
GENES = ["prior_n", "decay", "width_mult", "temp"]


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(NUM_FEATURES, HIDDEN)
        self.l2 = nn.Linear(HIDDEN, HIDDEN)
        self.out = nn.Linear(HIDDEN, NUM_ACTIONS)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        return self.out(x)


def num_params() -> int:
    return sum(p.numel() for p in PolicyNet().parameters())


def gene_values(raw: torch.Tensor) -> dict[str, float]:
    """Map unconstrained gene parameters to their working ranges."""
    r = raw.tolist()
    return {
        "prior_n": float(math.exp(np.clip(r[0], -2.0, 4.0)) * 6.0),   # ~0.8..330
        "decay": float(1.0 - 0.10 / (1.0 + math.exp(-r[1]))),         # 0.90..1
        "width_mult": float(math.exp(np.clip(r[2], -2.0, 2.0))),
        "temp": float(0.15 + 0.85 / (1.0 + math.exp(-r[3]))),        # 0.15..1
    }


def default_genes() -> torch.Tensor:
    # prior_n 6, decay ~0.975, width_mult 1, temp ~0.4
    return torch.tensor([0.0, -1.1, 0.0, -1.0])


def new_genome(net: PolicyNet | None = None) -> torch.Tensor:
    net = net or PolicyNet()
    flat = torch.nn.utils.parameters_to_vector(net.parameters()).detach()
    return torch.cat([flat, default_genes()])


def genome_to_net(genome: torch.Tensor) -> PolicyNet:
    net = PolicyNet()
    torch.nn.utils.vector_to_parameters(genome[:num_params()].clone(),
                                        net.parameters())
    return net


class NeuralBot:
    """Decision function backed by a genome.

    The forward pass runs in NumPy for speed during mass simulation; it is
    the same computation as PolicyNet.forward.
    """

    def __init__(self, genome: torch.Tensor, greedy: bool = False,
                 seed: int | None = None):
        self.genome = genome.detach().clone()
        self.genes = gene_values(self.genome[num_params():])
        net = genome_to_net(self.genome)
        self.W = [(l.weight.detach().numpy().copy(), l.bias.detach().numpy().copy())
                  for l in (net.l1, net.l2, net.out)]
        self.model = OpponentModel(self.genes["prior_n"], self.genes["decay"])
        self.greedy = greedy
        self.rng = np.random.default_rng(seed)
        self.record = None   # set to a list to log (features, action)

    def logits(self, f: np.ndarray) -> np.ndarray:
        (w1, b1), (w2, b2), (w3, b3) = self.W
        h = np.tanh(w1 @ f + b1)
        h = np.tanh(w2 @ h + b2)
        return w3 @ h + b3

    def __call__(self, history, current):
        self.model.update(history)
        f, sit = features(current.numpy(), self.model, self.genes["width_mult"])
        z = self.logits(f)
        mask = legal_mask(sit)
        z = np.where(mask, z, -1e9)
        if self.greedy:
            a = int(np.argmax(z))
        else:
            z = z / self.genes["temp"]
            p = np.exp(z - z.max())
            p /= p.sum()
            a = int(self.rng.choice(len(p), p=p))
        if self.record is not None:
            self.record.append((f, mask, a))
        return action_to_decision(a, sit)


# Teacher: a hand-written solid strategy used to pretrain the network ---------
class Teacher:
    """Equity-vs-modeled-range strategy. Plays a tight, position-aware,
    push/fold-aware game, and adapts through the opponent model."""

    OPEN_WIDTH = {5: 0.14, 4: 0.17, 3: 0.21, 2: 0.27, 1: 0.40, 0: 0.0}

    def __init__(self, seed=None, record=None):
        self.model = OpponentModel(6.0, 0.985)
        self.rng = np.random.default_rng(seed)
        self.record = record

    def choose(self, f, sit) -> int:
        eq, raw, pot_odds = f[26], f[2], f[3]
        fold_eq, pct = f[14], 1.0 - f[22]
        behind = int(round(f[12] * 5))
        stack = sit.stack
        r = self.rng.random()
        if sit.street == 0:
            unopened = f[25] > 0.5
            if unopened:
                width = self.OPEN_WIDTH.get(behind, 0.14)
                width = min(1.0, width * (0.6 + 1.2 * fold_eq))
                if stack <= 12:
                    width = min(1.0, width * (14.0 / max(stack, 1.0)) ** 0.8)
                    return A_ALLIN if pct <= width else (
                        A_FOLD if sit.to_call > 0 else A_CALL)
                if pct <= width:
                    return A_SMALL
                if sit.to_call > 0 and sit.to_call <= 0.5 and pct <= 2 * width:
                    return A_CALL
                return A_FOLD if sit.to_call > 0 else A_CALL
            # Facing a raise.
            if eq > 0.60 and sit.can_raise:
                return A_ALLIN if stack < 25 or f[16] > 0.5 else A_BIG
            if eq > pot_odds + 0.04:
                return A_CALL
            return A_FOLD
        # Postflop. Shown-card stats: bluff = how often the bettor's shown
        # bets were bluffs; callers = how strong opponents' shown calls were.
        bluff, callers = f[27], f[29]
        if sit.to_call <= 0:
            value_line = 0.52 - 0.5 * (0.62 - callers)   # thinner vs stations
            if eq > 0.85:
                return A_2X if callers < 0.55 else A_POT
            if eq > max(value_line + 0.16, 0.6):
                return A_TWO_THIRDS
            if eq > value_line:
                return A_HALF
            if fold_eq > 0.55 and r < 0.35 and callers > 0.55:
                return A_THIRD
            return A_CALL
        if eq > 0.78 and sit.can_raise:
            return A_POT
        # Call lighter against players shown to bluff, tighter against
        # players whose shown bets are always strong.
        margin = 0.03 - 0.6 * (bluff - 0.30)
        if eq > pot_odds + margin:
            return A_CALL
        return A_FOLD

    def __call__(self, history, current):
        self.model.update(history)
        f, sit = features(current.numpy(), self.model, 1.0)
        a = self.choose(f, sit)
        # Vary raise sizes a little so pretraining sees every size.
        if a in RAISE_ACTIONS and self.rng.random() < 0.25:
            a = int(np.clip(a + self.rng.choice([-1, 1]), RAISE_ACTIONS[0],
                            RAISE_ACTIONS[-1]))
        mask = legal_mask(sit)
        if not mask[a]:
            a = A_CALL
        if self.record is not None:
            self.record.append((f, mask, a))
        return action_to_decision(a, sit)
