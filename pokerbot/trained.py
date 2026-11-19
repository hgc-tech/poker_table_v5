"""Load trained bots from weights/population.pt."""
from __future__ import annotations

from pathlib import Path

import torch

from .agents import Bot
from .neural import NeuralBot

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "weights" / "population.pt"


def load_champions(path: Path | str = DEFAULT_PATH, k: int = 5,
                   min_games: int = 40) -> list[torch.Tensor]:
    """The k highest-rated genomes with at least `min_games` tournaments
    played (so the rating is reliable), from an evolved population."""
    data = torch.load(path)
    genomes, games = data["genomes"], data.get("games", [])
    order = [i for i in range(len(genomes)) if games and games[i] >= min_games]
    order += [i for i in range(len(genomes)) if i not in order]
    return [genomes[i] for i in order[:k]]


def trained_bots(path: Path | str = DEFAULT_PATH, k: int = 5,
                 names: list[str] | None = None) -> list[Bot]:
    genomes = load_champions(path, k)
    names = names or [f"Bot{i + 1}" for i in range(k)]
    return [Bot(n, NeuralBot(g)) for n, g in zip(names, genomes)]
