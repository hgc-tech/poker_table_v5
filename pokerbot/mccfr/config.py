"""Read cfr_config.txt ("key = value value ..." lines, '#' comments).

The same file drives the C++ trainer (native/main.cpp) and the Python side.
"key@6 = ..." overrides "key = ..." for 6-player blueprints. Relative paths
are relative to the config file's folder.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "cfr_config.txt"
STARTER_CONFIG = ROOT / "starter" / "cfr_config.txt"   # optional download: briefly trained blueprints

# Search size menus sent to the engine with every decision (pot fractions).
SEARCH_KEYS = ["search_bet", "search_raise", "search_inner_bet", "search_inner_raise",
               "search_preflop_open", "search_preflop_raise"]


def parse(path: Path) -> dict[str, list[str]]:
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.replace(",", " ").split()
    return out


class CfrConfig:
    def __init__(self, path: str | Path = DEFAULT_CONFIG):
        self.path = Path(path).resolve()
        self.kv = parse(self.path)
        base = self.path.parent
        self.abstraction_dir = base / self.get("abstraction_dir", "abstraction")
        self.blueprint_dir = base / self.get("blueprint_dir", "blueprints")
        self.checkpoint_dir = base / self.get("checkpoint_dir", "checkpoints")
        self.payouts = tuple(float(x) for x in self.kv.get("payouts", ["0.65", "0.35"]))
        self.players = [int(float(x)) for x in self.kv.get("players", ["2", "3", "4", "5", "6"])]
        self.stacks = [float(x) for x in self.kv.get("stacks_bb", ["10", "15", "20", "25", "30", "40"])]
        self._engine = None

    def get(self, key: str, default: str) -> str:
        v = self.kv.get(key)
        return v[0] if v else default

    def num(self, key: str, default: float) -> float:
        v = self.kv.get(key)
        return float(v[0]) if v else default

    def stacks_for(self, players: int) -> list[float]:
        v = self.kv.get(f"stacks_bb@{players}")
        return [float(x) for x in v] if v else self.stacks

    def profiles_for(self, players: int) -> list[str]:
        """Unequal-stack blueprints from "profiles@N = 10-30-30 ..." (stacks by position, button first)."""
        out = []
        for p in self.kv.get(f"profiles@{players}", []):
            parts = [float(x) for x in p.replace("/", "-").split("-")]
            if len(parts) != players or min(parts) <= 0:
                raise ValueError(f"profiles@{players}: '{p}' needs {players} positive stacks separated by '-'")
            out.append("-".join(f"{x:g}" for x in parts))
        return out

    def search_menus(self) -> dict[str, list[float]]:
        return {k: [float(x) for x in self.kv[k] if x not in ("none", "-")] for k in SEARCH_KEYS if k in self.kv}

    def strategies(self) -> list[Path]:
        return sorted(self.blueprint_dir.glob("*.strat"))

    def engine(self):
        if self._engine is None:
            from .native import Engine
            if not self.strategies():
                raise FileNotFoundError(
                    f"No blueprints in {self.blueprint_dir}. Train them with "
                    f"'python train_cfr.py --config {self.path.name}'.")
            self._engine = Engine(self.abstraction_dir, self.blueprint_dir)
        return self._engine

    def blueprint_name(self, players: int, stack: float | str) -> str:
        return f"p{players}_s{stack if isinstance(stack, str) else f'{stack:g}'}"


def default_config() -> CfrConfig:
    """Your own trained blueprints if there are any, otherwise the starter set if installed."""
    cfg = CfrConfig(DEFAULT_CONFIG)
    if not cfg.strategies() and STARTER_CONFIG.exists():
        return CfrConfig(STARTER_CONFIG)
    return cfg
