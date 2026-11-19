"""Save a snapshot of the current blueprints, to measure later training against.

    python snapshot.py day1          # snapshots/day1/ holds today's blueprints
    python compare.py --opponent snapshot --against snapshots/day1

The snapshot uses hard links, so it takes no extra disk space until training
exports newer blueprints (each export writes a new file; the snapshot keeps
the old one). Delete a snapshot folder when you no longer need it.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pokerbot.mccfr.config import CfrConfig, default_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", default=time.strftime("%Y%m%d_%H%M"))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = CfrConfig(args.config) if args.config else default_config()
    strats = cfg.strategies()
    if not strats:
        sys.exit(f"No blueprints in {cfg.blueprint_dir}.")
    out = cfg.path.parent / "snapshots" / args.name
    if out.exists():
        sys.exit(f"{out} already exists; pick another name.")
    out.mkdir(parents=True)
    for p in strats:
        try:
            os.link(p, out / p.name)
        except OSError:
            shutil.copy2(p, out / p.name)
    print(f"Saved {len(strats)} blueprints to {out}")


if __name__ == "__main__":
    main()
