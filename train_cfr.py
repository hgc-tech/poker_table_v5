"""Train the v5 MCCFR blueprints on this machine.

    python train_cfr.py --memory              # RAM per blueprint and the bucket level each one uses
    python train_cfr.py --abstraction         # build the card abstraction only
    python train_cfr.py --hours 24            # train every blueprint, round robin, for 24 hours
    python train_cfr.py                       # ... until you press Ctrl-C
    python train_cfr.py --only 6x20 --minutes 90
    python train_cfr.py --only 3x10-30-30          # one unequal-stack blueprint (profiles@3 in the config)
    python train_cfr.py --status              # iterations, hours and memory so far
    python train_cfr.py --lbr                 # exploitability estimate of the heads-up blueprints

Training resumes from the checkpoints in checkpoint_dir, so you can stop with
Ctrl-C at any time (the current blueprint is saved first) and run the same
command later to continue. After each slice the strategy is exported to
blueprint_dir, where play.py finds it.

Settings are in cfr_config.txt ("key@6 = ..." overrides a setting for
6-player blueprints).
"""
from __future__ import annotations

import argparse
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

from pokerbot.mccfr.config import DEFAULT_CONFIG, CfrConfig
from pokerbot.mccfr.native import EXE, build


def checkpoint_info(path: Path) -> tuple[int, float, int, float]:
    """(iterations, seconds, bucket level, GiB of regrets) stored in a checkpoint, or zeros."""
    try:
        with open(path, "rb") as f:
            magic, n = struct.unpack("<II", f.read(8))
            txt = f.read(n).decode(errors="replace")
            it, secs, _chunks, used = struct.unpack("<QdQQ", f.read(32))
            level = 0
            for line in txt.splitlines():
                if line.startswith("level="):
                    level = int(line.split("=", 1)[1])
            return it, secs, level, used / 2 ** 30
    except (OSError, struct.error, ValueError):
        return 0, 0.0, -1, 0.0


class Job:
    def __init__(self, cfg: CfrConfig, players: int, stack: str):
        self.cfg, self.players, self.stack = cfg, players, stack     # "20" or "10-30-30" (by position)
        self.max_stack = max(float(x) for x in stack.split("-"))
        self.name = cfg.blueprint_name(players, stack)
        self.ckpt = cfg.checkpoint_dir / f"{self.name}.ckpt"
        self.strat = cfg.blueprint_dir / f"{self.name}.strat"
        self.weight = 1.0

    def label(self):
        return f"{self.players} players, {self.stack} bb"


def jobs_for(cfg: CfrConfig, only: list[str] | None) -> list[Job]:
    jobs = []
    skip = set(cfg.kv.get("skip", []))
    for n in cfg.players:
        for s in [f"{x:g}" for x in cfg.stacks_for(n)] + cfg.profiles_for(n):
            if f"{n}x{s}" in skip or (only and f"{n}x{s}" not in only):
                continue
            jobs.append(Job(cfg, n, s))
    return jobs


def run(cmd: list[str]) -> int:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen([str(c) for c in cmd])
    try:
        return p.wait()
    except KeyboardInterrupt:
        # The trainer received the same Ctrl-C; let it save its checkpoint.
        return p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--memory", action="store_true", help="print memory estimates and exit")
    ap.add_argument("--abstraction", action="store_true", help="build the card abstraction and exit")
    ap.add_argument("--status", action="store_true", help="show training progress and exit")
    ap.add_argument("--hours", type=float, default=0, help="total training time (0 = until Ctrl-C)")
    ap.add_argument("--minutes", type=float, default=0, help="total training time in minutes")
    ap.add_argument("--slice", type=float, default=30,
                    help="minutes per blueprint per round for the largest blueprint (smaller ones get less)")
    ap.add_argument("--rounds", type=int, default=0, help="stop after this many rounds (0 = no limit)")
    ap.add_argument("--only", nargs="*", help="blueprints to train, e.g. 6x20 2x10")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--lbr", action="store_true",
                    help="estimate the exploitability of the heads-up blueprints (local best response) and exit")
    ap.add_argument("--lbr-hands", type=int, default=200000)
    args = ap.parse_args()

    build()
    cfg = CfrConfig(args.config)
    common = ["--config", cfg.path]
    if args.threads:
        common += ["--threads", str(args.threads)]
    if args.memory:
        run([EXE, "memory", "--config", cfg.path, "--players", ",".join(str(p) for p in cfg.players)])
        return
    if not (cfg.abstraction_dir / "meta.txt").exists() or args.abstraction:
        rc = run([EXE, "abstraction", "--out", cfg.abstraction_dir] + common)
        if rc != 0 or args.abstraction:
            return
    jobs = jobs_for(cfg, args.only)
    if not jobs:
        sys.exit("No blueprints selected.")
    if args.status:
        print(f"{'Blueprint':<30}{'Iterations':>16}{'Hours':>9}{'Level':>7}{'GiB':>8}  Exported")
        for j in jobs:
            it, secs, lv, gib = checkpoint_info(j.ckpt)
            lvs = f"L{lv}" if lv >= 0 else "-"
            print(f"{j.label():<30}{it:>16,}{secs / 3600:>9.2f}{lvs:>7}{gib:>8.2f}  "
                  f"{'yes' if j.strat.exists() else 'no'}")
        return
    if args.lbr:
        for j in jobs:
            if j.players != 2 or not j.ckpt.exists():
                continue
            run([EXE, "lbr", "--in", j.ckpt, "--hands", str(args.lbr_hands), "--config", cfg.path,
                 "--abstraction", cfg.abstraction_dir] + common[2:])
        return
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.blueprint_dir.mkdir(parents=True, exist_ok=True)
    # Time slices grow with the square root of the tree size.
    sizes = {}
    for j in jobs_for(cfg, None):     # weights relative to every blueprint, even with --only
        sizes[j.name] = max(1.0, float(j.players) ** 3 * j.max_stack)
    big = max(sizes.values())
    for j in jobs:
        j.weight = max(0.1, (sizes[j.name] / big) ** 0.5)
    budget = args.hours * 3600 + args.minutes * 60
    start = time.time()
    stop = False
    rnd = 0
    while not stop:
        rnd += 1
        failed = 0
        for j in jobs:
            left = budget - (time.time() - start) if budget else None
            if left is not None and left <= 1:
                stop = True
                break
            minutes = args.slice * j.weight
            if left is not None:
                minutes = min(minutes, left / 60)
            print(f"\n== Round {rnd}: {j.label()} for {minutes:.1f} min ==", flush=True)
            rc = run([EXE, "train", "--players", j.players, "--stack", j.stack,
                      "--out", j.ckpt, "--minutes", f"{minutes:.3f}",
                      "--export", j.strat, "--config", j.cfg.path, "--abstraction", j.cfg.abstraction_dir]
                     + (["--threads", str(args.threads)] if args.threads else []))
            if rc == 2:
                print(f"Skipping {j.label()} (see the error above).")
                failed += 1
                continue
            if rc != 0:
                print("Training stopped." if rc == 130 else f"Training stopped (exit code {rc}).")
                stop = True
                break
        if failed == len(jobs):
            print("Every blueprint failed; stopping.")
            stop = True
        if (budget and time.time() - start >= budget) or (args.rounds and rnd >= args.rounds):
            stop = True
    print("\nDone. Run 'python train_cfr.py --status' to see progress, or 'python play.py' to play.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
