"""Measure the v5 bots against other bots in full tournaments.

Opponents:
    neural     the v3 neural bots in weights/ (fast)
    v4         the v4 MCCFR bots from a v4 install (--v4 ~/poker_table)
    snapshot   v5 bots playing an earlier snapshot of your own blueprints
               (--against snapshots/<name>; make one with snapshot.py)
    all_in, check_call, jam_fold_wide, ...   the fixed sparring bots

Each match is a lineup: "3v3" (3 v5 bots, 3 opponents), "1v5" or "5v1". Seats
are shuffled every tournament. The score is the average prize share per seat
under the payouts in the config; an average seat earns 1/6 (16.7%). Intervals
are 95% confidence intervals over tournaments. A v5 share above 16.7% with an
interval that excludes 16.7% means v5 is stronger.

    python compare.py --opponent neural --tournaments 300
    python compare.py --opponent v4 --v4 ~/poker_table --tournaments 300
    python compare.py --opponent snapshot --against snapshots/day1 --tournaments 300
    python compare.py --opponent v4 --v4 ~/poker_table --search-ms 200 --tournaments 60   # with search (slower)

Without --search-ms both sides play their blueprints only, which measures the
blueprints and is fast. Writes reports/compare_<opponent>.md (or --out).
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pokerbot import Bot, Table                                  # noqa: E402
from pokerbot.mccfr import CfrConfig, SearchSettings, default_config  # noqa: E402
from pokerbot.mccfr.bot import MCCFRBot, OpponentStats           # noqa: E402
from pokerbot.mccfr.native import Engine, build_v4_library       # noqa: E402
from pokerbot.sparring import ALL_FIXED_BOTS                     # noqa: E402

_W = {}


def v4_config(v4_root: Path) -> CfrConfig:
    own = CfrConfig(v4_root / "cfr_config.txt")
    if own.strategies():
        return own
    return CfrConfig(v4_root / "starter" / "cfr_config.txt")


def _init(cfg_path, opponent, v4_root, against, search_ms):
    cfg = CfrConfig(cfg_path)
    _W["cfg"] = cfg
    _W["engine"] = cfg.engine()
    mode = "search" if search_ms > 0 else "blueprint"
    _W["settings"] = SearchSettings(mode=mode, search_ms=search_ms, threads=1, menus=cfg.search_menus())
    _W["other_settings"] = SearchSettings(mode=mode, search_ms=search_ms, threads=1)
    if opponent == "v4":
        vc = v4_config(Path(v4_root))
        _W["other"] = Engine(vc.abstraction_dir, vc.blueprint_dir, lib=build_v4_library(v4_root))
    elif opponent == "snapshot":
        _W["other"] = Engine(cfg.abstraction_dir, against)
        _W["other_settings"].menus = cfg.search_menus()
    elif opponent == "neural":
        import torch
        torch.set_num_threads(1)
        from pokerbot.trained import load_champions
        _W["genomes"] = load_champions(k=6)


def _play(job):
    kinds, opponent, seed = job
    rng = np.random.default_rng(seed)
    order = rng.permutation(6)
    cfg, eng = _W["cfg"], _W["engine"]
    stats = OpponentStats(eng, 6, cfg.payouts)
    ostats = OpponentStats(_W["other"], 6, cfg.payouts) if "other" in _W else None
    agents = [None] * 6
    for i, (slot, kind) in enumerate(zip(order, kinds)):
        if kind == "v5":
            agents[slot] = MCCFRBot(f"V{i}", eng, _W["settings"], cfg.payouts, seed * 10 + i, stats)
        elif opponent in ("v4", "snapshot"):
            agents[slot] = MCCFRBot(f"O{i}", _W["other"], _W["other_settings"], cfg.payouts, seed * 10 + i, ostats)
        elif opponent == "neural":
            from pokerbot.neural import NeuralBot
            g = _W["genomes"][i % len(_W["genomes"])]
            agents[slot] = Bot(f"N{i}", NeuralBot(g, seed=seed * 10 + i))
        else:
            agents[slot] = Bot(f"F{i}", ALL_FIXED_BOTS[opponent](seed * 10 + i))
    t = Table(agents, seed=seed)
    ranking = t.run()
    pay = list(cfg.payouts) + [0.0] * 6
    share = [0.0] * 6
    for place, s in enumerate(ranking):
        share[s] = pay[place]
    return [(kind, share[slot], 1 + ranking.index(slot)) for slot, kind in zip(order, kinds)]


def match(pool, kinds, opponent, n, seed0):
    res = pool.map(_play, [(kinds, opponent, seed0 + i) for i in range(n)])
    out = {}
    for rows in res:
        per = {}
        for kind, share, place in rows:
            per.setdefault(kind, []).append((share, place))
        for kind, v in per.items():          # one number per tournament per side
            out.setdefault(kind, []).append((np.mean([x[0] for x in v]), np.mean([x[1] for x in v])))
    summary = {}
    for kind, v in out.items():
        a = np.array(v)
        m = a[:, 0].mean()
        ci = 1.96 * a[:, 0].std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0
        summary[kind] = (m, ci, a[:, 1].mean())
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="v5 config (default cfr_config.txt)")
    ap.add_argument("--opponent", default="neural",
                    choices=["neural", "v4", "snapshot", *ALL_FIXED_BOTS])
    ap.add_argument("--v4", default=None, help="v4 poker_table folder (for --opponent v4)")
    ap.add_argument("--against", default=None, help="snapshot blueprint folder (for --opponent snapshot)")
    ap.add_argument("--lineups", nargs="*", default=["3v3", "1v5", "5v1"])
    ap.add_argument("--tournaments", type=int, default=300, help="tournaments per lineup")
    ap.add_argument("--search-ms", type=float, default=0, help="search time per decision (0 = blueprints only)")
    ap.add_argument("--workers", type=int, default=2,
                    help="parallel processes; each maps the blueprints it meets (memory is shared via the OS)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = CfrConfig(args.config) if args.config else default_config()
    if args.opponent == "v4" and not args.v4:
        sys.exit("--opponent v4 needs --v4 PATH (your v4 poker_table folder)")
    if args.opponent == "snapshot" and not args.against:
        sys.exit("--opponent snapshot needs --against FOLDER")
    against = str(Path(args.against).resolve()) if args.against else None
    v4_root = str(Path(args.v4).expanduser().resolve()) if args.v4 else None
    if v4_root:
        build_v4_library(v4_root)
    mode = f"{args.search_ms:g} ms search" if args.search_ms > 0 else "blueprints only"
    name = {"neural": "v3 neural bots", "v4": f"v4 bots ({v4_root})",
            "snapshot": f"snapshot {against}"}.get(args.opponent, f"{args.opponent} bots")
    lines = [f"# v5 vs {name}", "",
             f"{mode}; {args.tournaments} tournaments per lineup, seats shuffled. Score = average prize share per "
             f"seat with payouts {', '.join(f'{p:.0%}' for p in cfg.payouts)}; an average seat earns 16.7%. "
             "Intervals are 95% confidence intervals.", "",
             "| Lineup | Side | Prize share per seat | Mean place |", "|---|---|---|---|"]
    with Pool(args.workers, initializer=_init,
              initargs=(str(cfg.path), args.opponent, v4_root, against, args.search_ms)) as pool:
        for i, lu in enumerate(args.lineups):
            a, b = (int(x) for x in lu.lower().split("v"))
            if a + b != 6:
                sys.exit(f"lineup {lu}: the two numbers must add up to 6")
            kinds = ["v5"] * a + [args.opponent] * b
            t0 = time.time()
            s = match(pool, kinds, args.opponent, args.tournaments, 1000 * (i + 1))
            for kind, (m, ci, place) in sorted(s.items(), key=lambda kv: kv[0] != "v5"):
                side = "v5" if kind == "v5" else args.opponent
                lines.append(f"| {a} v5 vs {b} {args.opponent} | {side} | {m:.1%} ± {ci:.1%} | {place:.2f} |")
                print(f"{a} v5 vs {b} {args.opponent:<10} {side:<10} {m:6.1%} ± {ci:.1%}  place {place:.2f}  "
                      f"({time.time() - t0:.0f} s)", flush=True)
    out = Path(args.out) if args.out else ROOT / "reports" / f"compare_{args.opponent}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
