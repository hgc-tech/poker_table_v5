"""Play or watch a tournament in the terminal.

    python play.py                      # you vs five MCCFR bots (blueprint + search), oval table
    python play.py --search-ms 1500     # let the bots search longer per decision
    python play.py --no-search          # blueprint only (fast)
    python play.py --bots neural        # the earlier neuroevolution bots
    python play.py --bots check_call    # you vs five check/call bots
    python play.py --seed 7 --delay 0.5 # half-second pause after each bot action
    python play.py --watch              # six bots play a full tournament
    python play.py --watch --show-cards # ... with every player's cards face up
    python play.py --diagnostic         # practice with every player's cards face up
    python play.py --ui plain           # the original scrolling text log

Make the terminal at least 100 columns wide and 40 rows tall for the oval view.
"""
import argparse

from pokerbot import STARTING_STACK, Bot, HumanCLI, Table
from pokerbot.sparring import ALL_FIXED_BOTS as SPARRING


def make_bots(kind="mccfr", weights=None, k=5, cfr=None, settings=None, seed=None, log=None):
    # Swap in other decision functions here, one per seat.
    if kind == "mccfr":
        from pokerbot.mccfr import mccfr_bots
        return mccfr_bots(cfr, k=k, settings=settings, seed=seed, num_seats=6 if k == 5 else k,
                          log=log)
    if kind == "neural":
        from pokerbot.trained import DEFAULT_PATH, trained_bots
        return trained_bots(weights or DEFAULT_PATH, k=k)
    return [Bot(f"Bot{i}", SPARRING[kind]()) for i in range(1, k + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bots", default="mccfr",
                    choices=["mccfr", "neural", *SPARRING],
                    help="which bots fill the other seats (default: MCCFR blueprint + search)")
    ap.add_argument("--cfr-config", default=None,
                    help="config for the MCCFR bots (default: cfr_config.txt)")
    ap.add_argument("--search-ms", type=float, default=800,
                    help="MCCFR bots: search time per decision in milliseconds")
    ap.add_argument("--threads", type=int, default=2, help="MCCFR bots: search threads")
    ap.add_argument("--depth", type=int, default=6,
                    help="MCCFR bots: actions searched ahead (the search also stops at the end of the street)")
    ap.add_argument("--no-search", action="store_true", help="MCCFR bots: play the blueprint only")
    ap.add_argument("--max-jam-spr", type=float, default=4,
                    help="MCCFR bots: no all-in when the effective stack is more than this many pots "
                         "(preflop: unless 30 bb or less); 0 = no limit")
    ap.add_argument("--no-model", action="store_true",
                    help="MCCFR bots: do not exploit (no opponent model)")
    ap.add_argument("--review-ms", type=float, default=300,
                    help="search time per decision when reviewing (0 = blueprint only)")
    ap.add_argument("--weights", default=None,
                    help="population file for --bots neural (default weights/population.pt)")
    ap.add_argument("--stack", type=int, default=STARTING_STACK)
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds to pause after each bot action")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--watch", nargs="?", type=int, const=100_000, default=0,
                    metavar="HANDS",
                    help="no human: six bots play (optionally only HANDS hands)")
    ap.add_argument("--show-cards", action="store_true",
                    help="watch mode: show every player's hole cards")
    ap.add_argument("--ui", choices=["rich", "plain"], default="rich")
    ap.add_argument("--name", default="You", help="your name at the table")
    ap.add_argument("--diagnostic", action="store_true",
                    help="practice mode: see every player's cards during play and "
                         "in the review, and each MCCFR bot's plan in the action log "
                         "(bots still learn only from public actions)")
    ap.add_argument("--auto-muck", action="store_true",
                    help="muck automatically when your hand can't win at showdown")
    ap.add_argument("--no-review", action="store_true",
                    help="do not write the decision review after the tournament")
    ap.add_argument("--review-dir", default=None,
                    help="where to write review files (default reviews/<date_time>)")
    args = ap.parse_args()

    cfr = settings = None
    if args.bots == "mccfr":
        from pokerbot.mccfr import CfrConfig, SearchSettings, default_config
        cfr = CfrConfig(args.cfr_config) if args.cfr_config else default_config()
        settings = SearchSettings(mode="blueprint" if args.no_search else "search",
                                  search_ms=args.search_ms, threads=args.threads, depth=args.depth)
        settings.max_jam_spr = args.max_jam_spr
        if args.no_model:
            settings.rnr_max = 0.0
        print(f"MCCFR bots: {len(cfr.strategies())} blueprints from {cfr.blueprint_dir}")

    observer, log = None, print
    plan_log = None
    if args.ui == "rich":
        from pokerbot.ui import RichHuman, TableView, WatchPacer
        log = None
        if args.watch:
            view = TableView(hero=None, show_all=args.show_cards)
            observer = WatchPacer(view, pause=max(2.0, 3 * args.delay))
            agents = make_bots(args.bots, args.weights, k=6, cfr=cfr, settings=settings, seed=args.seed)
        else:
            view = TableView(hero=0, show_all=args.diagnostic)
            observer = view
            if args.diagnostic:
                plan_log = lambda msg: view.log.append((msg, "log"))  # noqa: E731
            agents = [RichHuman(view, args.name, auto_muck=args.auto_muck)] + make_bots(
                args.bots, args.weights, cfr=cfr, settings=settings, seed=args.seed, log=plan_log)
    elif args.watch:
        agents = make_bots(args.bots, args.weights, k=6, cfr=cfr, settings=settings, seed=args.seed)
    else:
        plan_log = print if args.diagnostic else None
        agents = [HumanCLI(args.name, auto_muck=args.auto_muck)] + make_bots(
            args.bots, args.weights, cfr=cfr, settings=settings, seed=args.seed, log=plan_log)

    recorder = None
    if not args.watch and not args.no_review:
        from pokerbot.review import ReviewRecorder
        recorder = ReviewRecorder(agents[0])
        agents = [recorder] + [recorder.watch(b) for b in agents[1:]]

    table = Table(agents, starting_stack=args.stack, seed=args.seed, log=log,
                  bot_delay=args.delay if (args.ui == "rich" or not args.watch) else 0.0,
                  observer=observer)
    try:
        ranking = table.run(max_hands=args.watch or None,
                            stop_when_busted=None if args.watch else 0)
    except (KeyboardInterrupt, SystemExit):
        print("\nGame stopped.")
        ranking = None
    if ranking:
        print("\nFinal standings:")
        for place, seat in enumerate(ranking, 1):
            print(f"  {place}. {table.names[seat]}  ({table.stacks[seat]:,} chips)")
    if recorder is not None and recorder.hands:
        if args.bots == "neural" or cfr is None:
            from pokerbot.review import write_reviews
            from pokerbot.trained import DEFAULT_PATH, load_champions
            if cfr is None and args.bots != "neural":
                from pokerbot.mccfr import default_config
                cfr = default_config()
            else:
                print("Reviewing your decisions against the top six neural bots...")
                out = write_reviews(recorder, table.history,
                                    load_champions(args.weights or DEFAULT_PATH, k=6),
                                    args.review_dir, reveal_all=args.diagnostic)
                print(f"Review written to {out}/ (start with summary.txt)")
                return
        from pokerbot.mccfr import SearchSettings
        from pokerbot.mccfr.review import write_mccfr_reviews
        rs = SearchSettings(mode="search" if args.review_ms > 0 else "blueprint",
                            search_ms=args.review_ms, threads=args.threads, depth=args.depth,
                            menus=cfr.search_menus())
        print("Reviewing your decisions against the MCCFR bot...")
        out = write_mccfr_reviews(recorder, cfr, rs, args.review_dir, reveal_all=args.diagnostic)
        print(f"Review written to {out}/ (start with summary.txt)")


if __name__ == "__main__":
    main()
