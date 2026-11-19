import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from pokerbot import (HAND_DIM, Bot, Decision, Table, check_call,
                      context_fields)
from pokerbot.cards import best_hand, parse_card


def cards(s):
    return [parse_card(x) for x in s.split()]


def test_evaluator():
    sf = best_hand(cards("As Ks Qs Js Ts 2d 3c"))
    quads = best_hand(cards("9s 9d 9h 9c Ks 2d 3c"))
    wheel = best_hand(cards("Ad 2s 3h 4c 5d Kc Qc"))
    six_high = best_hand(cards("2s 3h 4c 5d 6d Kc Qc"))
    assert sf > quads > six_high > wheel
    assert best_hand(cards("Ah Ad Kc Kd 2s 3h 7c")) > best_hand(cards("Ah Ad Qc Qd Js Th 7c"))


def random_bot(rng):
    def decide(history, current):
        f = context_fields(current)
        r = rng.random()
        if r < 0.15:
            return Decision.fold()
        if r < 0.35 and f["can_raise"].item():
            bb_chips = f["big_blind_chips"].item() * 1000
            return Decision.raise_to(int(rng.choice([
                f["min_raise_to_bb"].item(), f["max_raise_to_bb"].item(),
                f["pot_bb"].item()]) * bb_chips))
        return Decision.call()
    return decide


def test_random_tournaments_conserve_chips():
    for seed in range(30):
        rng = random.Random(seed)
        agents = [Bot(f"B{i}", random_bot(rng) if i % 2 else check_call)
                  for i in range(6)]
        t = Table(agents, seed=seed)
        total = sum(t.stacks)
        while len(t.alive()) > 1 and t.hand_count < 400:
            st = t.play_hand()
            assert sum(t.stacks) == total, (seed, st.summary())
            assert all(s >= 0 for s in t.stacks)
            assert sum(st.net) == 0
        ranking = t.run()
        assert sorted(ranking) == list(range(6))
        assert len(t.alive()) == 1


def test_tensor_shapes_and_masking():
    seen = []

    def spy(history, current):
        seen.append((history.shape, current.shape, current.clone()))
        return check_call(history, current)

    agents = [Bot("spy", spy)] + [Bot(f"B{i}", check_call) for i in range(5)]
    t = Table(agents, seed=1)
    for _ in range(3):
        t.play_hand()
    assert seen[0][0] == (0, HAND_DIM)
    assert seen[-1][0][0] == 2 and seen[-1][1] == (HAND_DIM,)
    # Current-hand tensor shows exactly the spy's own two cards.
    hole = context_fields(seen[-1][2])["hole_cards"]
    assert hole.sum().item() == 2 and hole[:52].sum().item() == 2
    # History shows opponents' cards when shown at showdown.
    past = t.history.tensor(0)
    assert context_fields(past)["hole_cards"].sum(dim=-1).min().item() >= 2


def test_side_pots():
    agents = [Bot(f"B{i}", check_call) for i in range(3)]
    t = Table(agents, seed=0)
    from pokerbot import HandState
    st = HandState(hand_id=1, num_seats=3, names=t.names, button=0,
                   small_blind=10, big_blind=20, ante=0, level=1,
                   starting_stacks=[100, 500, 500])
    st.hole_cards = [cards("As Ad"), cards("Ks Kd"), cards("Qs Qd")]
    st.board = cards("2c 7d 9h Jc 3s")
    st.committed_total = [100, 400, 400]
    st.stacks = [0, 100, 100]
    t._settle(st)
    assert st.stacks == [300, 700, 100]   # aces win main, kings win side


def test_schedule():
    t = Table([Bot(f"B{i}", check_call) for i in range(6)], seed=0)
    expect = {0: (10, 20, 2), 7: (10, 20, 2), 8: (15, 30, 3), 71: (75, 150, 15),
              72: (100, 200, 20), 77: (100, 200, 20), 78: (125, 250, 25),
              119: (400, 800, 80), 120: (500, 1000, 100), 500: (500, 1000, 100)}
    for hands, blinds in expect.items():
        t.hand_count = hands
        lvl = t.schedule[t.level_index()]
        assert (lvl.small_blind, lvl.big_blind, lvl.ante) == blinds, hands



def test_opponent_model_reads_history():
    from pokerbot.opponent import OpponentModel
    from pokerbot.sparring import all_in_bot, premium_only_bot
    agents = [Bot("cc", check_call), Bot("shove", all_in_bot),
              Bot("nit", premium_only_bot)] + [Bot(f"c{i}", check_call) for i in range(3)]
    t = Table(agents, seed=3)
    for _ in range(25):
        if len(t.alive()) < 2:
            break
        t.play_hand()
    m = OpponentModel(prior_n=4, decay=1.0)
    m.update(t.history.tensor(0))
    if t.stacks[1] > 0:
        assert m.rate(1, "shove") > 0.6
    assert m.rate(2, "vpip") < 0.2
    assert m.seen == t.history.size


def test_neural_bot_plays_legal_tournament():
    from pokerbot.neural import NeuralBot, Teacher, new_genome
    agents = [Bot(f"N{i}", NeuralBot(new_genome(), seed=i)) for i in range(3)] + \
             [Bot(f"T{i}", Teacher(i)) for i in range(3)]
    t = Table(agents, seed=5)
    total = sum(t.stacks)
    ranking = t.run()
    assert sorted(ranking) == list(range(6)) and sum(t.stacks) == total



def test_review_files(tmp_dir=None):
    import tempfile
    from pokerbot.review import ReviewRecorder, write_reviews
    from pokerbot.sparring import make_random_aggro
    from pokerbot.trained import DEFAULT_PATH, load_champions, trained_bots
    if not DEFAULT_PATH.exists():
        import pytest
        pytest.skip("the v3 neural bots (weights/) are not installed")
    me = ReviewRecorder(Bot("You", make_random_aggro(1)))
    t = Table([me] + trained_bots(), seed=2)
    for _ in range(6):
        if len(t.alive()) < 2:
            break
        t.play_hand()
    out = write_reviews(me, t.history, load_champions(k=6),
                        tmp_dir or tempfile.mkdtemp())
    files = sorted(p.name for p in out.iterdir())
    assert "summary.txt" in files and "hand_001.csv" in files and len(files) == 2 * t.hand_count + 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all tests passed")
