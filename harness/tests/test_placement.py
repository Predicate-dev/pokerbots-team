"""`place_value`: a stack's worth when the game is ranked by finishing place."""

import itertools

import pytest

from poker_harness.placement import place_value, rating_points, rating_weights

EVEN = {f"seat{i}": 200 for i in range(6)}
NOISE = 24.0  # chips per hand (12 BB at the 1/2 game)
EVEN_VALUE = sum(rating_weights(6)) / 6


def value(stacks: dict[str, int], hands_left: int, before: dict[str, int] | None = None) -> float:
    return place_value(stacks, "seat0", hands_left, NOISE, before)


def test_equal_rated_points_and_large_penalty_for_finishing_last() -> None:
    points = [rating_points(place) for place in range(1, 7)]
    assert points == pytest.approx([0.83, 0.63, 0.38, 0.05, -0.45, -1.45], abs=0.004)
    weights = rating_weights(6)
    assert weights[4] - weights[5] == pytest.approx(5 * (weights[0] - weights[1]))


def test_even_stacks_have_a_uniform_chance_at_each_place_until_the_end() -> None:
    assert value(EVEN, 29) == EVEN_VALUE
    assert value(EVEN, 0) == 1.0  # finished ties share a place


def test_with_no_hands_left_it_is_the_exact_ranking_and_margin_does_not_matter() -> None:
    by_one = EVEN | {"seat0": 201}
    by_a_lot = EVEN | {"seat0": 600}
    assert value(by_one, 0) == value(by_a_lot, 0) == 1.0
    assert value(EVEN | {"seat0": 199}, 0) == 0.0
    assert value({"seat0": 250, "seat1": 300, "seat2": 250, "seat3": 100}, 0) == rating_weights(4)[1]


def test_the_same_lead_is_worth_more_with_fewer_hands_left() -> None:
    ahead = EVEN | {"seat0": 240, "seat1": 160}
    assert EVEN_VALUE < value(ahead, 29) < value(ahead, 5) < value(ahead, 1) < value(ahead, 0) == 1.0


def test_more_chips_are_never_worse_but_each_extra_chip_is_worth_less_when_ahead() -> None:
    values = [value(EVEN | {"seat0": stack}, 10) for stack in (200, 300, 400, 500)]
    assert values == sorted(values)
    gains = [b - a for a, b in itertools.pairwise(values)]
    assert gains[0] > gains[1] > gains[2]


def test_a_coin_flip_for_the_whole_stack_costs_the_leader_and_helps_the_last_seat() -> None:
    leader = {"seat0": 400, "seat1": 200, "seat2": 200, "seat3": 200, "seat4": 200}
    wins, loses = leader | {"seat0": 600, "seat1": 0}, leader | {"seat0": 200, "seat1": 400}
    assert (value(wins, 3) + value(loses, 3)) / 2 < value(leader, 3)

    trailer = {"seat0": 100, "seat1": 300, "seat2": 300, "seat3": 300, "seat4": 200}
    wins, loses = trailer | {"seat0": 200, "seat4": 100}, trailer | {"seat0": 0, "seat4": 300}
    assert (value(wins, 3, trailer) + value(loses, 3, trailer)) / 2 > value(trailer, 3)


def test_a_busted_seat_is_last_however_many_hands_are_left() -> None:
    assert value(EVEN | {"seat0": 0, "seat1": 400}, 29) == 0.0
    assert value(EVEN | {"seat0": 1, "seat1": 399}, 29) > 0.0
    assert value(EVEN | {"seat1": 0, "seat0": 400}, 29) > 0.9


def test_seats_that_are_both_out_rank_by_when_and_with_what_they_busted() -> None:
    before = {"seat0": 80, "seat1": 50, "seat2": 0, "seat3": 300}
    after = {"seat0": 0, "seat1": 0, "seat2": 0, "seat3": 430}
    # Above seat1 (busted the same hand with less) and seat2 (already out), below seat3.
    assert value(after, 5, before) == rating_weights(4)[1]
    assert place_value(after, "seat1", 5, NOISE, before) == rating_weights(4)[2]
    assert place_value(after, "seat2", 5, NOISE, before) == 0.0
    assert value(after, 5) == rating_weights(4)[1]  # without `before`, seats that are out tie


def test_it_needs_the_hero_and_someone_else() -> None:
    with pytest.raises(ValueError):
        place_value({"seat1": 200, "seat2": 200}, "seat0", 3, NOISE)
    with pytest.raises(ValueError):
        place_value({"seat0": 200}, "seat0", 3, NOISE)
