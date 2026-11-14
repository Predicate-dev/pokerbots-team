"""Training-only tasks for the forked harness. Nothing here is an official dataset; `tasks.py` is unchanged.

Three families, each using an opt-in feature of the harness (`harness/src/poker_harness/training.py`, `grading.py`):

- `strong_hand_tasks`: the official single-hand task, on deals where the policy seat holds a playable hand. Trash
  hands are folded by every attempt, so a GRPO group on one has no gaps and teaches nothing. A share of weak hands is
  kept on purpose: a model that only ever sees good cards never practices folding.
- `endgame_tasks`: the last K hands of a 30-hand game from seeded, uneven stacks (short and big stacks, some seats
  already out), graded on the finishing place (`grader_params.reward = "placement"`), which is what the arena rates.
- `mixed_table_tasks`: the first K hands of a game against a seeded mix of opponents: scripted players (no model
  call), the opponent model held to a style by a persona, and the plain opponent model. The scratchpad written after
  one hand is read in the next, so notes about a seat can pay off inside the task.

The personas are generic poker styles written here. They are not the arena's house players, whose prompts and play
are not public.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

from ac2.runtime import Message, Task

from .tasks import SEATS, TABLE_RULES, Opponents, hand_task

HARNESS_SRC = Path(__file__).resolve().parents[1] / "harness" / "src"
GAME_HANDS = 30  # the event's game length; windows of it keep the prompts' "Hand N of 30"
TOTAL_CHIPS_PER_SEAT = 200
MAX_TASK_HANDS = 4  # AC2 carries at most 1 MiB per rollout, and a task's token budget covers all of its hands
MIN_LIVE_STACK = 4  # two big blinds: the smallest stack a seat still in the game is given

SCRIPTED_STYLES = ("scripted:maniac", "scripted:nit", "scripted:calling_station", "scripted:random")
PERSONAS: dict[str, str] = {
    "tight_passive": (
        "Your style at this table: tight and passive. Fold all but strong starting hands, prefer calling to raising, "
        "and give up when an opponent bets big without a strong made hand."
    ),
    "loose_aggressive": (
        "Your style at this table: loose and aggressive. Play many hands, raise often before the flop, and keep "
        "betting after it with draws and bluffs as well as made hands."
    ),
    "calling_station": (
        "Your style at this table: you hate folding. Call bets with any pair or any draw, rarely raise, and rarely "
        "fold once you have put chips in the pot."
    ),
    "short_tempered": (
        "Your style at this table: you play well until you lose a pot, then you play too many hands too hard for "
        "the next few hands before settling down."
    ),
    "talker": (
        "Your style at this table: you use the `say` message on most actions, sometimes truthfully and sometimes "
        "not, and you play a solid, ordinary game underneath."
    ),
    "by_the_book": (
        "Your style at this table: careful and mathematical. Play by position and pot odds, bet for value with "
        "strong hands, bluff rarely, and avoid risking your whole stack without a very strong hand."
    ),
}


def _spots() -> Any:
    """`poker_harness.spots`, from this repo's harness. It needs PokerKit (the root dev group installs it)."""
    if str(HARNESS_SRC) not in sys.path:
        sys.path.insert(0, str(HARNESS_SRC))
    try:
        from poker_harness import spots
    except ImportError as exc:  # most likely PokerKit is missing from this environment
        raise SystemExit(f"can't import the harness ({exc}); run `uv sync` in the repo root first") from exc
    return spots


def strong_hand_tasks(
    count: int,
    base_seed: int,
    opponents: Opponents,
    min_chen: int = 6,
    weak_fraction: float = 0.15,
    seats: int = SEATS,
) -> list[Task]:
    """Single-hand tasks whose policy seat holds a hand of at least `min_chen` (Bill Chen's preflop score; 6 is about
    the top third of hands), except for a `weak_fraction` share that holds less. Seats still cycle through every
    position. The deck depends only on the seed, so the hole cards are known before the hand is played."""
    if not 0.0 <= weak_fraction < 1.0:
        raise ValueError("weak_fraction must be at least 0 and below 1")
    spots = _spots()
    rng = random.Random(base_seed)
    tasks = []
    for i in range(count):
        seat = i % seats
        want_weak = rng.random() < weak_fraction
        for _ in range(10_000):
            seed = rng.getrandbits(48)
            if (spots.chen_score(spots.hole_cards(seed, seat, seats=seats)) < min_chen) == want_weak:
                break
        else:
            raise ValueError(f"no deal with a Chen score {'below' if want_weak else 'of at least'} {min_chen}")
        tasks.append(hand_task(seed, seat, opponents, seats))
    return tasks


def endgame_stacks(rng: random.Random, policy_seat: int, seats: int = SEATS) -> list[int]:
    """Seeded stacks for a late hand: all the game's chips, some seats already out (0), the policy seat short,
    medium or big, and the rest split unevenly among the other seats still in."""
    total = TOTAL_CHIPS_PER_SEAT * seats
    busted = min(rng.choices([0, 1, 2, 3], weights=[45, 30, 18, 7])[0], seats - 2)
    others = [s for s in range(seats) if s != policy_seat]
    live_others = sorted(rng.sample(others, len(others) - busted))
    low, high = {"short": (8, 60), "medium": (100, 320), "big": (360, 620)}[rng.choice(["short", "medium", "big"])]
    mine = min(rng.randint(low, high), total - MIN_LIVE_STACK * len(live_others))
    rest = total - mine
    weights = [rng.random() ** 1.5 + 0.08 for _ in live_others]
    shares = [max(MIN_LIVE_STACK, round(rest * w / sum(weights))) for w in weights]
    shares[shares.index(max(shares))] += rest - sum(shares)  # rounding goes to the biggest stack
    stacks = [0] * seats
    stacks[policy_seat] = mine
    for seat, chips in zip(live_others, shares, strict=True):
        stacks[seat] = chips
    return stacks


def endgame_task(
    seed: int,
    policy_seat: int,
    opponents: Opponents,
    hands_left: int,
    stacks: list[int],
    chip_weight: float | None = None,
) -> Task:
    """The last `hands_left` hands of a 30-hand game from `stacks`, graded on the finishing place."""
    if not 1 <= hands_left <= MAX_TASK_HANDS:
        raise ValueError(f"hands_left must be 1-{MAX_TASK_HANDS}")
    seats = len(stacks)
    first_hand = GAME_HANDS - hands_left + 1
    return Task(
        input=[
            Message(
                role="user",
                content=f"Play hands {first_hand}-{GAME_HANDS} of a {seats}-seat no-limit hold'em game from seat "
                f"{policy_seat}.",
            )
        ],
        env_params=TABLE_RULES
        | {
            "seed": seed,
            "seats": seats,
            "total_hands": GAME_HANDS,
            "first_hand": first_hand,
            "initial_stacks": stacks,
            "policy_seats": [policy_seat],
            **opponents.env_params(),
        },
        grader_params={"reward": "placement"} | ({} if chip_weight is None else {"chip_weight": chip_weight}),
    )


def endgame_tasks(
    count: int,
    base_seed: int,
    opponents: Opponents,
    max_hands_left: int = 3,
    chip_weight: float | None = None,
    seats: int = SEATS,
) -> list[Task]:
    """Endgames of 1 to `max_hands_left` hands; the policy seat cycles through every seat."""
    if not 1 <= max_hands_left <= MAX_TASK_HANDS:
        raise ValueError(f"max_hands_left must be 1-{MAX_TASK_HANDS}")
    rng = random.Random(base_seed)
    tasks = []
    for i in range(count):
        seat = i % seats
        seed = rng.getrandbits(48)
        hands_left = rng.randint(1, max_hands_left)
        tasks.append(endgame_task(seed, seat, opponents, hands_left, endgame_stacks(rng, seat, seats), chip_weight))
    return tasks


def opponent_mix(
    rng: random.Random, policy_seat: int, seats: int, scripted_share: float, persona_share: float
) -> dict[str, Any]:
    """`env_params.opponent_seats` for one table: each other seat is scripted, a persona, or (left out) the plain
    opponent model. Keys are strings, as JSON keeps them."""
    mix: dict[str, Any] = {}
    for seat in range(seats):
        if seat == policy_seat:
            continue
        roll = rng.random()
        if roll < scripted_share:
            mix[str(seat)] = rng.choice(SCRIPTED_STYLES)
        elif roll < scripted_share + persona_share:
            mix[str(seat)] = {"persona": PERSONAS[rng.choice(sorted(PERSONAS))]}
    return mix


def mixed_table_task(seed: int, policy_seat: int, opponents: Opponents, hands: int, mix: dict[str, Any], seats: int) -> Task:
    """The first `hands` hands of a 30-hand game against `mix`, graded on EV summed over those hands."""
    if not 1 <= hands <= MAX_TASK_HANDS:
        raise ValueError(f"hands must be 1-{MAX_TASK_HANDS}")
    return Task(
        input=[
            Message(
                role="user",
                content=f"Play hands 1-{hands} of a {seats}-seat no-limit hold'em game from seat {policy_seat}.",
            )
        ],
        env_params=TABLE_RULES
        | {
            "seed": seed,
            "seats": seats,
            "total_hands": GAME_HANDS,
            "last_hand": hands,
            "policy_seats": [policy_seat],
            "opponent_seats": mix,
            "ev_labels": True,
            "ev_runouts": 64,
            **opponents.env_params(),
        },
        # K hands of EV in the band one hand uses: K big blinds per reward point.
        grader_params={"reward": "ev", "ev_unit_bb": hands},
    )


def mixed_table_tasks(
    count: int,
    base_seed: int,
    opponents: Opponents,
    hands: int = 3,
    scripted_share: float = 0.4,
    persona_share: float = 0.4,
    seats: int = SEATS,
) -> list[Task]:
    """Mixed-opponent tables; the policy seat cycles through every seat."""
    if scripted_share < 0 or persona_share < 0 or scripted_share + persona_share > 1:
        raise ValueError("scripted_share and persona_share must be shares that sum to at most 1")
    rng = random.Random(base_seed)
    tasks = []
    for i in range(count):
        seat = i % seats
        seed = rng.getrandbits(48)
        mix = opponent_mix(rng, seat, seats, scripted_share, persona_share)
        tasks.append(mixed_table_task(seed, seat, opponents, hands, mix, seats))
    return tasks


# The scripted imitations of the arena's six house bots (harness/src/poker_harness/house_bots.py).
HOUSE_STYLES = ("house_alice", "house_bob", "house_stan", "house_carlos", "house_pete", "house_tim")


def house_table_task(
    seed: int, policy_seat: int, opponents: Opponents, first_hand: int, hands: int, stacks: list[int] | None,
    lineup: dict[str, str], placement: bool,
) -> Task:
    """Hands `first_hand`..`first_hand + hands - 1` of a 30-hand game against scripted house-bot imitations. Every
    opponent seat is scripted, so the task makes no opponent calls. Graded on finishing place (`placement`) or on EV
    summed over its hands."""
    if not 1 <= hands <= MAX_TASK_HANDS:
        raise ValueError(f"hands must be 1-{MAX_TASK_HANDS}")
    last_hand = min(GAME_HANDS, first_hand + hands - 1)
    seats = len(lineup) + 1
    params: dict[str, Any] = {
        "seed": seed,
        "seats": seats,
        "total_hands": GAME_HANDS,
        "first_hand": first_hand,
        "last_hand": last_hand,
        "policy_seats": [policy_seat],
        "opponent_seats": lineup,
        **opponents.env_params(),
    }
    if stacks is not None:
        params["initial_stacks"] = stacks
    if not placement:
        params |= {"ev_labels": True, "ev_runouts": 64}
    return Task(
        input=[
            Message(
                role="user",
                content=f"Play hands {first_hand}-{last_hand} of a {seats}-seat no-limit hold'em game from seat "
                f"{policy_seat}.",
            )
        ],
        env_params=TABLE_RULES | params,
        grader_params={"reward": "placement"} if placement else {"reward": "ev", "ev_unit_bb": last_hand - first_hand + 1},
    )


def house_table_tasks(
    count: int, base_seed: int, opponents: Opponents, hands: int = 1, placement: bool = False, seats: int = SEATS,
    *, talk_mode: str = "normal",
) -> list[Task]:
    """Windows of event games against the house-bot imitations. A third start on hand 1 with full stacks; the rest
    start mid-game with uneven stacks (and sometimes busted seats). Each table seats five of the six styles in a
    seeded order; the policy seat cycles through every seat."""
    if seats != SEATS:
        raise ValueError("house tables are 6-max")
    rng = random.Random(base_seed)
    tasks = []
    for i in range(count):
        seat = i % seats
        seed = rng.getrandbits(48)
        styles = rng.sample(HOUSE_STYLES, seats - 1)
        lineup = {str(s): f"scripted:{style}" for s, style in zip([x for x in range(seats) if x != seat], styles)}
        if rng.random() < 1 / 3:
            first_hand, stacks = 1, None
        else:
            first_hand = rng.randint(2, GAME_HANDS - hands + 1)
            stacks = endgame_stacks(rng, seat, seats)
        task = house_table_task(seed, seat, opponents, first_hand, hands, stacks, lineup, placement)
        if talk_mode != "normal":
            task.env_params["house_talk"] = talk_mode
        tasks.append(task)
    return tasks
