"""Sample grader: how the harness scores each rollout for AC2.

How it is wired
---------------
A run that names no grader is graded by the harness itself, through AC2's Grader API (routes in `server.py`):

    POST /submit_grading      start grading one rollout; AC2 may repeat it, so it is idempotent
    POST /get_grading_status  queued | running | completed | failed
    POST /collect_score       the reply `grade_log` builds below: {"score", "reasoning", "artifacts"}

`server.py` calls `check_grader_params` with the task's `grader_params`, then `grade_log` with the finished game log,
or `failure_score` when the rollout never produced one. Nothing in this module talks to the network.

What it scores (deliberately naive)
-----------------------------------
The trained seat's result in big blinds. With `env_params.ev_labels`, an all-in before the river counts its expected
share of the pot instead of the cards that came (`ev.py`); otherwise it is the chips actually won.

    score = 6 + clip(hand_reward, -5, +5),  hand_reward = clip(EV, -100 BB, +100 BB) / 1 BB

So every real hand scores in [1, 11]. Using 1 BB per reward point (not 20) makes typical single-hand pots
actually use that band, so GRPO sees real within-group gaps. Model-caused failures score exactly 0: a format
error (no usable `submit_action`, or a reply cut off at the token cap), or a rollout AC2 reports as failed or
truncated. Rewards are never negative, a failure is always worse than any real hand, and failed samples still
train instead of being dropped. The +6 is a shift, not a rescale, so group-relative advantages (GRPO) are unchanged.

Extension points (search for "EXTENSION POINT")
-----------------------------------------------
1. `hand_reward`: the raw reward for a played hand. Replace it with your own.
2. `GRADER_PARAMS`: per-task knobs your reward reads from the dataset's `grader_params`.
3. `artifacts`: anything you add to the reply is stored with each score row on AC2, for analysis.
Keep failures at 0 and real hands above it, or a model can learn that failing pays.

Placement reward (`grader_params.reward = "placement"`)
----------------------------------------------------
The arena rates an entry on where it finishes a game, not on its chips: first by 1 chip equals first by 400, and a
busted seat ranks below every seat still in. Chips are a fair stand-in for that on a fresh first hand, but not late
in a game, where a leader should protect its lead and a trailing seat has to gamble. A task that sets
`reward: "placement"` is scored on how much it moved the trained seat's expected finishing place instead:

    hand_reward = placement_scale * (place_value(after) - place_value(before))

`place_value` (`placement.py`) is expected rating value, with the equal-rated Plackett-Luce weights scaled to [0, 1].
`hands_left` is how many hands the game still has after this task's hands. `placement_scale` (default 5) scales
changes in expected rating value; `placement_noise_bb` (default 12) is how far the gap between two stacks drifts per
hand. When EV labels are available, a heads-up all-in before the river averages the rating value of each outcome;
every other hand uses the stacks as they ended. Use it with mid-game tasks (`env_params.starting_stacks`,
`first_hand`, `play_hands`); tasks without `reward` are scored on chips exactly as before.

Ledger accuracy (`grader_params.ledger_weight`)
----------------------------------------------
At the event the scratchpad is the model's only memory of a game, and `ledger.py` gives it a fixed layout whose
numbers have one right answer after every hand. A task with `ledger_weight: w` (0 < w < 10) hands `w` points of the
10-point band to that answer:

    score = 6 + clip(hand_reward, -5, +5) * (10 - w) / 10 + w * (accuracy - 0.5)

`accuracy` is the share of the reference ledger's facts (stacks and counters for every seat) that the trained seat's
scratchpad gets right after its last after-hand update, starting from the scratchpad it was given (blank, or the
task's `starting_ledger`). A perfect ledger and a missing one are `w` points apart whatever the cards did, real
hands still score in [1, 11], and failures still score 0. It works with either reward.

Personas (`grader_params.persona_weight`)
-----------------------------------------
`persona.py` defines two ways to size bets, "small" and "big", held for 10 hands at a time and named on the
ledger's MODE line. A task with `persona_weight: p` hands `p` more points of the band to playing the part:

    score = 6 + clip(hand_reward, -5, +5) * (10 - w - p) / 10 + w * (accuracy - 0.5) + p * (persona - 0.5)

`persona` averages two things: whether the MODE line after the hand is the right one (kept through its block, or
drawn from the seat's first hole card when a new block starts), and, when a mode was in force and the seat opened
or bet, the share of those sizes that fit it. The hand itself is scored the same in either mode, so neither persona
can be a throwaway: both have to win.

On single-hand tasks the scratchpad starts blank and its after-hand update is never
read, so nothing here rewards or exercises scratchpad use; how to train that is up to you.

Training-only rewards (opt-in through the task's `grader_params`; a task without them scores exactly as above)
-------------------------------------------------------------------------------------------------------------
    reward       "ev" (the default above) or "placement".
    chip_weight  placement only: the share of the reward that comes from chips, 0-1 (default 0.25).
    ev_unit_bb   ev only: big blinds per reward point (default 1). A K-hand task sums K hands of EV, so K keeps the
                 sum inside the same [-5, +5] band a single hand uses.

"placement" is for multi-hand carry-over games, where the arena rates the finishing place, not the chips. The seat's
place comes from the same "elimination" ranking the arena uses (`game.placements`: seats still in rank above every
busted seat, by chips; busted seats by when they busted). With n seats that had chips when the task started (a
seat that starts at 0 always ranks below them, so it doesn't count):

    rank_share = (n - place) / (n - 1)                       1 for first place, 0 for last
    chip_share = the seat's final chips / all chips in play   0 when busted
    score      = 1 + 10 * ((1 - chip_weight) * rank_share + chip_weight * chip_share)

so a played game still scores in [1, 11] and a model-caused failure still scores exactly 0. The chip term is there
because place alone is a step function: sixteen attempts at the same task often finish in the same place, and GRPO
learns nothing from a group with no gaps. With several policy seats the reward is their average.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .ev import _contributions
from .game import placements
from .ledger import accuracy as ledger_accuracy
from .ledger import parse as parse_ledger
from .ledger import update as update_ledger
from .persona import sizing_checks
from .placement import DEFAULT_NOISE_BB, place_value, rating_weights

REWARD_OFFSET = 6.0  # shifts the clipped reward [-5, +5] to [1, 11]
REWARD_FLOOR, REWARD_CEILING = -5.0, 5.0
FAILURE_SCORE = 0.0  # every model-caused failure
EV_CLIP_BB = 100.0  # one hand can't swing the reward by more than a full stack
EV_UNIT_BB = 1.0  # 1 BB of EV = 1 reward point so typical hands use most of [1, 11]
MAX_SCORE_BYTES = 1 << 20  # AC2 rejects collect_score replies over 1 MiB

# EXTENSION POINT 2: the `grader_params` keys your reward accepts. Unknown keys fail the grading, so a typo in a
# dataset can't silently change the reward.
PLACEMENT_PARAMS = frozenset({"hands_left", "placement_scale", "placement_noise_bb"})
NOTE_WEIGHTS = ("ledger_weight", "persona_weight")  # points of the 10-point band graded on the scratchpad
GRADER_PARAMS: frozenset[str] = frozenset({"reward", "chip_weight", "ev_unit_bb", *NOTE_WEIGHTS}) | PLACEMENT_PARAMS
REWARDS = ("ev", "chips", "placement")
DEFAULT_CHIP_WEIGHT = 0.25
PLACEMENT_SCALE = 5.0  # a full rating-value swing fits within the score band without clipping other outcomes


class GradingError(ValueError):
    """The log can't be scored. AC2 drops the sample (GRADER_ERROR) instead of training on a made-up reward."""


@dataclass
class SeatResult:
    """What the trained seat did over one game log."""

    ev_bb: float  # EV in big blinds (all-in luck removed when the log has EV labels)
    realized_bb: float  # chips actually won, in big blinds
    method: str  # how ev_bb was computed: "realized", "allin", "showdown", ...
    decisions: int
    format_errors: int
    folded: bool
    hands: int
    place_before: float | None = None  # expected normalized rating; only with the placement reward
    place_after: float | None = None
    ledger_accuracy: float | None = None  # share of the reference ledger the scratchpad got right; with ledger_weight
    persona: float | None = None  # right MODE line and sizes that fit the mode, in [0, 1]; with persona_weight


def check_grader_params(grader_params: dict[str, Any] | None) -> dict[str, Any]:
    """The task's `grader_params`, after refusing any key not in `GRADER_PARAMS`."""
    params = dict(grader_params or {})
    unknown = set(params) - GRADER_PARAMS
    if unknown:
        raise GradingError(f"unknown grader_params: {sorted(unknown)}")
    for key in NOTE_WEIGHTS:
        weight = params.get(key)
        if weight is not None and (isinstance(weight, bool) or not isinstance(weight, int | float) or weight <= 0):
            raise GradingError(f"grader_params.{key} must be a positive number")
    if sum(params.get(key) or 0 for key in NOTE_WEIGHTS) >= 10:
        raise GradingError(f"{' + '.join(NOTE_WEIGHTS)} must leave some of the 10 points for the hand itself")
    reward = params.get("reward", "ev")
    if reward not in REWARDS:
        raise GradingError(f"grader_params.reward must be one of {REWARDS}, not {reward!r}")
    if "chip_weight" in params and reward != "placement":
        raise GradingError("grader_params.chip_weight only applies to reward 'placement'")
    if "ev_unit_bb" in params:
        value = params["ev_unit_bb"]
        if isinstance(value, bool) or not isinstance(value, int | float) or not 1e-9 <= value <= 1e9:
            raise GradingError(f"grader_params.ev_unit_bb must be a number from 1e-09 to 1e+09, got {value!r}")
        if reward not in ("ev", "chips"):
            raise GradingError("grader_params.ev_unit_bb only applies to reward 'ev'")
    if reward == "placement":
        if "chip_weight" in params:
            value = params["chip_weight"]
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
                raise GradingError(f"grader_params.chip_weight must be a number from 0 to 1, got {value!r}")
            if set(params) & PLACEMENT_PARAMS:
                raise GradingError("chip_weight can't be combined with the expected-placement settings")
        elif "hands_left" in params:
            left = params.get("hands_left")
            if isinstance(left, bool) or not isinstance(left, int) or left < 0:
                raise GradingError("the placement reward needs grader_params.hands_left: hands left after this task, 0+")
            for key in ("placement_scale", "placement_noise_bb"):
                value = params.get(key, 1)
                if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                    raise GradingError(f"grader_params.{key} must be a positive number")
    elif set(params) & PLACEMENT_PARAMS:
        raise GradingError(f"{sorted(set(params) & PLACEMENT_PARAMS)} need grader_params.reward = 'placement'")
    return params


def _stacks_before(hand: dict[str, Any], names: list[str]) -> dict[str, int]:
    """Every seat's stack before `hand` (0 for a seat that was already out and so not dealt in)."""
    if "stacks_before" in hand:
        return {name: hand["stacks_before"].get(name, 0) for name in names}
    return {name: hand["starting_state"].get(name, {}).get("starting_stack", 0) for name in names}


def place_values(log: dict[str, Any], seat: str, params: dict[str, Any]) -> tuple[float, float]:
    """The trained seat's place value before the first hand of the log and after its last (`placement.py`)."""
    info, hands = log["session_info"], log["hands"]
    names = info["player_names"]
    noise = float(params.get("placement_noise_bb", DEFAULT_NOISE_BB)) * info["big_blind"]
    left = params["hands_left"]
    start = _stacks_before(hands[0], names)
    if start[seat] <= 0:
        raise GradingError(f"{seat} has no chips at the start of the task")
    before = place_value(start, seat, left + len(hands), noise)

    for hand in hands:  # once the seat is out its place is settled: only seats out no later than it rank below
        if seat in hand["starting_state"] and hand["ending_stacks"][seat] == 0:
            bust_hand = hand
            break
    else:
        bust_hand = None
    last = hands[-1]
    label = last.get("ev", {}).get(seat) or {}
    folded = {a["player"] for a in last["actions"] if a["action"]["type"] == "fold"}
    live = [name for name in last["starting_state"] if name not in folded]
    if len(hands) == 1 and label.get("method") == "allin" and len(live) == 2 and seat in live:
        # A tie and a mix of wins and losses have different rank value.
        (other,) = (n for n in live if n != seat)
        contrib = _contributions(last)
        pot = sum(contrib.values())
        behind = {name: start[name] - contrib.get(name, 0) for name in names}
        wins = place_value(behind | {seat: behind[seat] + pot}, seat, left, noise, start)
        loses = place_value(behind | {other: behind[other] + pot}, seat, left, noise, start)
        if "heads_up_outcomes" in label:
            win, tie, loss = label["heads_up_outcomes"]
            half, odd = divmod(pot, 2)
            split = behind | {name: behind[name] + half + (odd if i == 0 else 0) for i, name in enumerate(live)}
            ties = place_value(split, seat, left, noise, start)
            return before, win * wins + tie * ties + loss * loses
        return before, label["ev_share"] * wins + (1.0 - label["ev_share"]) * loses
    if bust_hand is not None:
        return before, place_value(bust_hand["ending_stacks"], seat, left, noise, _stacks_before(bust_hand, names))
    return before, place_value(log["final_stacks"], seat, left, noise)


def scratchpad_scores(log: dict[str, Any], seat: str) -> tuple[float, float]:
    """(ledger accuracy, persona score) for the trained seat, both in [0, 1].

    The reference ledger starts from the scratchpad the seat had before its first hand and adds every hand it was
    dealt into (`ledger.py`). Accuracy is how much of it the seat's final scratchpad gets right. The persona score
    averages whether the final MODE line is the reference's and, for hands played with a mode in force, the share of
    the seat's open raises and bets sized to fit it (`persona.py`)."""
    private = (log.get("private") or {}).get(seat)
    dealt = [hand for hand in log["hands"] if seat in hand["starting_state"]]
    if private is None or not dealt:
        raise GradingError(f"grading the scratchpad needs a scratchpad table and a hand dealt to {seat}")
    before = (private.get("hands") or {}).get(str(dealt[0]["hand_number"]), {}).get("scratchpad_before", "")
    reference = parse_ledger(before)
    total = reference.total if reference else 0  # the layout names the game's length; it is not graded
    big_blind = log["session_info"]["big_blind"]
    sized: list[bool] = []
    for hand in dealt:
        if reference and reference.mode and hand["hand_number"] <= reference.mode_through:
            sized += sizing_checks(hand, seat, reference.mode, big_blind)
        reference = update_ledger(reference, hand, seat, total)
    final = private.get("scratchpad", "")
    kept = parse_ledger(final)
    line_right = kept is not None and (kept.mode, kept.mode_through) == (reference.mode, reference.mode_through)
    parts = [float(line_right)] + ([sum(sized) / len(sized)] if sized else [])
    return ledger_accuracy(final, reference), sum(parts) / len(parts)


def seat_result(log: dict[str, Any], policy_seats: list[str], params: dict[str, Any] | None = None) -> SeatResult:
    """Sum the trained seat's results over every hand in the log."""
    policy = set(policy_seats)
    if not log.get("hands") or not policy:
        raise GradingError("the rollout has no hands or no policy seats")
    info = log["session_info"]
    big_blind = info["big_blind"]
    start = dict(zip(info["player_names"], info["starting_stacks"], strict=True))
    seats = [(hand, seat) for hand in log["hands"] for seat in policy if seat in hand["starting_state"]]
    actions = [a for hand in log["hands"] for a in hand["actions"] if a["player"] in policy]
    labels = [hand.get("ev", {}).get(seat) for hand, seat in seats]
    if labels and all(label is not None for label in labels):
        ev_bb = sum(label["ev_bb"] for label in labels)
        realized_bb = sum(label["realized_bb"] for label in labels)
        method = labels[0]["method"]
    else:  # no EV labels on the task: score the chips actually won
        realized_bb = ev_bb = sum(log["final_stacks"][seat] - start[seat] for seat in policy) / big_blind
        method = "realized"
    place_before = place_after = None
    if (params or {}).get("reward") == "placement" and "hands_left" in (params or {}):
        if len(policy) != 1:
            raise GradingError("the placement reward scores exactly one policy seat")
        place_before, place_after = place_values(log, next(iter(policy)), params or {})
    accuracy = persona = None
    if any((params or {}).get(key) for key in NOTE_WEIGHTS):
        if len(policy) != 1:
            raise GradingError(f"{' and '.join(NOTE_WEIGHTS)} score exactly one policy seat")
        accuracy, persona = scratchpad_scores(log, next(iter(policy)))
    return SeatResult(
        ledger_accuracy=accuracy if (params or {}).get("ledger_weight") else None,
        persona=persona if (params or {}).get("persona_weight") else None,
        place_before=place_before,
        place_after=place_after,
        ev_bb=ev_bb,
        realized_bb=realized_bb,
        method=method,
        decisions=len(actions),
        format_errors=sum(bool(a["model_response"].get("format_error")) for a in actions),
        folded=any(a["action"]["type"] == "fold" for a in actions),
        hands=len(log["hands"]),
    )


def hand_reward(result: SeatResult, params: dict[str, Any]) -> float:
    """EXTENSION POINT 1: the raw reward for a game the seat played without failing, before clipping and the +6.

    The default is EV in units of 1 BB, clipped to +-100 BB. `result` has everything the log says about the trained
    seat; read more from the log in `seat_result` if you need it. `params` is the task's checked `grader_params`.
    """
    if result.place_before is not None and result.place_after is not None:  # grader_params.reward = "placement"
        return float(params.get("placement_scale", PLACEMENT_SCALE)) * (result.place_after - result.place_before)
    return max(-EV_CLIP_BB, min(EV_CLIP_BB, result.ev_bb)) / float(params.get("ev_unit_bb", EV_UNIT_BB))


@dataclass
class Placement:
    """Where the trained seat finished a multi-hand game (averaged over the policy seats when there are several)."""

    place: float  # 1 = best; tied seats share a place
    seats: int  # seats with chips when the log started
    final_chips: float
    chip_share: float  # of all chips in play
    busted: bool  # any policy seat ended with no chips
    rating_value: float  # normalized rating value using every original seat, including prior eliminations


def placement(log: dict[str, Any], policy_seats: list[str]) -> Placement:
    """The policy seats' finish under the arena's "elimination" ranking, from the full game log."""
    if not log.get("hands") or not policy_seats:
        raise GradingError("the rollout has no hands or no policy seats")
    chips = log["final_stacks"]
    total = sum(chips.values())
    if total <= 0 or any(seat not in chips for seat in policy_seats):
        raise GradingError("the log has no final stacks for the policy seats")
    places = placements(log, "elimination")
    weights = rating_weights(len(places))
    # Seats that started the log with no chips always rank last, so they aren't places the policy can reach: rank
    # among the seats in play at the start, so 0 means last of those whatever the table's size.
    info = log["session_info"]
    in_play = sum(chips > 0 for chips in info.get("starting_stacks", [])) or len(places)
    count = len(policy_seats)
    mine = sum(chips[seat] for seat in policy_seats) / count
    return Placement(
        place=sum(places[seat] for seat in policy_seats) / count,
        seats=in_play,
        final_chips=mine,
        chip_share=mine / total,
        busted=any(chips[seat] == 0 for seat in policy_seats),
        rating_value=sum(weights[places[seat] - 1] for seat in policy_seats) / count,
    )


def placement_reward(result: Placement, params: dict[str, Any]) -> float:
    """The placement reward before the +1: 0 for last place with no chips, 10 for first place with every chip."""
    weight = float(params.get("chip_weight", DEFAULT_CHIP_WEIGHT))
    return 10.0 * ((1.0 - weight) * result.rating_value + weight * result.chip_share)


def failure_score(reason: str) -> dict[str, Any]:
    """The reply for a rollout that failed before producing a log (AC2 reported it failed or truncated)."""
    return {"score": FAILURE_SCORE, "reasoning": reason, "artifacts": {"failure": True}}


def grade_log(log: dict[str, Any], policy_seats: list[str], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score one finished game for its policy seats, as a `collect_score` reply."""
    result = seat_result(log, policy_seats, params)
    if (params or {}).get("reward") == "placement" and "hands_left" not in (params or {}):
        return grade_placement(log, policy_seats, result, params or {})
    if result.format_errors:  # model-caused failure: exactly 0, below every real hand
        score = FAILURE_SCORE
    else:
        played = max(REWARD_FLOOR, min(REWARD_CEILING, hand_reward(result, params or {})))
        # `ledger_weight` and `persona_weight` points of the band go to the scratchpad and to playing the part.
        notes = [(result.ledger_accuracy, "ledger_weight"), (result.persona, "persona_weight")]
        weighed = [(float((params or {})[key]), value) for value, key in notes if value is not None]
        played = played * (10.0 - sum(weight for weight, _ in weighed)) / 10.0
        played += sum(weight * (value - 0.5) for weight, value in weighed)
        score = REWARD_OFFSET + played
    placed = result.place_before is not None and result.place_after is not None
    place_note = f" Place value {result.place_before:.3f} -> {result.place_after:.3f}." if placed else ""
    place_artifacts = (
        {"reward": "placement", "place_before": result.place_before, "place_after": result.place_after}
        if placed
        else {}
    )
    if result.ledger_accuracy is not None:
        place_note += f" Ledger {result.ledger_accuracy:.0%} right."
        place_artifacts["ledger_accuracy"] = result.ledger_accuracy
    if result.persona is not None:
        place_note += f" Persona {result.persona:.0%}."
        place_artifacts["persona"] = result.persona
    return {
        "score": score,
        "reasoning": f"Policy seat {'/'.join(sorted(policy_seats))}: EV {result.ev_bb:+.1f} BB "
        f"(realized {result.realized_bb:+.1f}), {result.method}, {result.format_errors} format errors.{place_note}",
        # EXTENSION POINT 3: add any per-rollout numbers you want to analyze later.
        "artifacts": {
            "ev_bb": result.ev_bb,
            "realized_bb": result.realized_bb,
            "ev_method": result.method,
            "policy_decisions": result.decisions,
            "policy_format_errors": result.format_errors,
            "folded": result.folded,
            "hands": result.hands,
            **place_artifacts,
        },
    }


def grade_placement(
    log: dict[str, Any], policy_seats: list[str], result: SeatResult, params: dict[str, Any]
) -> dict[str, Any]:
    """`reward: "placement"`: the finishing place with a chip term, in [1, 11]; a format error still scores 0."""
    finish = placement(log, policy_seats)
    score = FAILURE_SCORE if result.format_errors else 1.0 + placement_reward(finish, params)
    return {
        "score": score,
        "reasoning": f"Policy seat {'/'.join(sorted(policy_seats))}: place {finish.place:g} of {finish.seats} with "
        f"{finish.final_chips:g} chips ({finish.chip_share:.0%} of the chips) after {result.hands} hands, "
        f"{result.format_errors} format errors.",
        "artifacts": {
            "reward": "placement",
            "place": finish.place,
            "seats": finish.seats,
            "final_chips": finish.final_chips,
            "chip_share": finish.chip_share,
            "busted": finish.busted,
            "rating_value": finish.rating_value,
            "realized_bb": result.realized_bb,
            "policy_decisions": result.decisions,
            "policy_format_errors": result.format_errors,
            "hands": result.hands,
        },
    }


def bounded(score: dict[str, Any]) -> dict[str, Any]:
    """Drop artifacts, then truncate reasoning, until the reply fits AC2's 1 MiB limit."""
    if len(json.dumps(score)) <= MAX_SCORE_BYTES:
        return score
    trimmed = {"score": score["score"], "reasoning": score.get("reasoning", ""), "artifacts": {"trimmed": True}}
    overflow = len(json.dumps(trimmed)) - MAX_SCORE_BYTES
    if overflow > 0:
        trimmed["reasoning"] = trimmed["reasoning"][: max(0, len(trimmed["reasoning"]) - overflow - 16)]
    return trimmed
