"""Training-only table setups, all opt-in through `env_params`. The arena never sets any of them.

A task that sets none of these keys plays exactly as before; `tests/test_training.py` checks that, and the engine
parity and official-rules tests still pin the official game. Every key is validated before the table starts, and a
bad value fails the rollout with a message that names the key.

Mid-game starts (`table_start`)
-------------------------------
    initial_stacks      list of `seats` integers, one per seat; 0 means that seat already busted. Carry-over games only
                        (`reset_stacks` false). At least two seats, and every policy seat, must have chips.
    first_hand          the hand number the task starts on (1..total_hands). Prompts show the true number, so a task
                        with `first_hand: 28` at a 30-hand table opens with "## Hand 28 of 30".
    last_hand           the last hand number played (first_hand..total_hands); default: the game's last hand.
    button_seat         the seat with the button on the first hand played. Default: where the button would be on
                        that hand if nobody had busted (seat `seats - 1` on hand 1, one seat on per hand), moved on to
                        the next seat with chips.
    initial_scratchpad  the scratchpad the task starts with: a string for every policy seat, or {seat: string}.
                        Scratchpad tables only, not for scripted seats; at most `scratchpad_chars` characters.

Per-seat opponents (`opponent_specs`)
-------------------------------------
    opponent_seats      {seat: spec} for seats that are not policy seats. A spec is
                        - "scripted:<archetype>": a scripted player from `engine/scripted.py` (random, maniac, nit,
                          calling_station). No model call.
                        - an object with any of `model` (default: the task's `opponent_model`), `persona` (text
                          appended to that seat's system prompt, to hold it to a style), `api` ("chat" or
                          "responses"), `reasoning_effort` and `thinking`.
                        Seats it doesn't list are played as before: by `opponent_model`, or by the policy when the
                        task names no opponent model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .engine.scripted import ARCHETYPES as ENGINE_ARCHETYPES
from .engine.scripted import ScriptedPlayer
from .house_bots import HOUSE_STYLES, HouseStylePlayer
from .prompts import TableRules

START_KEYS = ("initial_stacks", "first_hand", "last_hand", "button_seat", "initial_scratchpad")
SCRIPTED = "scripted:"
# Every scripted style a seat can name: the engine's four archetypes and the house-bot imitations (`house_bots.py`).
ARCHETYPES = (*ENGINE_ARCHETYPES, *HOUSE_STYLES)


def scripted_player(name: str, spec: str, seed: int, *, talk_mode: str = "normal") -> ScriptedPlayer | HouseStylePlayer:
    """The scripted seat a checked `"scripted:<style>"` spec names."""
    style = spec.removeprefix(SCRIPTED)
    if style in HOUSE_STYLES:
        return HouseStylePlayer(name, style, seed, talk_mode=talk_mode)
    return ScriptedPlayer(name, style, seed)
SPEC_KEYS = frozenset({"model", "persona", "api", "reasoning_effort", "thinking"})
MAX_PERSONA_CHARS = 2000


class TrainingSetupError(ValueError):
    """A training-only `env_params` key has a value the table can't use."""


@dataclass
class TableStart:
    """Where a table starts. The defaults are the official start: equal stacks, hand 1, an empty scratchpad."""

    initial_stacks: dict[int, int] | None = None
    first_hand: int = 1
    last_hand: int | None = None
    button_seat: int | None = None
    scratchpads: dict[int, str] = field(default_factory=dict)


def _integer(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingSetupError(f"env_params.{key} must be an integer, got {value!r}")
    return value


def _seat(value: Any, key: str, seats: int) -> int:
    try:
        seat = int(value) if isinstance(value, str) else _integer(value, key)
    except ValueError as exc:
        raise TrainingSetupError(f"env_params.{key}: {value!r} is not a seat number") from exc
    if not 0 <= seat < seats:
        raise TrainingSetupError(f"env_params.{key}: seat {seat} is outside 0-{seats - 1}")
    return seat


def table_start(params: dict[str, Any], rules: TableRules, policy: set[int]) -> TableStart:
    """The task's start, checked. Without any of `START_KEYS` this is `TableStart()`, the official start."""
    if not any(key in params for key in START_KEYS):
        return TableStart()
    seats = rules.seats
    start = TableStart()

    if "initial_stacks" in params:
        raw = params["initial_stacks"]
        if rules.reset_stacks:
            raise TrainingSetupError("env_params.initial_stacks needs a carry-over game (reset_stacks false)")
        if not isinstance(raw, list) or len(raw) != seats:
            raise TrainingSetupError(f"env_params.initial_stacks must list one stack per seat ({seats})")
        stacks = {seat: _integer(chips, "initial_stacks") for seat, chips in enumerate(raw)}
        if any(chips < 0 for chips in stacks.values()):
            raise TrainingSetupError("env_params.initial_stacks can't be negative")
        if sum(chips > 0 for chips in stacks.values()) < 2:
            raise TrainingSetupError("env_params.initial_stacks needs at least two seats with chips")
        busted_policy = sorted(seat for seat in policy if stacks.get(seat, 0) == 0)
        if busted_policy:
            raise TrainingSetupError(f"env_params.initial_stacks gives policy seats {busted_policy} no chips")
        start.initial_stacks = stacks

    if "first_hand" in params:
        start.first_hand = _integer(params["first_hand"], "first_hand")
        if not 1 <= start.first_hand <= rules.total_hands:
            raise TrainingSetupError(f"env_params.first_hand must be 1-{rules.total_hands}, got {start.first_hand}")
    if "last_hand" in params:
        start.last_hand = _integer(params["last_hand"], "last_hand")
        if not start.first_hand <= start.last_hand <= rules.total_hands:
            raise TrainingSetupError(
                f"env_params.last_hand must be {start.first_hand}-{rules.total_hands}, got {start.last_hand}"
            )

    live = sorted(s for s in range(seats) if start.initial_stacks is None or start.initial_stacks[s] > 0)
    if "button_seat" in params:
        start.button_seat = _seat(params["button_seat"], "button_seat", seats)
        if start.button_seat not in live:
            raise TrainingSetupError(f"env_params.button_seat {start.button_seat} has no chips")
    elif start.first_hand > 1 or start.initial_stacks is not None:
        # Where the button would be on this hand had nobody busted, moved on to the next seat with chips.
        natural = (seats - 1 + start.first_hand - 1) % seats
        start.button_seat = next(s for s in [*range(natural, seats), *range(natural)] if s in live)

    if "initial_scratchpad" in params:
        raw = params["initial_scratchpad"]
        if not rules.scratchpad:
            raise TrainingSetupError("env_params.initial_scratchpad needs a scratchpad table")
        if isinstance(raw, str):
            pads = dict.fromkeys(sorted(policy), raw)
        elif isinstance(raw, dict):
            pads = {_seat(seat, "initial_scratchpad", seats): text for seat, text in raw.items()}
        else:
            raise TrainingSetupError("env_params.initial_scratchpad must be a string or {seat: string}")
        for seat, text in pads.items():
            if not isinstance(text, str):
                raise TrainingSetupError(f"env_params.initial_scratchpad for seat {seat} is not a string")
            if len(text) > rules.scratchpad_chars:
                raise TrainingSetupError(
                    f"env_params.initial_scratchpad for seat {seat} has {len(text)} characters; "
                    f"the limit is {rules.scratchpad_chars}"
                )
        start.scratchpads = pads
    return start


def opponent_specs(params: dict[str, Any], seats: int, policy: set[int]) -> dict[int, str | dict[str, Any]]:
    """`env_params.opponent_seats`, checked: seat -> "scripted:<archetype>" or a model spec. Empty without the key."""
    raw = params.get("opponent_seats")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TrainingSetupError("env_params.opponent_seats must be {seat: spec}")
    specs: dict[int, str | dict[str, Any]] = {}
    for key, spec in raw.items():
        seat = _seat(key, "opponent_seats", seats)
        if seat in policy:
            raise TrainingSetupError(f"env_params.opponent_seats: seat {seat} is a policy seat")
        if isinstance(spec, str):
            archetype = spec.removeprefix(SCRIPTED)
            if not spec.startswith(SCRIPTED) or archetype not in ARCHETYPES:
                known = ", ".join(f"{SCRIPTED}{name}" for name in ARCHETYPES)
                raise TrainingSetupError(f"env_params.opponent_seats[{seat}]: {spec!r} is not one of {known}")
        elif isinstance(spec, dict):
            unknown = set(spec) - SPEC_KEYS
            if unknown:
                raise TrainingSetupError(f"env_params.opponent_seats[{seat}]: unknown keys {sorted(unknown)}")
            if not (spec.get("model") or params.get("opponent_model")):
                raise TrainingSetupError(f"env_params.opponent_seats[{seat}] names no model and the task has none")
            if spec.get("model") is not None and not isinstance(spec["model"], str):
                raise TrainingSetupError(f"env_params.opponent_seats[{seat}].model must be a string")
            persona = spec.get("persona", "")
            if not isinstance(persona, str) or len(persona) > MAX_PERSONA_CHARS:
                raise TrainingSetupError(
                    f"env_params.opponent_seats[{seat}].persona must be a string of at most {MAX_PERSONA_CHARS} "
                    "characters"
                )
            if spec.get("api", "chat") not in ("chat", "responses"):
                raise TrainingSetupError(f"env_params.opponent_seats[{seat}].api must be chat or responses")
        else:
            raise TrainingSetupError(f"env_params.opponent_seats[{seat}] must be a string or an object")
        specs[seat] = spec
    return specs


def spec_params(params: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """The task's params with one model spec's overrides, in the keys `completions.opponent_params` reads."""
    out = dict(params)
    if spec.get("model"):
        out["opponent_model"] = spec["model"]
    if "api" in spec:
        out["opponent_api"] = spec["api"]
    if "reasoning_effort" in spec:
        out["opponent_reasoning_effort"] = spec["reasoning_effort"]
    if "thinking" in spec:
        out["opponent_thinking"] = spec["thinking"]
    return out
