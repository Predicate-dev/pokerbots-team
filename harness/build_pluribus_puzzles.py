"""Build a second puzzle bank from the six-max Pluribus hands in uoftcprg/phh-dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import replace
from math import ceil, floor
from pathlib import Path

from pokerkit import CheckingOrCalling, CompletionBettingOrRaisingTo, Folding, HandHistory, State

from poker_harness.completions import EVENT_PARAMS, table_rules
from poker_harness.engine.table import STREETS, DecisionContext, SeatView, codes, legal_actions
from poker_harness.prompts import decision_prompt, scratchpad_block, system_prompt
from puzzles import STACK_SHARE, equity

RULES = table_rules(EVENT_PARAMS)
CHIP_SCALE = 50
DECISIONS = (CheckingOrCalling, CompletionBettingOrRaisingTo, Folding)


def chips(amount: int) -> int | float:
    whole, remainder = divmod(amount, CHIP_SCALE)
    return amount / CHIP_SCALE if remainder else whole


def puzzle_at(state: State, actions: list[dict], players: list[str], hand_id: str) -> dict | None:
    seat = state.actor_index
    if seat is None or state.street_index is None:
        return None
    stack = state.stacks[seat]
    call = state.checking_or_calling_amount or 0
    street = STREETS[state.street_index]
    if call == 0 or (call < STACK_SHARE * stack and street != "river"):
        return None

    hole = codes(state.hole_cards[seat])
    board = codes(state.get_board_cards(0))
    opponents = [codes(state.hole_cards[i]) for i in state.player_indices if i != seat and state.statuses[i]]
    pot = state.total_pot_amount
    price = call / (pot + call)
    puzzle_id = f"pluribus:{hand_id}:{seat}:{len(actions)}"
    share = equity(hole, opponents, board, puzzle_id, dead=codes(state.mucked_cards))
    ev = share * (pot + call) - call

    # The game state keeps exact scaled chips. The arena submit_action tool takes whole numbers for raises.
    legal = legal_actions(state)
    low = ceil(legal.min_raise_to / CHIP_SCALE)
    high = floor(legal.max_raise_to / CHIP_SCALE)
    legal = replace(
        legal,
        call_amount=chips(call),
        min_raise_to=low,
        max_raise_to=high,
        raise_type=legal.raise_type if low <= high else None,
    )
    views = [
        SeatView(
            i,
            f"seat{i}",
            chips(state.stacks[i]),
            chips(state.bets[i]),
            "folded" if not state.statuses[i] else "all-in" if state.stacks[i] == 0 else "",
        )
        for i in state.player_indices
    ]
    context = DecisionContext(
        hand_number=1,
        total_hands=RULES.total_hands,
        street=street,
        seat=seat,
        position=seat,
        button_seat=RULES.seats - 1,
        hole_cards=hole,
        board=board,
        pot=chips(pot),
        seats=views,
        hand_actions=actions,
        legal=legal,
    )
    return {
        "id": puzzle_id,
        "entry": players[seat],
        "deployment": None,
        "ours": False,
        "kind": "stack_call" if call >= STACK_SHARE * stack else "river_bet",
        "street": street,
        "hole": hole,
        "board": board,
        "to_call": chips(call),
        "pot": chips(pot),
        "best": "continue" if share >= price else "fold",
        "equity": round(share, 3),
        "price": round(price, 3),
        "margin": 0.0,
        "regret_if_wrong": round(abs(ev) / CHIP_SCALE, 1),
        "players_left": RULES.seats,
        "villains": len(opponents),
        "closes_action": len(state.actor_indices) == 1,
        "messages": [
            {"role": "system", "content": system_prompt(RULES)},
            {"role": "user", "content": scratchpad_block("", RULES.scratchpad_chars)},
            {"role": "user", "content": decision_prompt(context, RULES.table_talk)},
        ],
    }


def hand_puzzles(hand: HandHistory, hand_id: str, entry: str | None = None) -> Iterator[dict]:
    if (
        hand.variant != "NT"
        or list(hand.starting_stacks) != [RULES.starting_stack * CHIP_SCALE] * RULES.seats
        or list(hand.blinds_or_straddles) != [RULES.small_blind * CHIP_SCALE, RULES.big_blind * CHIP_SCALE, 0, 0, 0, 0]
        or any(hand.antes)
        or not hand.players
        or len(hand.players) != RULES.seats
    ):
        raise ValueError(f"{hand_id}: expected six-max Pluribus format with 50/100 blinds and 10,000-chip stacks")

    actions: list[dict] = []
    pending = None
    operation_count = 0
    previous_actor, previous_street, previous_bet = None, None, 0
    for state, action in hand.state_actions:
        for operation in state.operations[operation_count:]:
            if not isinstance(operation, DECISIONS):
                continue
            seat = operation.player_index
            if action is None or seat != previous_actor or previous_street is None:
                raise ValueError(f"{hand_id}: could not match a recorded action to its preceding state")
            if isinstance(operation, Folding):
                move = {"type": "fold"}
            elif isinstance(operation, CheckingOrCalling):
                move = {"type": "call" if operation.amount else "check"}
            else:
                kind = "raise" if previous_street == 0 or previous_bet else "bet"
                move = {"type": kind, "amount": chips(operation.amount)}
            if pending is not None:
                took = "fold" if move["type"] == "fold" else "continue"
                yield pending | {"took": took, "recorded_action": move["type"]}
            actions.append({"player": f"seat{seat}", "seat": seat, "street": STREETS[previous_street], "action": move})
        operation_count = len(state.operations)
        previous_street = state.street_index
        previous_bet = max(state.bets)
        previous_actor = state.actor_index
        pending = (
            puzzle_at(state, actions, hand.players, hand_id)
            if previous_actor is not None and (entry is None or hand.players[previous_actor] == entry)
            else None
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="path to phh-dataset/data/pluribus")
    parser.add_argument("--out", type=Path, default=Path("runs/pluribus_puzzles.jsonl"))
    parser.add_argument("--entry", help="choose one reference player, e.g. Pluribus (default: all seats)")
    args = parser.parse_args()
    paths = sorted(args.source.rglob("*.phh"))
    if not paths:
        parser.error(f"no .phh hands found under {args.source}")
    rows = []
    for path in paths:
        hand_id = path.relative_to(args.source).with_suffix("").as_posix()
        with path.open("rb") as file:
            hand = HandHistory.load(file)
        rows.extend(hand_puzzles(hand, hand_id, args.entry))
    if not rows:
        parser.error("no matching decisions found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    counts = Counter(row["kind"] for row in rows)
    print(f"{len(paths)} hands, {len(rows)} puzzles -> {args.out}")
    print(f"{counts['stack_call']} stack calls, {counts['river_bet']} other river decisions")
    print(f"{sum(row['entry'] == 'Pluribus' for row in rows)} decisions played by Pluribus")


if __name__ == "__main__":
    main()
