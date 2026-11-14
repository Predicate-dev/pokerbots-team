"""Scripted imitations of the arena's six house bots, for training data and local scoring without model calls.

These are approximations, calibrated to statistics mined from the arena's first public replays (2026-11-14, six
30-hand tables). They are not the real house bots, which are models with personas; they copy the measured habits:

    style    looseness / raising    what stood out in the replays
    alice    3% / 0%                folds almost everything, even to small raises; survives to 2nd place
    bob      42% / 42%              shoves all-in preflop with any ace, pair, broadway and some junk; never calls
    stan     38% / 1%               never raises, calls nearly anything, including all-ins with weak hands
    carlos   45% / 33%              loose and aggressive; table talk is unreliable
    pete     35% / 25%              tight-aggressive, standard sizes
    tim      36% / 25%              like pete, but folds to raises more often

Each style also says a line in character, as the real bots do, so a model can learn to tell seats apart by talk.
Used through `env_params.opponent_seats` as `"scripted:house_<style>"` (see `training.py`).
"""

from __future__ import annotations

import random

from .engine.table import Decision, DecisionContext
from .spots import RANKS, chen_score

HOUSE_STYLES = ("house_alice", "house_bob", "house_stan", "house_carlos", "house_pete", "house_tim")

TALK = {
    "alice": {"fold": ["oh no, sorry, i fold", "please do not raise me"], "play": ["oh no, i have to play this"]},
    "bob": {"fold": ["not this time"], "play": ["all in, baby"]},
    "stan": {"fold": ["not buying this one", "not this time"], "play": ["i call, show me", "you are bluffing, i can tell"]},
    "carlos": {"fold": ["ugh, i probably should fold this"], "play": ["i flopped the nuts, save your chips"]},
    "pete": {"fold": ["the equity threshold is not met."], "play": ["a textbook opening size."]},
    "tim": {"fold": ["good luck, everyone"], "play": ["nice raise", "good luck, everyone"]},
}


def _connects(hole_cards: list[str], board: list[str]) -> bool:
    """A pocket pair, or a hole card pairing the board: the crude "I have something" test these styles use."""
    ranks = [c[0] for c in hole_cards]
    return ranks[0] == ranks[1] or any(r in {c[0] for c in board} for r in ranks)


def _bob_range(hole_cards: list[str]) -> bool:
    """Any ace, any pair, or two cards ten or higher."""
    ranks = [c[0] for c in hole_cards]
    return "A" in ranks or ranks[0] == ranks[1] or all(RANKS.index(r) >= RANKS.index("T") for r in ranks)


class HouseStylePlayer:
    """One scripted house-bot imitation. Deterministic for a given (name, seed)."""

    def __init__(self, name: str, style: str, seed: int = 0) -> None:
        if style not in HOUSE_STYLES:
            raise ValueError(f"unknown house style {style!r}")
        self.name = name
        self.style = style.removeprefix("house_")
        self.rng = random.Random(f"{name}:{style}:{seed}")

    # -- helpers -----------------------------------------------------------------------------------------------
    def _passive(self, ctx: DecisionContext) -> Decision:
        return Decision("check" if ctx.legal.can_check else "call")

    def _give_up(self, ctx: DecisionContext) -> Decision:
        return Decision("check" if ctx.legal.can_check else "fold")

    def _raise_to(self, ctx: DecisionContext, target: int) -> Decision:
        legal = ctx.legal
        if not legal.raise_type:
            return self._passive(ctx)
        return Decision(legal.raise_type, max(legal.min_raise_to, min(legal.max_raise_to, target)))

    def _shove(self, ctx: DecisionContext) -> Decision:
        return self._raise_to(ctx, ctx.legal.max_raise_to)

    # -- styles ------------------------------------------------------------------------------------------------
    def _choose(self, ctx: DecisionContext) -> Decision:
        legal, score, pre = ctx.legal, chen_score(ctx.hole_cards), ctx.street == "preflop"
        big_call = legal.call_amount > 10  # facing more than a normal open
        roll = self.rng.random()
        if self.style == "alice":
            if pre and score >= 12:  # AA, KK, QQ, AK suited
                return self._passive(ctx)
            if not pre and score >= 12 and _connects(ctx.hole_cards, ctx.board):
                return self._passive(ctx)
            return self._give_up(ctx)
        if self.style == "bob":
            if pre:
                if _bob_range(ctx.hole_cards) or roll < 0.15:
                    return self._shove(ctx)
                return self._give_up(ctx)
            return self._shove(ctx) if roll < 0.5 else self._give_up(ctx)
        if self.style == "stan":
            if pre and score < 4 and legal.call_amount > 0:
                return Decision("fold") if legal.can_fold else self._passive(ctx)
            return self._passive(ctx)
        # carlos, pete and tim: open-raise a range, continue against raises with a tighter one
        opens, continues, fold_to_bet = {"carlos": (4, 7, 0.45), "pete": (6, 8, 0.6), "tim": (6, 9, 0.7)}[self.style]
        if pre:
            if big_call:
                return self._passive(ctx) if score >= continues + 1 else self._give_up(ctx)
            if score >= opens and legal.raise_type and legal.call_amount <= 2:
                return self._raise_to(ctx, 6)
            if score >= continues or (score >= opens and legal.call_amount <= 6):
                return self._passive(ctx)
            return self._give_up(ctx)
        hit = _connects(ctx.hole_cards, ctx.board)
        if legal.call_amount > 0:  # facing a bet
            if hit and (score >= continues or roll > fold_to_bet):
                return self._passive(ctx)
            if self.style == "carlos" and roll < 0.2:
                return self._passive(ctx)
            return self._give_up(ctx)
        bluff = {"carlos": 0.45, "pete": 0.25, "tim": 0.2}[self.style]
        if legal.raise_type and (hit or roll < bluff):
            return self._raise_to(ctx, max(legal.min_raise_to, (ctx.pot * 2) // 3))
        return self._passive(ctx)

    async def decide(self, ctx: DecisionContext, history: str) -> Decision:
        decision = self._choose(ctx)
        lines = TALK[self.style]["fold" if decision.type in ("fold", "check") else "play"]
        say = self.rng.choice(lines) if self.rng.random() < 0.6 else ""
        decision.model_response = {
            "model_name": self.name,
            "provider": "scripted",
            "latency_seconds": 0.0,
            "in_tokens": 0,
            "out_tokens": 0,
            "reasoning_tokens": 0,
            "cost": 0.0,
            "reasoning": "",
            "fallback_used": False,
            "error": None,
            "say": say,
        }
        return decision
