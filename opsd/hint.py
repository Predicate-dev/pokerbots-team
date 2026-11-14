"""The prompted strategy hint that OPSD distills into the weights.

At the event the harness owns the system prompt, so a strategy you want the model to follow has to live in its
weights. This hint is the "tight survival" strategy, written from the arena's first public replays (2026-11-14):
most seats bust, the table is ranked by finishing place, and a bot that folds nearly everything was top-rated. The
opponent types below are the habits measured in those replays; they are described by behavior, not by seat or name,
because the model only ever sees seat numbers.
"""

SEPARATOR = "\n\n"

HINT = """Strategy notes for this game:
- You are ranked by finishing place, not by chips won. Everyone who still has chips at the end ranks above everyone
  who busted, and most players at these tables bust. Never risk your whole stack without a very strong hand. Folding
  is cheap: the blinds are tiny next to the stacks.
- Only the finishing order counts, never the margin. Finishing one chip ahead of a player ranks exactly the same as
  finishing a thousand ahead, so do not make big bets or big calls to win chips you do not need. When you are ahead
  of the players still in, protect the lead. Take a big risk only when you are behind and it is the only way to move
  up a place.
- Before the flop, fold most hands. Play pairs 77 and better, AK, AQ, AJ suited and KQ suited, and open to about three
  big blinds. Fold everything else, including from the blinds against a raise.
- Facing an all-in, or any bet that would commit most of your stack: call only with QQ, KK, AA or AK. Against a
  player who has been moving all-in constantly, also call with TT, JJ and AQ, which are well ahead of that range.
  Otherwise fold, even with a hand that looks decent.
- Read the opponents from their actions and table talk, and adjust:
  - A player who shoves all-in before the flop again and again is doing it with any ace, any pair and worse. Do not
    fight them with medium hands; wait for a strong one and call.
  - A player who calls nearly every bet and never raises will not fold. Never bluff them. Bet big for value with top
    pair or better, and if they ever raise, believe it.
  - A player who folds almost everything will give up their blinds to a raise. If they put chips in, they are strong.
  - Table talk that claims a big hand is unreliable. Decide from the betting.
- After the flop, bet strong made hands for value at about two thirds of the pot. Give up missed hands when an
  opponent bets, and do not chase draws against big bets.
- Once several players have busted and you have a healthy stack, tighten up further: every bust moves you up a
  place, and you gain little from chips you do not need. With a short stack, wait for a strong hand and move all-in
  yourself instead of calling off your chips.
- Keep your reasoning short and decisive, then call submit_action exactly once."""
