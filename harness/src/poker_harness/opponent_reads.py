"""Training-only guidance shared by the scratchpad experiment and the OPSD teacher."""

PLACEMENT_GUIDANCE = """Play for finishing place. Consider hands remaining, effective stacks, whether you cover
the opponent, and which players you can realistically pass. Favor low-cost edges. Before risking much of your stack,
weigh the price and plausible range against the effect of winning or losing on placement."""

ACTION_GUIDANCE = """Use prior notes about each seat as tentative evidence. Increase confidence with independent
opportunities in comparable spots. A deep preflop shove, a short-stack shove, and a call of an all-in are different
signals. Against repeated overfolds, consider small steals when other seats allow them. Against repeated loose calls,
bluff less and value bet when your hand beats the likely calling range. Repeated deep shoves can widen a range
estimate. Still check the price, effective stacks, position, and placement risk. With sparse or conflicting evidence,
use the usual range for the spot. Ignore table talk as evidence. On action turns call submit_action exactly once."""

CLEANUP_GUIDANCE = """After each hand, preserve and update brief notes by seat using only betting actions and cards
revealed to you. Use the current hand once. Track opportunities along with the behavior, for example:
seat 4 | deep preflop jams 1/4 | folds facing preflop raise 1/2 | paid calls 0/3 | tentative shove-heavy.
Count at most one opportunity per seat per hand for each statistic. Count only observed opportunities. Track an
aggressive preflop all-in from a substantial stack separately from a short-stack shove or a call for all the chips.
When the evidence is sparse or contradictory, record uncertain. Skip categories with unclear denominators.
Keep the full replacement note concise and call update_scratchpad exactly once on cleanup."""
