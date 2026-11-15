# Arena replay tools

Download every finished arena table's raw JSON and print a per-entry playing profile
(looseness, raising, preflop all-ins, fold-to-raise, busts, places, table talk).

    python3 arena_tools/fetch_replays.py arena_data     # skips games already downloaded
    python3 arena_tools/profile.py arena_data

Standard library only. `arena_data/` is ignored by git.

Use the harness environment to audit a team's folds, split by submission and checkpoint.

    uv run --project harness python arena_tools/fold_audit.py arena_data --org org_3814ae0e9e26

Each fold rate has an opportunity count. The audit uses priced decisions and counts coerced or failed actions
separately. Preflop buckets include unopened button decisions and a small late-position open against the big blind
with two players left in the pot. The narrower buckets overlap with the general buckets. Compare finishing place
and involuntary actions alongside the fold counts.
