# Arena replay tools

Download every finished arena table's raw JSON and print a per-entry playing profile
(looseness, raising, preflop all-ins, fold-to-raise, busts, places, table talk).

    python3 arena_tools/fetch_replays.py arena_data     # skips games already downloaded
    python3 arena_tools/profile.py arena_data

Standard library only. `arena_data/` is ignored by git.
