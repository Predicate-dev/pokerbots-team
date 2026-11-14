"""Placement and multi-hand EV summaries use games and hands consistently."""

from vals_tools.score import summarize


def test_multihand_ev_is_normalized_and_unscored_tasks_are_visible() -> None:
    result = summarize({
        "a": {"ev_bb": 12.0, "hands": 3, "policy_format_errors": 0},
        "b": {"ev_bb": -2.0, "hands": 1, "policy_format_errors": 0},
        "c": {"failure": True},
    })
    assert result["games"] == 2 and result["hands"] == 4
    assert result["ev_bb_per_hand"] == 2.5 and result["ev_bb_per_100"] == 250.0
    assert result["unscored_tasks"] == 1


def test_placement_pairs_by_task_and_counts_busts() -> None:
    rows = {"a": {"place": 2, "hands": 3, "busted": False, "policy_format_errors": 0}}
    base = {"a": {"place": 4, "hands": 2, "busted": True, "policy_format_errors": 0}, "b": {}}
    result = summarize(rows, base)
    assert result["mean_place"] == 2 and result["bust_rate"] == 0
    assert result["placement_vs_base"] == {"paired_games": 1, "mean_places_gained": 2.0, "ci95": None}
    assert result["pairing"] == {"candidate_only": 0, "base_only": 1}
