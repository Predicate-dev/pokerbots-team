"""The match-level holdout used by arena replay training and decision tests."""

from hashlib import sha1


def held_out(match_id: str) -> bool:
    return int(sha1(match_id.encode()).hexdigest(), 16) % 6 == 0
