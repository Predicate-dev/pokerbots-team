import json

from sft import build_dataset


def test_sft_variants_keep_the_original_replay_action(monkeypatch) -> None:
    example = {
        "messages": [{"role": "user", "content": "Your hole cards: Qs Js"}],
        "action": {"action": "raise", "amount": 5},
        "hole": ["Qs", "Js"],
        "match": "sample",
        "hand": 3,
        "street": "preflop",
    }
    monkeypatch.setattr(build_dataset, "talk_line", lambda example, rng: "raising with pocket aces.")

    (talk,) = build_dataset.build([example], 1.0, talk=True)
    (plain,) = build_dataset.build([example], 1.0)

    assert json.loads(talk.input[-1].arguments)["say"] == "raising with pocket aces."
    assert json.loads(plain.input[-1].arguments) == {"action": "raise", "amount": 5}
    assert example["action"] == {"action": "raise", "amount": 5}
