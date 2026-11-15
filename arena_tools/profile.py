"""Per-entry playing profile from arena replay JSON (read-only analysis)."""
import json, glob, sys
from collections import defaultdict, Counter

RANKS = "23456789TJQKA"
def hand_class(cards):
    (a, b) = sorted(cards, key=lambda c: RANKS.index(c[0]), reverse=True)
    return a[0] + b[0] + ("" if a[0] == b[0] else ("s" if a[1] == b[1] else "o"))

P = defaultdict(lambda: Counter())
says = defaultdict(Counter)
allin_hands = defaultdict(list)   # hole cards when the entry put its whole stack in preflop
places = defaultdict(list)
games = 0
for path in sorted(glob.glob(sys.argv[1] + "/*.json")):
    d = json.load(open(path)); games += 1
    name = {s["seat"]: s["entry"]["name"] + ("" if s["entry"]["kind"] == "house" else " [team]") for s in d["seats"]}
    for r in d["results"]:
        n = name[r["seat"]]; places[n].append(r["place"])
        P[n]["games"] += 1; P[n]["busted"] += r["eliminated_on_hand"] is not None
        P[n]["fallbacks"] += r["fallbacks"]; P[n]["decisions"] += r["decisions"]
    for h in d["hands"]:
        hole = {p["seat"]: p["hole_cards"] for p in h["players"]}
        start = {p["seat"]: p["starting_stack"] for p in h["players"]}
        vpip, pfr, shoved, faced_raise, folded_to_raise = set(), set(), set(), set(), set()
        raised = False
        for a in h["actions"]:
            s, n = a["seat"], name[a["seat"]]
            if a.get("say"): says[n][a["say"]] += 1
            if a["street"] == "preflop":
                if raised and a["to_call"] > 0: faced_raise.add(s)
                if a["action"] in ("call", "bet", "raise") and a["to_call"] >= 0 and (a["action"] != "call" or a["to_call"] > 0): vpip.add(s)
                if a["action"] in ("bet", "raise"): pfr.add(s); raised = True
                if a["action"] == "fold" and raised and a["to_call"] > 0: folded_to_raise.add(s)
                if a["action"] in ("call", "bet", "raise") and a["stack_after"] == 0 and s not in shoved:
                    shoved.add(s); allin_hands[n].append(hand_class(hole[s]) + ("(call)" if a["action"] == "call" else ""))
            if a["action"] in ("bet", "raise"): P[n]["aggr"] += 1
            if a["action"] == "call": P[n]["calls"] += 1
            if a["stack_after"] == 0 and a["action"] in ("call", "bet", "raise"): P[n]["allin_actions"] += 1
        for s in hole:
            n = name[s]; P[n]["hands"] += 1
            P[n]["vpip"] += s in vpip; P[n]["pfr"] += s in pfr; P[n]["pf_allin"] += s in shoved
            P[n]["faced_raise"] += s in faced_raise; P[n]["folded_to_raise"] += s in folded_to_raise
print(f"{games} games")
print(f"{'entry':26} {'gms':>3} {'hands':>5} {'VPIP':>5} {'PFR':>5} {'PFallin':>7} {'fold2raise':>10} {'aggr/call':>9} {'bust':>5} {'places'}")
for n, c in sorted(P.items(), key=lambda kv: sum(places[kv[0]]) / len(places[kv[0]])):
    hd = max(1, c["hands"])
    print(f"{n:26} {c['games']:>3} {c['hands']:>5} {100*c['vpip']/hd:>4.0f}% {100*c['pfr']/hd:>4.0f}% {100*c['pf_allin']/hd:>6.0f}% "
          f"{(100*c['folded_to_raise']/c['faced_raise'] if c['faced_raise'] else 0):>9.0f}% {c['aggr']:>4}/{c['calls']:<4} {c['busted']}/{c['games']:<3} {places[n]}")
print()
for n in sorted(allin_hands): print(f"{n:26} preflop all-in with: {' '.join(allin_hands[n][:40])}")
print()
for n in sorted(says): print(f"{n:26} says: " + " | ".join(t for t, _ in says[n].most_common(4)))
