# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub"]
# ///
import statistics
import time
from pathlib import Path

from jev import Jev
from plain import Plain

STATE = "INCIDENT REPORT INC-4471. A feature flag rollout at 13:31 UTC opened one Postgres connection per request, exhausting the pg-eu-3 limit of 2,000 and causing 30 second gateway timeouts in eu-central-1 only. The flag was reverted at 14:44 and the incident mitigated at 15:41. 212 orders were confirmed without a captured payment because the authorization circuit breaker was misconfigured to fail open; 173 were re-authorized and 39 cancelled. No duplicate charges, no chargebacks, no payment data exposed. Automated alerting fired at 13:44, three minutes after the first customer report."
FACTS = [("Was the root cause a Postgres connection leak?", "yes"), ("Did the outage affect regions outside eu-central-1?", "no"), ("Were any duplicate charges issued?", "no"), ("Were all 212 affected orders re-authorized?", "no"), ("Did automated alerting fire before the first customer report?", "no"), ("Was the feature flag reverted?", "yes"), ("Did the gateway time out after 30 seconds?", "yes"), ("Was the Postgres connection limit 2,000?", "yes"), ("Was any customer payment data exposed?", "no"), ("Were any chargebacks filed?", "no"), ("Did the authorization circuit breaker fail open?", "yes"), ("Were 39 orders cancelled?", "yes"), ("Is the incident identified as INC-4471?", "yes"), ("Did the rollout open one Postgres connection per request?", "yes"), ("Was the saturated database a MySQL cluster?", "no"), ("Were orders confirmed without a captured payment?", "yes"), ("Did the incident start with a feature flag rollout?", "yes"), ("Was the flag reverted before the incident was mitigated?", "yes"), ("Did a customer report the problem before automated alerting fired?", "yes"), ("Was the pg-eu-3 connection limit exhausted?", "yes"), ("Were more than 200 orders confirmed without a captured payment?", "yes"), ("Does the report mention a rolled back database migration?", "no"), ("Was the authorization circuit breaker correctly configured?", "no"), ("Did the incident begin before 14:00 UTC?", "yes")]
HARD = [("Was the incident mitigated within two hours of the flag rollout?", "no"), ("Did more than an hour pass between the flag rollout and the revert?", "yes"), ("Were fewer than 40 of the affected orders cancelled?", "yes"), ("Did reverting the flag resolve the incident immediately?", "no"), ("Is it true that no affected order was left without a captured payment?", "no"), ("Did the first customer report arrive at or before 13:41 UTC?", "yes")]
TRICKY = [("Were exactly 39 orders cancelled?", "yes"), ("Were fewer than 170 orders re-authorized?", "no"), ("Did the incident last more than two hours in total?", "yes"), ("Was the flag reverted more than one hour after the first alert?", "no"), ("Did the outage begin more than ten minutes before the first alert?", "yes"), ("Were at least 80 percent of the affected orders re-authorized?", "yes"), ("Did the incident affect us-east-1?", "no"), ("Did the incident occur on a weekend?", "no"), ("Does the report state the number of failed checkout attempts?", "no"), ("Does the report say who reverted the flag?", "no"), ("Were the 39 cancelled orders refunded?", "no"), ("Did the connection limit exhaust because of a slow query?", "no"), ("Did the circuit breaker cause the gateway timeouts?", "no"), ("Did the gateway timeouts cause the uncaptured payments?", "no"), ("Would the uncaptured payments have happened if the breaker had failed closed?", "no"), ("Did the incident take longer to mitigate than to detect?", "yes"), ("Was the incident detected within fifteen minutes of the rollout?", "yes"), ("Were more orders re-authorized than cancelled?", "yes"), ("Did the revert happen in the same hour as the rollout?", "no"), ("Does the report mention Postgres?", "yes")]  # counting, relative timing, absent information, causal chains
QUESTIONS = FACTS + HARD + TRICKY
SELECTS = [("Which of these are stated consequences of the incident?", {"timeouts": "Gateway timeouts in eu-central-1", "chargebacks": "Chargebacks were filed", "uncaptured": "Orders confirmed without a captured payment", "data_leak": "Customer payment data was exposed"}, {"timeouts", "uncaptured"}), ("Which of these are stated facts about the response?", {"reverted": "The feature flag was reverted", "migration": "A database migration was rolled back", "reauth": "Some affected orders were re-authorized", "refund": "Every affected customer was refunded"}, {"reverted", "reauth"})]


def run(llm: Jev, think: bool) -> tuple[list[tuple[str, bool, float]], float]:
    # answer every item in one mode, return one (pick, correct, probability of the expected answer) per item plus the wall time
    started = time.perf_counter()
    graded = list(zip(llm.decide_many(STATE, [(question, ["yes", "no"]) for question, _ in QUESTIONS], think=think), [expected for _, expected in QUESTIONS]))
    graded += [(decision, "yes" if option in expected else "no") for question, options, expected in SELECTS for option, decision in llm.select(STATE, question, options, think=think).items()]
    return [(d.argmax, d.argmax == e, d.probabilities[e]) for d, e in graded], time.perf_counter() - started


with Jev() as llm:
    plain = Plain.__new__(Plain)
    plain.__dict__ = llm.__dict__  # both read the one running server, a second one would not fit on the card
    llm.decide(STATE, QUESTIONS[0][0], ["yes", "no"])  # the first request compiles llama.cpp's graphs, keep that out of the timings
    results = {(name, think): run(model, think) for name, model in (("jev", llm), ("plain", plain)) for think in (False, True)}
    lines = [f"{'mode':<12}{'accuracy':>10}{'p(expected)':>13}{'s/item':>9}{'total s':>10}", "-" * 54]
    for (name, think), (graded, seconds) in results.items():
        odds = f"{statistics.fmean(p for _, _, p in graded):.3f}" if name == "jev" else "-"  # plain only ever writes one letter, so its odds are its accuracy
        lines.append(f"{name + (' think' if think else ''):<12}{sum(c for _, c, _ in graded) / len(graded):>9.1%}{odds:>13}{seconds / len(graded):>9.2f}{seconds:>10.1f}")
    lines += ["-" * 54] + [f"jev and plain pick the same answer on {statistics.fmean(a[0] == b[0] for a, b in zip(results[('jev', think)][0], results[('plain', think)][0])):.1%} of items, think={think}" for think in (False, True)]
    table = "\n".join(lines)
    print(table)
    Path(__file__).resolve().parent.joinpath("bench.txt").write_text(f"{table}\n")
