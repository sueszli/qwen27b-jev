# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=5.8", "accelerate", "huggingface_hub"]
# ///
from jev import Decision, Jev

DIM, RESET = "\033[2m", "\033[0m"


def show(question: str, d: Decision) -> None:
    print(f"{d.argmax:>7}  " + "  ".join(f"{k} {p:.2f}" for k, p in d.probabilities.items()) + f"  {question}")


STATE = "INCIDENT REPORT INC-4471. A feature flag rollout at 13:31 UTC opened one Postgres connection per request, exhausting the pg-eu-3 limit of 2,000 and causing 30 second gateway timeouts in eu-central-1 only. The flag was reverted at 14:44 and the incident mitigated at 15:41. 212 orders were confirmed without a captured payment because the authorization circuit breaker was misconfigured to fail open; 173 were re-authorized and 39 cancelled. No duplicate charges, no chargebacks, no payment data exposed. Automated alerting fired at 13:44, three minutes after the first customer report."
QUESTIONS = ["Was the root cause a Postgres connection leak?", "Did the outage affect regions outside eu-central-1?", "Were any duplicate charges issued?", "Were all 212 affected orders re-authorized?", "Did automated alerting fire before the first customer report?", "Was the incident mitigated within two hours of the flag rollout?"]

with Jev() as llm:
    question = "Is there evidence that the deployment succeeded?"
    first = llm.decide("The deployment completed at 14:02 UTC. Health checks passed in all three zones.", question, {"yes": "It succeeded.", "no": "It did not.", "unclear": "Cannot tell."})
    show(question, first)
    print(f"{DIM}{first.input_tokens} tokens, {first.seconds:.2f}s{RESET}\n")

    many = llm.decide_many(STATE, [(q, ["yes", "no"]) for q in QUESTIONS])
    for question, d in zip(QUESTIONS, many):
        show(question, d)
    print(f"{DIM}{len(QUESTIONS)} questions over one shared state, {many[0].seconds:.2f}s{RESET}")
