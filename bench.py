# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=5.8", "accelerate", "huggingface_hub", "xgrammar"]
# ///
"""reading the answer off the logits against writing it under a json schema, with and without a think trace first.
accuracy is against a hand written key, so it measures the four readouts against each other, not the model against the world."""

from example import QUESTIONS, STATE
from jev import Jev
from json_baseline import decide_json

OPTIONS = ["yes", "no"]
ANSWERS = ["yes", "no", "yes", "no", "no", "yes", "no", "no", "no", "no", "yes", "no", "no", "no", "no", "yes", "no", "no", "yes", "no", "no"]
# the three that are not a quotation: 8 is no because 39 of the 212 orders were cancelled rather than re-authorized,
# 17 is no because ACT-5 was proposed and explicitly not accepted, 19 is yes because 13:58 to 15:41 is 103 minutes.

#
# utils
#
def row(name: str, argmaxes: list[str | None], seconds: list[float], output_tokens: list[int], probabilities: list[float | None]) -> str:
    correct = sum(a == b for a, b in zip(argmaxes, ANSWERS))
    mean_p = [p for p in probabilities if p is not None]
    return f"    {name:<12} {correct:>2}/{len(ANSWERS)}     {sum(seconds) / len(seconds):>6.2f}     {sum(output_tokens) / len(output_tokens):>7.1f}     {f'{sum(mean_p) / len(mean_p):.3f}' if mean_p else '-':>7}"


def disagreements(name: str, argmaxes: list[str | None], other: str, others: list[str | None]) -> None:
    for index, (mine, theirs) in enumerate(zip(argmaxes, others)):
        if mine != theirs:
            print(f"    q{index + 1:<2} {name}={mine} {other}={theirs} key={ANSWERS[index]}  {QUESTIONS[index]}")

#
# benchmark
#
if __name__ == "__main__":
    llm = Jev()
    assert len(ANSWERS) == len(QUESTIONS), "the answer key does not cover every question"
    results: dict[str, tuple[list[str | None], list[float], list[int], list[float | None]]] = {}

    for think in (False, True):
        decisions = [llm.decide(STATE, question, OPTIONS, think=think) for question in QUESTIONS]
        results[f"jev{'+think' if think else ''}"] = ([d.argmax for d in decisions], [d.seconds for d in decisions], [d.output_tokens for d in decisions], [d.probabilities[a] for d, a in zip(decisions, ANSWERS)])
        for question, key, decision in zip(QUESTIONS, ANSWERS, decisions):
            print(f"jev  think={think:<1} {decision.argmax:>3} p(key)={decision.probabilities[key]:.3f} {decision.seconds:6.2f}s out={decision.output_tokens:4d} {question}")

    for think in (False, True):
        written = [decide_json(llm, STATE, question, OPTIONS, think=think) for question in QUESTIONS]
        results[f"json{'+think' if think else ''}"] = ([w.answer for w in written], [w.seconds for w in written], [w.output_tokens for w in written], [None] * len(written))
        for question, w in zip(QUESTIONS, written):
            print(f"json think={think:<1} {w.answer!s:>3} {w.raw:<18} {w.seconds:6.2f}s out={w.output_tokens:4d} {question}")

    print()
    print("    condition    correct   sec/q    out tok/q    p(key)")
    for name, (argmaxes, seconds, output_tokens, probabilities) in results.items():
        print(row(name, argmaxes, seconds, output_tokens, probabilities))
    print()
    for name in ("jev+think", "json", "json+think"):
        disagreements(name, results[name][0], "jev", results["jev"][0])
    print(f"    {sum(a is None for a in results['json'][0]) + sum(a is None for a in results['json+think'][0])} unparsable json answers of {2 * len(QUESTIONS)}")
