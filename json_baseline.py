# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=5.8", "accelerate", "huggingface_hub", "xgrammar"]
# ///
"""the ordinary way to do this: make the model write the answer, and use a json schema to keep it well formed.
jev.py reads the answer off the logits instead. run bench.py to see what the writing costs."""

import json
import time
from dataclasses import dataclass

import torch
import xgrammar
from xgrammar.contrib.hf import LogitsProcessor  # no hand rolled trie needed

from jev import LETTERS, SYSTEM, Jev, encode_decision

JSON_SYSTEM = SYSTEM.replace("Respond with only its uppercase letter, with no explanation or reasoning.", 'Respond with only the json object {"answer": "<uppercase letter>"}, with no explanation or reasoning.')
GRAMMARS: dict[int, xgrammar.CompiledGrammar] = {}  # compiling a schema takes about a second, and only the option count varies

#
# setup
#
def answer_grammar(llm: Jev, count: int) -> xgrammar.CompiledGrammar:
    if count not in GRAMMARS:
        schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": list(LETTERS[:count])}}, "required": ["answer"], "additionalProperties": False}
        info = xgrammar.TokenizerInfo.from_huggingface(llm.tokenizer, vocab_size=llm.model.config.vocab_size)  # the model's vocab, not the tokenizer's: a short bitmask leaves the last few hundred logits unmasked
        GRAMMARS[count] = xgrammar.GrammarCompiler(info).compile_json_schema(json.dumps(schema))
    return GRAMMARS[count]

#
# inference
#
def generate_answer(llm: Jev, ids: list[int], grammar: xgrammar.CompiledGrammar, max_new_tokens: int) -> list[int]:
    input_ids = torch.tensor([ids], dtype=torch.long, device=next(llm.model.parameters()).device)
    processor = LogitsProcessor(grammar)  # one matcher per call, it keeps the position it reached
    return llm.model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=llm.pad, logits_processor=[processor])[0, len(ids) :].tolist()


def parse_answer(raw: str, pairs: list[tuple[str, str]]) -> str | None:
    try:
        letter = json.loads(raw)["answer"]
    except (json.JSONDecodeError, KeyError):  # only reachable when generation hits max_json_tokens mid object, the grammar rules out everything else
        return None
    assert letter in LETTERS[: len(pairs)], f"the grammar allowed the out of range letter {letter!r}"
    return pairs[LETTERS.index(letter)][0]


@dataclass
class JsonDecision:
    answer: str | None  # option id, None only if the json was cut off
    raw: str  # what the model wrote, without special tokens
    input_tokens: int  # the prompt, never the think trace
    output_tokens: int  # think trace plus json, everything the model had to write
    seconds: float  # wall time of the whole call


def decide_json(llm: Jev, state: str | dict | list, question: str, options: list[str] | dict[str, str], think: bool = False, max_json_tokens: int = 24) -> JsonDecision:
    started = time.perf_counter()
    pairs, _, ids = encode_decision(llm.tokenizer, llm.slots, state, question, options, think, JSON_SYSTEM)
    prompt_tokens, trace = len(ids), []
    with torch.inference_mode():
        if think:
            trace = llm.trace(ids)  # unconstrained until </think>, then the grammar takes over
            ids = ids + trace + llm.tail
        generated = generate_answer(llm, ids, answer_grammar(llm, len(pairs)), max_json_tokens)
    raw = llm.tokenizer.decode(generated, skip_special_tokens=True)
    return JsonDecision(parse_answer(raw, pairs), raw, prompt_tokens, len(trace) + len(generated), time.perf_counter() - started)


if __name__ == "__main__":
    DIM, RESET = "\033[2m", "\033[0m"
    llm = Jev()
    for think in (False, True):
        written = decide_json(llm, "The deployment completed at 14:02 UTC. Health checks passed in all three zones.", "Is there evidence that the deployment succeeded?", {"yes": "It succeeded.", "no": "It did not.", "unclear": "Cannot tell."}, think)
        print(f"answer: {written.answer}")
        print(f"raw: {written.raw}")
        print(f"{DIM}think={think}, {written.input_tokens} input tokens, {written.output_tokens} output tokens, {written.seconds:.2f}s{RESET}")
