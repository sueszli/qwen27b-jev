# Plan

OpenJev (https://github.com/TheoLeeCJ/openjev) on Qwen3.8-27B. Three files:

    jev.py       the library, a uv script with inline deps, ~120 loc
    example.py   a demo, like quick-qwen27b/example.py
    run.sh       local gpu or slurm, copied from robot-disco/run-qwen27b.sh minus the server

No package, no tests dir, no Makefile. Nothing runs on the login node.

## What jev does

One forward pass, read the logits of the answer letters A..P at the first
generation position, softmax. No token sampled, no JSON parsed, no grammar.
A long state can be prefilled once and branched across many questions.

## jev.py

    # /// script
    # requires-python = ">=3.12"
    # dependencies = ["torch", "transformers", "accelerate"]
    # ///

    class Jev:
        def __init__(self, model="Qwen/Qwen3.8-27B", revision=None, weights_dir=None, seed=41): ...
        def decide(self, state, question, options) -> dict: ...
        def decide_many(self, state, questions) -> list[dict]: ...

- `__init__`: `set_storage` and `set_seed` lifted from quick-qwen27b; load
  `AutoModelForCausalLM` bf16 on `cuda:0`, `trust_remote_code=False`.
  Check whether Qwen3.8-27B is a multimodal config needing
  `get_text_config()` the way openjev handles qwen3_5.
- `decide(state, question, options)`: options is `list[str]` (descriptions)
  or `dict[id, description]`. Builds the frozen openjev prompt
  `{"evidence", "criterion", "options":[{"letter","description"}]}` with the
  openjev system prompt, chat template with `enable_thinking=False`, asserts
  each letter is a single round-trip token at the boundary, one forward with
  `logits_to_keep=1`, returns `{"probabilities": {id: p}, "logits": {id: l},
  "argmax": id, "input_tokens": n, "seconds": s}`.
- `decide_many(state, questions)`: `questions` is a list of
  `(question, options)` over one exact state. Prefill the state prefix with
  `use_cache=True`, `cache.reorder_cache` to replicate across the batch, one
  padded suffix forward with continued `position_ids` and
  `logits_to_keep=<end positions>`. Returns one `decide` dict per question.
  This is openjev `shared.py`; `serial.py` and `reranker.py` are dropped.

Deferred: `decide(..., think=True)` that generates the `<think>` trace first
and reads the letter logits after it. Same reasoning as chat, a distribution
comes out instead of a string.

## example.py

    from jev import Jev
    llm = Jev()
    print(llm.decide("The deployment completed at 14:02 UTC. Health checks passed.",
                     "Is there evidence that the deployment succeeded?",
                     {"yes": "It succeeded.", "no": "It did not.", "insufficient": "Cannot tell."}))
    print(llm.decide_many(long_state, [(q, {"yes": "Yes", "no": "No"}) for q in criteria]))

Prints probabilities and seconds, then the same 21 questions through
`decide` one by one for the timing comparison.

## run.sh

robot-disco `run-qwen27b.sh` with the llama-server / vllm serve parts removed:
`nvidia-smi` present -> run the command here, else `sbatch` one h200 with
`--mem=128G`, `HF_HOME` on netscratch as in `env.sh`, tail the log.

    $ ./run.sh uv run example.py

## Steps

1. this plan, README, LICENSE                                   (done)
2. `jev.py` with `decide`, `example.py`, `run.sh`; smoke on one h200
3. `decide_many`; check argmax agreement with `decide` on the same rows
   (openjev saw 5-6 flips in 777)
4. timings and agreement into README
5. decide on `think=True`
