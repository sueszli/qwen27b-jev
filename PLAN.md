# Plan

Reproduce the OpenJev interface (https://github.com/TheoLeeCJ/openjev) on
Qwen3.8-27B with as little code as possible: one flat module, one test file,
one launcher, plain-text README, same layout as sueszli/exojit.

## What we build

A semantic decision operator. Input is a row

    {"id": ..., "state": <str|json>, "question": <str>,
     "options": [{"id": ..., "description": ...}, ...]}   # 2..16 options

Output is a probability per option, read from the model's next-token logits
over the answer letters A..P at the first generation position. No token is
ever sampled, no JSON is parsed. Extra: a state can be prefilled once and
branched across many questions.

## Layout

    qwen27b_jev.py      the whole library and the cli          (~150 loc)
    tests/test_jev.py   prompt/softmax/validation tests, no gpu (~40 loc)
    run.sh              sbatch wrapper for one h200, like robot-disco run-qwen27b.sh
    examples.jsonl      the openjev example rows, verbatim
    README, PLAN.md, LICENSE, Makefile, pyproject.toml, .gitignore

OpenJev's `reranker.py` and `serial.py` are dropped: the reranker lost to
direct logits on every general-decision workload, and serial is dominated by
shared. `benchmarks/` is out of scope for now.

## qwen27b_jev.py

Collapse openjev `core.py` + `direct.py` + `shared.py` + `cli.py` into one file.

1. `validate(row)`: same rules as openjev (2..16 options, unique ids,
   nonempty finite-json state). Asserts, not exceptions.
2. `messages(row)`: frozen system prompt + user payload
   `{"evidence", "criterion", "options":[{"letter","description"}]}`.
   Keep the exact openjev wording so results stay comparable.
3. `load(model, revision)`: `AutoModelForCausalLM` in bf16 on `cuda:0`,
   pinned hf revision, `trust_remote_code=False`. Check whether Qwen3.8-27B
   is `qwen3_5`-style multimodal needing `get_text_config()`; branch as
   openjev does if so.
4. `encode(tok, row)`: chat template, `add_generation_prompt=True`,
   `enable_thinking=False`; assert each letter is one round-trip token and
   appending it does not change the prompt's tokenization.
5. `score(model, tok, row)`: one forward with `logits_to_keep=1`, index the
   letter slots, softmax. Return `{id, option_ids, probabilities,
   option_logits, input_tokens, seconds}`.
6. `score_shared(model, tok, rows)`: rows share one exact state. Prefill the
   state prefix with `use_cache=True`, `cache.reorder_cache` to replicate the
   kv across the batch, then one padded suffix forward with continued
   `position_ids` and `logits_to_keep=<end positions>`. Same return shape
   plus prefill/suffix timings.
7. `cli`: click. `qwen27b-jev --mode direct|shared --input rows.jsonl
   --output out.jsonl [--model Qwen/Qwen3.8-27B --revision <sha>]`.
   Output must not exist yet.

Deferred, only if step 5 shows the no-thinking 27B readout is good enough:

8. `score_thinking`: generate the `<think>...</think>` trace normally, then
   read the letter logits at the position after it instead of sampling.
   Same reasoning as chat, but a distribution over options comes out.

## run.sh

Copy the `slurm_gpu`/`local_gpu` shape from robot-disco `run-qwen27b.sh`,
but no server: `sbatch --gres=gpu:nvidia_h200:1 --mem=128G` running
`uv run qwen27b-jev ...` directly, `HF_HOME` on netscratch as in `env.sh`.
Never run the model on the login node.

## Tests

CPU only, run in `make precommit`:

- prompt excludes fields outside `state/question/options`
- softmax finite and normalized on extreme logits
- duplicate option ids rejected, >16 options rejected
- structured json state serialized into the prompt
- `encode` letter-slot assertions, using a small tokenizer if one is cached,
  else skipped

GPU checks go through `run.sh` and are not part of precommit.

## Steps

1. scaffold: pyproject, Makefile, README, LICENSE, .gitignore, examples.jsonl
2. `qwen27b_jev.py` steps 1-5 and 7, `tests/test_jev.py`, `make precommit`
3. `run.sh`; smoke on one h200 with `examples.jsonl` in direct mode
4. step 6 shared mode; smoke with 21 questions over one 8k-char state,
   compare against direct on the same rows (openjev saw 5-6/777 argmax flips)
5. write timings and agreement into README
6. decide on step 8

## Non-goals

Reranker mode, browser/webgpu demo, the 706-row frozen evaluation matrix,
calibration claims. Probabilities are conditional on the supplied options.
