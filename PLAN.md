# Plan

Port https://github.com/TheoLeeCJ/openjev from Qwen3.5-4B to Qwen3.8-27B.
Read this whole file before writing code. Every decision that is not marked
"open" is settled; do not reopen it.

## Deliverables

Exactly three files, next to README, PLAN.md, LICENSE, .gitignore:

    jev.py       the library: a uv script with inline deps, one class, ~120 loc
    example.py   the demo, shape of https://github.com/sueszli/quick-qwen27b/blob/master/example.py
    run.sh       runs a command on a gpu: local if present, else slurm

No package, no pyproject, no tests directory, no Makefile, no benchmarks. The
style reference is https://github.com/sueszli/quick-qwen27b/blob/master/qwen.py:
flat functions under `# utils` / `# setup` / `# inference` comment banners,
asserts instead of exceptions, one class at the bottom.

## Facts about the model, checked 2026 against the hub

- `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Pin it.
- `model_type` is `qwen3_5`, architecture `Qwen3_5ForConditionalGeneration`,
  a vision-language model. Text weights are ~52 GiB bf16. We only want the
  text model: `config.get_text_config()` gives `qwen3_5_text`, loaded with
  `transformers.Qwen3_5ForCausalLM`, exactly what openjev `core.py` does.
- The text model is a hybrid: 64 layers, 48 `linear_attention` (Gated
  DeltaNet) and 16 `full_attention`. Its cache is therefore not a plain
  key/value cache. This matters for `decide_many`, see below.
- `vocab_size` 248320. `tokenizer_config` needs transformers >= 5.8.
- Chat template with `add_generation_prompt=True, enable_thinking=False`
  ends in `<|im_start|>assistant\n<think>\n\n</think>\n\n`. The first
  generated token after that is the answer. Never pass `reasoning_effort`
  when thinking is off.

## What jev does

Prompt the model with a state, a question and 2..16 options labelled A..P.
End the prompt where the model would answer. One forward pass. Take the
logits of the 2..16 letter tokens at the last position, softmax, return.
Nothing is sampled, appended, generated or parsed.

## jev.py

    # /// script
    # requires-python = ">=3.12"
    # dependencies = ["torch", "transformers>=5.8", "accelerate", "huggingface_hub"]
    # ///

    class Jev:
        def __init__(self, weights_dir=None, model="Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", seed=41)
        def decide(self, state, question, options) -> Decision
        def decide_many(self, state, questions) -> list[Decision]

    @dataclass
    class Decision:
        probabilities: dict[str, float]   # option id -> p, sums to 1
        logits: dict[str, float]          # option id -> raw logit
        argmax: str                       # option id with max p
        input_tokens: int
        seconds: float                    # wall time of the whole call

Types: `state: str | dict | list` (non-str is `json.dumps`ed, must be
nonempty and finite), `question: str`, `options: list[str] | dict[str, str]`.
A list means id == description. A dict maps id -> description. 2..16
options, ids unique. `questions: list[tuple[str, options]]`.

### `__init__`

1. `set_storage(weights_dir)` and `set_seed(seed)` copied from quick-qwen27b
   `qwen.py`. Default `weights_dir` is `Path(__file__).parent / "weights"`.
2. `assert torch.cuda.is_available()`.
3. Load config, `get_text_config()`, `Qwen3_5ForCausalLM.from_pretrained(...,
   config=text_config, dtype=torch.bfloat16, device_map={"": "cuda:0"},
   revision=revision, trust_remote_code=False)`. `.eval()`.
4. Tokenizer from the same repo and revision.
5. Precompute `self.slots`: token id of each letter `A`..`P`. Assert each
   encodes to exactly one token and decodes back to the letter. Assert all 16
   are distinct.

### prompt

Frozen, identical to openjev so numbers are comparable:

    system: "Apply the supplied criterion to the supplied evidence. Choose
             exactly one listed option. Respond with only its uppercase
             letter, with no explanation or reasoning."
    user:   json.dumps({"evidence": state, "criterion": question,
                        "options": [{"letter": "A", "description": ...}, ...]},
                       ensure_ascii=False)

Rendered with `tokenizer.apply_chat_template(messages, tokenize=False,
add_generation_prompt=True, enable_thinking=False)`, then
`tokenizer.encode(prompt, add_special_tokens=False)`.

Boundary assert, from openjev `direct.py`: for each used letter,
`encode(prompt + letter) == encode(prompt) + [slot]`. If this fails the
letter merged with the preceding newline and readout would be wrong.

### `decide`

One forward: `model(input_ids, attention_mask, use_cache=False,
logits_to_keep=1).logits[0, -1]`. Index with the first `len(options)` slots,
`.float()`, softmax in python (`math.exp(l - max)`), build `Decision`.
Wrap in `torch.inference_mode()`. Time from call entry to return.

### `decide_many`

All questions share one exact `state`. Goal: pay for the state once.

Openjev `shared.py` does it with a plain kv cache: prefill the state prefix
with `use_cache=True`, `cache.reorder_cache(zeros(n))` to replicate it n
times, then one padded forward over the n suffixes with `position_ids`
continuing from the prefix and `logits_to_keep=<each suffix's last index>`.

Open: whether that works on Qwen3.8's hybrid cache. The DeltaNet layers keep
a recurrent state, not kv rows, and `reorder_cache` may not exist or may not
replicate it. Do this in order and stop at the first that works:

a. Try openjev's approach verbatim. If `cache.reorder_cache` exists and the
   forward runs, check agreement with `decide` on the same rows (below).
b. Else, prefill the prefix once and loop: for each question, deep-copy the
   cache, forward the suffix alone with `use_cache=True`. No padding, no
   batch. Slower than (a) but still skips the prefix n-1 times.
c. Else, fall back to calling `decide` n times and say so in a comment.

The prefix is the token sequence up to and including the state, computed
as in openjev `_state_prefix`: render the full prompt for a dummy question,
cut the text right after the serialized `"evidence": <state>`, encode, drop
the last token (JSON punctuation can merge across the boundary). Assert every
full prompt starts with this prefix.

Acceptance for (a) or (b): on 21 questions over one ~8k-char state, argmax
agrees with `decide` on >= 20/21 and the max absolute probability difference
is < 0.05. Openjev saw 5-6 argmax flips in 777 from bf16 noise; more than
that means the cache path is wrong, not noisy.

### not doing

Reranker mode and serial mode from openjev. `think=True` (generate the
`<think>` trace, then read the letter logits after `</think>`) is a possible
follow-up once the plain readout is measured; do not build it now.

## example.py

    from jev import Jev

    llm = Jev()
    print(llm.decide("The deployment completed at 14:02 UTC. Health checks passed in all three zones.",
                     "Is there evidence that the deployment succeeded?",
                     {"yes": "It succeeded.", "no": "It did not.", "unclear": "Cannot tell."}))

Then a shared-state demo: one ~8k-char made-up incident report as `state`,
21 yes/no questions about it. Run them through `decide_many` and through
`decide` in a loop, print both timings and the number of argmax
disagreements. Print `Decision` fields on one line each, like quick-qwen27b
prints tokens and seconds. No argparse.

## run.sh

`#!/usr/bin/env bash`, `set -euo pipefail`, `usage: run.sh <command...>`.
Copy the control flow of robot-disco `run-qwen27b.sh` and drop everything
about llama-server, vllm, ports, ssh tunnels and health checks. What remains:

- `nvidia-smi -L` shows a GPU: `exec "$@"` here.
- else `sbatch` exists: `sbatch --parsable -J qwen27b-jev -p seas_gpu
  --gres=gpu:nvidia_h200:1 -c 16 --mem=128G -t 4:00:00 -o logs/%j.out
  --wrap "$*"`, then `tail -F` the log until the job leaves the queue.
  Set `HF_HOME` to a netscratch path before sbatch so weights are not
  downloaded into `$HOME`; take the pattern from robot-disco `env.sh`.
- else: print "no gpu and no slurm", exit 1.

The login node has 1 core and 8 GB. `run.sh` is the only way anything in
this repo touches the model.

## Steps, each one a PR

1. `jev.py` with `__init__`, prompt, `decide`. `example.py` with only the
   first call. `run.sh`. Smoke: `./run.sh uv run example.py` on one h200
   prints a `Decision` with `yes` as argmax.
2. `decide_many` per (a)/(b)/(c). Extend `example.py` with the 21-question
   demo. Record which path worked and the agreement numbers in the PR.
3. Put the measured timings and agreement into README under the demo.

## Definition of done for step 1

- `uv run --script jev.py` imports without a GPU (module level must not
  touch torch.cuda; only `Jev()` does).
- `./run.sh uv run example.py` completes on one h200 and the printed
  probabilities sum to 1 within 1e-6.
- `jev.py` under 150 lines, `example.py` under 40, `run.sh` under 40.
- No file other than the three above, README, PLAN.md, LICENSE, .gitignore.

## References

- OpenJev: https://github.com/TheoLeeCJ/openjev, especially
  `src/openjev_phase1/{core,direct,shared}.py`
- quick-qwen27b, the code style and `set_storage`/`set_seed`:
  https://github.com/sueszli/quick-qwen27b
- robot-disco, the launcher shape: https://github.com/metareflection/robot-disco
  (`run-qwen27b.sh`, `env.sh`)
- Qwen3.8-27B: https://huggingface.co/Qwen/Qwen3.8-27B
