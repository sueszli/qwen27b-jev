# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=5.8", "accelerate", "huggingface_hub"]
# ///
import copy
import importlib.util
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers

LETTERS = "ABCDEFGHIJKLMNOP"
SYSTEM = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning."

#
# utils
#
def set_storage(weights_dir: Path) -> Path:
    (weights_dir / "tmp").mkdir(parents=True, exist_ok=True)  # also creates weights_dir
    hf, pt, xdg = weights_dir / "hf", weights_dir / "torch", weights_dir / "cache"
    os.environ.update({"HF_HOME": str(hf), "HF_HUB_CACHE": str(hf), "TRANSFORMERS_CACHE": str(hf), "HF_XET_CACHE": str(hf / "xet"), "HF_HUB_DISABLE_TELEMETRY": "1", "TORCH_HOME": str(pt), "TORCHINDUCTOR_CACHE_DIR": str(pt / "inductor"), "TRITON_CACHE_DIR": str(pt / "triton"), "CUDA_CACHE_PATH": str(weights_dir / "cuda"), "XDG_CACHE_HOME": str(xdg), "XDG_DATA_HOME": str(xdg), "XDG_CONFIG_HOME": str(xdg), "TMPDIR": str(weights_dir / "tmp")})
    return weights_dir


def set_seed(seed: int = 41) -> int:
    os.environ.update({"PYTHONHASHSEED": str(seed), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    random.seed(seed)
    if importlib.util.find_spec("numpy"):
        importlib.import_module("numpy").random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def softmax(logits: list[float]) -> list[float]:
    assert len(logits) >= 2 and all(math.isfinite(l) for l in logits), "need at least two finite logits"
    weights = [math.exp(l - max(logits)) for l in logits]
    return [w / sum(weights) for w in weights]  # sums to 1 within float error

#
# setup
#
def load_text_model(model: str, revision: str) -> tuple:
    assert torch.cuda.is_available(), "no cuda device, use ./run.sh"
    common = {"revision": revision, "trust_remote_code": False, "cache_dir": os.environ["HF_HUB_CACHE"]}  # huggingface_hub read HF_HOME at import, before set_storage
    config = transformers.AutoConfig.from_pretrained(model, **common)
    assert config.model_type in ("qwen3_5", "qwen3_5_text"), f"unexpected model_type {config.model_type}"
    llm = transformers.Qwen3_5ForCausalLM.from_pretrained(model, config=config.get_text_config(), dtype=torch.bfloat16, device_map={"": "cuda:0"}, **common).eval()  # the repo is a vision-language model, get_text_config() drops the vision tower
    return llm, transformers.AutoTokenizer.from_pretrained(model, **common)


def slot_ids(tokenizer) -> tuple[list[int], int]:
    encoded = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    assert all(len(ids) == 1 and tokenizer.decode(ids) == letter for ids, letter in zip(encoded, LETTERS)), "an answer slot is not one exact round-trip token"
    assert len({ids[0] for ids in encoded}) == len(LETTERS), "answer slot tokens collide"
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id  # only used to pad shared suffixes
    assert pad is not None, "tokenizer has neither a pad nor an eos token to pad suffixes with"
    return [ids[0] for ids in encoded], pad

#
# inference
#
def encode_decision(tokenizer, slots: list[int], state: str | dict | list, question: str, options: list[str] | dict[str, str]) -> tuple[list[tuple[str, str]], str, list[int]]:
    assert isinstance(state, (str, dict, list)) and state and isinstance(question, str) and question and isinstance(options, (list, dict)), "need a nonempty state, a nonempty question and list or dict options"
    json.dumps([state, options], ensure_ascii=False, allow_nan=False)  # rejects nan, inf and unserializable data
    pairs = [(o, o) for o in options] if isinstance(options, list) else list(options.items())
    assert 2 <= len(pairs) <= len(LETTERS) and all(isinstance(i, str) and i and isinstance(d, str) and d for i, d in pairs) and len({i for i, _ in pairs}) == len(pairs), f"need 2..{len(LETTERS)} options with unique nonempty string ids and nonempty descriptions, got {len(pairs)}"
    payload = {"evidence": state, "criterion": question, "options": [{"letter": letter, "description": description} for letter, (_, description) in zip(LETTERS, pairs)]}
    prompt = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    assert all(tokenizer.encode(prompt + letter, add_special_tokens=False) == ids + [slot] for letter, slot in zip(LETTERS[: len(pairs)], slots[: len(pairs)])), "an answer boundary changes tokenization"  # a merge with the preceding newline would move the answer off its slot
    return pairs, prompt, ids


def state_prefix(tokenizer, slots: list[int], state: str | dict | list, encoded: list[list[int]]) -> list[int]:
    prompt = encode_decision(tokenizer, slots, state, "prefix boundary placeholder", {"yes": "Yes", "no": "No"})[1]  # criterion and options follow the evidence
    evidence = json.dumps({"evidence": state}, ensure_ascii=False)[:-1]  # the serialized state without its closing brace
    assert prompt.count(evidence) == 1, "cannot locate the serialized evidence in the rendered prompt"
    prefix = tokenizer.encode(prompt[: prompt.index(evidence)] + evidence, add_special_tokens=False)[:-1]  # json punctuation can merge across the cut
    assert prefix and all(len(ids) > len(prefix) and ids[: len(prefix)] == prefix for ids in encoded), "the state prefix does not start every full prompt"
    return prefix


def shared_logits(model, prefix: list[int], suffixes: list[list[int]], pad: int) -> list[torch.Tensor]:
    device, offset, width = next(model.parameters()).device, len(prefix), max(map(len, suffixes))
    ends, keep, positions = [len(s) - 1 for s in suffixes], sorted({len(s) - 1 for s in suffixes}), [list(range(offset, offset + len(s))) + [0] * (width - len(s)) for s in suffixes]
    cache = model(input_ids=torch.tensor([prefix], dtype=torch.long, device=device), attention_mask=torch.ones((1, offset), dtype=torch.long, device=device), use_cache=True, logits_to_keep=1).past_key_values
    assert cache is not None and cache.get_seq_length() == offset, "the state prefix did not fill the cache"
    try:  # path (a) of PLAN.md: transformers reorders DynamicLayer kv rows and LinearAttentionLayer conv and recurrent states alike, by index_select
        cache.reorder_cache(torch.zeros(len(suffixes), dtype=torch.long, device=device))
        padded = model(input_ids=torch.tensor([s + [pad] * (width - len(s)) for s in suffixes], dtype=torch.long, device=device), attention_mask=torch.tensor([[1] * (offset + len(s)) + [0] * (width - len(s)) for s in suffixes], dtype=torch.long, device=device), position_ids=torch.tensor(positions, dtype=torch.long, device=device), past_key_values=cache, use_cache=True, logits_to_keep=torch.tensor(keep, dtype=torch.long, device=device))
        return [padded.logits[row, keep.index(end)] for row, end in enumerate(ends)]
    except Exception as error:  # noqa: BLE001
        print(f"jev: replicated prefix cache failed ({type(error).__name__}: {error}), replaying one suffix at a time")
    del cache  # the failed attempt may hold most of the gpu, e.g. after an oom on a small card
    cache = model(input_ids=torch.tensor([prefix], dtype=torch.long, device=device), attention_mask=torch.ones((1, offset), dtype=torch.long, device=device), use_cache=True, logits_to_keep=1).past_key_values  # path (b): refill, path (a) may have left the cache half replicated
    return [model(input_ids=torch.tensor([s], dtype=torch.long, device=device), attention_mask=torch.ones((1, offset + len(s)), dtype=torch.long, device=device), position_ids=torch.tensor([p[: len(s)]], dtype=torch.long, device=device), past_key_values=copy.deepcopy(cache), use_cache=True, logits_to_keep=1).logits[0, -1] for s, p in zip(suffixes, positions)]


@dataclass
class Decision:
    probabilities: dict[str, float]  # option id -> p, sums to 1
    logits: dict[str, float]  # option id -> raw logit
    argmax: str  # option id with max p
    input_tokens: int
    seconds: float  # wall time of the whole call

    @staticmethod
    def read(pairs: list[tuple[str, str]], vocabulary: torch.Tensor, slots: list[int], input_tokens: int, seconds: float) -> "Decision":
        probabilities = softmax(logits := vocabulary.float()[slots[: len(pairs)]].cpu().tolist())
        return Decision({i: p for (i, _), p in zip(pairs, probabilities)}, {i: l for (i, _), l in zip(pairs, logits)}, pairs[max(range(len(pairs)), key=probabilities.__getitem__)][0], input_tokens, seconds)


class Jev:
    def __init__(self, weights_dir: str | Path | None = None, model: str = "Qwen/Qwen3.8-27B", revision: str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", seed: int = 41):
        set_storage(Path(weights_dir or Path(__file__).resolve().parent / "weights"))
        self.seed = set_seed(seed)
        self.model, self.tokenizer = load_text_model(model, revision)
        self.slots, self.pad = slot_ids(self.tokenizer)

    def decide(self, state: str | dict | list, question: str, options: list[str] | dict[str, str]) -> Decision:
        started = time.perf_counter()
        pairs, _, ids = encode_decision(self.tokenizer, self.slots, state, question, options)
        with torch.inference_mode():
            vocabulary = self.model(input_ids=(input_ids := torch.tensor([ids], dtype=torch.long, device=next(self.model.parameters()).device)), attention_mask=torch.ones_like(input_ids), use_cache=False, logits_to_keep=1).logits[0, -1]
        return Decision.read(pairs, vocabulary, self.slots, len(ids), time.perf_counter() - started)

    def decide_many(self, state: str | dict | list, questions: list[tuple[str, list[str] | dict[str, str]]]) -> list[Decision]:
        started = time.perf_counter()
        assert isinstance(questions, list) and questions, "questions must be a nonempty list of (question, options)"
        decisions = [encode_decision(self.tokenizer, self.slots, state, question, options) for question, options in questions]
        prefix = state_prefix(self.tokenizer, self.slots, state, [ids for _, _, ids in decisions])
        try:
            with torch.inference_mode():
                logits = shared_logits(self.model, prefix, [ids[len(prefix) :] for _, _, ids in decisions], self.pad)
        except Exception as error:  # path (c) of PLAN.md: no cache path survived, so pay for the state once per question  # noqa: BLE001
            print(f"jev: shared prefix cache unusable ({type(error).__name__}: {error}), one full forward per question")
            logits = None
        if logits is None:  # outside the except block, whose traceback pins the failed attempt's tensors
            return [self.decide(state, question, options) for question, options in questions]
        return [Decision.read(pairs, row, self.slots, len(ids), time.perf_counter() - started) for (pairs, _, ids), row in zip(decisions, logits)]  # every Decision reports the wall time of the whole shared call
