# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=5.8", "accelerate", "huggingface_hub"]
# ///
import copy
import importlib.util
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
import transformers

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def set_storage(weights_dir: Path) -> Path:
    # keep every download and cache under one directory
    (weights_dir / "tmp").mkdir(parents=True, exist_ok=True)
    hf, pt, xdg = weights_dir / "hf", weights_dir / "torch", weights_dir / "cache"
    os.environ.update({"HF_HOME": str(hf), "HF_HUB_CACHE": str(hf), "TRANSFORMERS_CACHE": str(hf), "HF_XET_CACHE": str(hf / "xet"), "HF_HUB_DISABLE_TELEMETRY": "1", "TORCH_HOME": str(pt), "TORCHINDUCTOR_CACHE_DIR": str(pt / "inductor"), "TRITON_CACHE_DIR": str(pt / "triton"), "CUDA_CACHE_PATH": str(weights_dir / "cuda"), "XDG_CACHE_HOME": str(xdg), "XDG_DATA_HOME": str(xdg), "XDG_CONFIG_HOME": str(xdg), "TMPDIR": str(weights_dir / "tmp")})
    return weights_dir


def set_seed(seed: int = 41) -> int:
    # make runs reproducible
    os.environ.update({"PYTHONHASHSEED": str(seed), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    random.seed(seed)
    if importlib.util.find_spec("numpy"):
        importlib.import_module("numpy").random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


@dataclass
class Decision:
    probabilities: dict[str, float]
    logits: dict[str, float]
    argmax: str
    reasoning: str
    input_tokens: int
    seconds: float


class Jev:
    def __init__(self, weights_dir: str | Path = WEIGHTS_DIR, model: str = "Qwen/Qwen3.8-27B", revision: str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", seed: int = 41, batch_size: int = 8, max_think_tokens: int = 4096):
        # load the text half of the model and find the token id of each answer letter
        set_storage(Path(weights_dir))
        self.seed = set_seed(seed)
        self.batch_size, self.max_think_tokens = batch_size, max_think_tokens
        assert torch.cuda.is_available(), "no cuda device"
        common = {"revision": revision, "trust_remote_code": False, "cache_dir": os.environ["HF_HUB_CACHE"]}  # huggingface_hub read HF_HOME at import, before set_storage
        config = transformers.AutoConfig.from_pretrained(model, **common)
        self.model = transformers.Qwen3_5ForCausalLM.from_pretrained(model, config=config.get_text_config(), dtype=torch.bfloat16, device_map={"": "cuda:0"}, **common).eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, **common)
        self.letters = "ABCDEFGHIJKLMNOP"
        encoded = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in self.letters]
        assert self.tokenizer.pad_token_id is not None and all(len(e) == 1 for e in encoded) and len({e[0] for e in encoded}) == len(encoded), "tokenizer lacks a pad token or one-token answer letters"
        self.slots = torch.tensor([e[0] for e in encoded], device=self.model.device)
        self.think_end = self.tokenizer.convert_tokens_to_ids("</think>")

    def prompt(self, state: str | dict | list, question: str, options: list[str] | dict[str, str], think: bool, candidate: str | None = None) -> tuple[list[str], str]:
        # render one chat prompt whose next token is the answer letter, or the start of a thinking block
        pairs = [(o, o) for o in options] if isinstance(options, list) else list(options.items())
        assert state and question and 2 <= len(pairs) <= len(self.letters) and len({i for i, _ in pairs}) == len(pairs), f"need a nonempty state and question and 2..{len(self.letters)} unique options"
        payload = {"evidence": state, "criterion": question, **({"candidate": candidate} if candidate else {}), "options": [{"letter": letter, "description": description} for letter, (_, description) in zip(self.letters, pairs)]}
        system = "Apply the supplied criterion to the supplied evidence. If a candidate is supplied, judge that candidate against the criterion. Choose exactly one listed option and answer with only its uppercase letter."
        messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
        return [i for i, _ in pairs], self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=think)

    @torch.inference_mode()
    def think(self, prompts: list[str]) -> tuple[list[str], list[str]]:
        # sample a thinking block for every prompt, then close it so the next token is the answer letter
        reasonings, completed = [], []
        for i in range(0, len(prompts), self.batch_size):
            batch = self.tokenizer(prompts[i : i + self.batch_size], return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to(self.model.device)
            generated = self.model.generate(**batch, max_new_tokens=self.max_think_tokens, do_sample=True, temperature=1.0, top_p=0.95, top_k=20, eos_token_id=[self.think_end, self.tokenizer.eos_token_id], pad_token_id=self.tokenizer.pad_token_id)
            for prompt, row in zip(prompts[i : i + self.batch_size], generated[:, batch.input_ids.shape[1] :]):
                reasoning = self.tokenizer.decode(row, skip_special_tokens=True).split("</think>")[0].strip()
                reasonings.append(reasoning)
                completed.append(f"{prompt}{reasoning}\n</think>\n\n")
        return reasonings, completed

    @torch.inference_mode()
    def score(self, prompts: list[str]) -> tuple[torch.Tensor, list[int]]:
        # prefill the tokens all prompts share once, then run every prompt's own tail against a copy of that cache
        encoded = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        shared = min(next((i for i, column in enumerate(zip(*encoded)) if len(set(column)) > 1), min(map(len, encoded))), min(map(len, encoded)) - 1)
        prefix = torch.tensor([encoded[0][:shared]], device=self.model.device)
        cache = self.model(input_ids=prefix, use_cache=True, logits_to_keep=1).past_key_values
        rows = []
        for i in range(0, len(encoded), self.batch_size):
            tails = [e[shared:] for e in encoded[i : i + self.batch_size]]
            width, keep = max(map(len, tails)), sorted({len(t) - 1 for t in tails})
            input_ids = torch.tensor([t + [self.tokenizer.pad_token_id] * (width - len(t)) for t in tails], device=self.model.device)
            attention_mask = torch.tensor([[1] * (shared + len(t)) + [0] * (width - len(t)) for t in tails], device=self.model.device)
            position_ids = torch.arange(shared, shared + width, device=self.model.device).expand(len(tails), -1)
            batch_cache = copy.deepcopy(cache)
            batch_cache.reorder_cache(torch.zeros(len(tails), dtype=torch.long, device=self.model.device))
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=batch_cache, use_cache=True, logits_to_keep=torch.tensor(keep, device=self.model.device)).logits
            rows.append(logits[torch.arange(len(tails)), [keep.index(len(t) - 1) for t in tails]])
        return torch.cat(rows).float(), [len(e) for e in encoded]

    def read(self, rendered: list[tuple[list[str], str]], think: bool) -> list[Decision]:
        # think if asked, score, softmax the letter logits of each prompt into a Decision
        started = time.perf_counter()
        order = sorted(range(len(rendered)), key=lambda i: len(rendered[i][1]))  # similar lengths share a batch, less padding
        ids, prompts = [rendered[i][0] for i in order], [rendered[i][1] for i in order]
        reasonings, prompts = self.think(prompts) if think else ([""] * len(prompts), prompts)
        logits, lengths = self.score(prompts)
        decisions = [None] * len(rendered)
        for position, option_ids, row, reasoning, length in zip(order, ids, logits, reasonings, lengths):
            option_logits = row[self.slots[: len(option_ids)]]
            probabilities = option_logits.softmax(dim=0)
            decisions[position] = Decision(dict(zip(option_ids, probabilities.tolist())), dict(zip(option_ids, option_logits.tolist())), option_ids[probabilities.argmax().item()], reasoning, length, time.perf_counter() - started)
        return decisions

    def decide_many(self, state: str | dict | list, questions: list[tuple[str, list[str] | dict[str, str]]], think: bool = False) -> list[Decision]:
        # exactly one option per question, all questions over the same state
        return self.read([self.prompt(state, question, options, think) for question, options in questions], think)

    def decide(self, state: str | dict | list, question: str, options: list[str] | dict[str, str], think: bool = False) -> Decision:
        # exactly one option
        return self.decide_many(state, [(question, options)], think)[0]

    def select(self, state: str | dict | list, question: str, options: list[str] | dict[str, str], think: bool = False) -> dict[str, Decision]:
        # all options that apply: one independent yes/no decision per option
        pairs = [(o, o) for o in options] if isinstance(options, list) else list(options.items())
        verdict = {"yes": "The candidate satisfies the criterion.", "no": "The candidate does not satisfy the criterion."}
        return dict(zip([i for i, _ in pairs], self.read([self.prompt(state, question, verdict, think, candidate=description) for _, description in pairs], think)))

    def close(self) -> None:
        # free the gpu
        del self.model
        torch.cuda.empty_cache()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()
