# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=5.8", "accelerate", "huggingface_hub"]
# ///
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


def set_storage(weights_dir: Path) -> Path:
    (weights_dir / "tmp").mkdir(parents=True, exist_ok=True)
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


@dataclass
class Decision:
    probabilities: dict[str, float]
    logits: dict[str, float]
    argmax: str
    input_tokens: int
    seconds: float


class Jev:
    def __init__(self, weights_dir: str | Path | None = None, model: str = "Qwen/Qwen3.8-27B", revision: str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", seed: int = 41, batch_size: int = 8):
        set_storage(Path(weights_dir or Path(__file__).resolve().parent / "weights"))
        self.seed = set_seed(seed)
        self.batch_size = batch_size
        assert torch.cuda.is_available(), "no cuda device"
        common = {"revision": revision, "trust_remote_code": False, "cache_dir": os.environ["HF_HUB_CACHE"]}  # huggingface_hub read HF_HOME at import, before set_storage
        config = transformers.AutoConfig.from_pretrained(model, **common)
        self.model = transformers.Qwen3_5ForCausalLM.from_pretrained(model, config=config.get_text_config(), dtype=torch.bfloat16, device_map={"": "cuda:0"}, **common).eval()  # the repo is a vision-language model, get_text_config() drops the vision tower
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, padding_side="right", **common)
        self.letters = "ABCDEFGHIJKLMNOP"
        encoded = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in self.letters]
        assert self.tokenizer.pad_token_id is not None and all(len(e) == 1 for e in encoded) and len({e[0] for e in encoded}) == len(encoded), "tokenizer lacks a pad token or one-token answer letters"
        self.slots = torch.tensor([e[0] for e in encoded], device=self.model.device)

    def prompt(self, state: str | dict | list, question: str, options: list[str] | dict[str, str]) -> tuple[list[str], str]:
        pairs = [(o, o) for o in options] if isinstance(options, list) else list(options.items())
        assert state and question and 2 <= len(pairs) <= len(self.letters) and len({i for i, _ in pairs}) == len(pairs), f"need a nonempty state and question and 2..{len(self.letters)} unique options"
        payload = {"evidence": state, "criterion": question, "options": [{"letter": letter, "description": description} for letter, (_, description) in zip(self.letters, pairs)]}
        messages = [{"role": "system", "content": "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning."}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
        return [i for i, _ in pairs], self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    @torch.inference_mode()
    def decide_many(self, state: str | dict | list, questions: list[tuple[str, list[str] | dict[str, str]]]) -> list[Decision]:
        started = time.perf_counter()
        ids, prompts = zip(*(self.prompt(state, question, options) for question, options in questions))
        decisions = []
        for i in range(0, len(prompts), self.batch_size):
            batch = self.tokenizer(list(prompts[i : i + self.batch_size]), return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            lengths = batch.attention_mask.sum(dim=1)
            logits = self.model(**batch, use_cache=False).logits[torch.arange(len(lengths)), lengths - 1].float()  # one row per prompt, read at its last real token
            for option_ids, row, length in zip(ids[i : i + self.batch_size], logits, lengths.tolist()):
                option_logits = row[self.slots[: len(option_ids)]]
                probabilities = option_logits.softmax(dim=0)
                decisions.append(Decision(dict(zip(option_ids, probabilities.tolist())), dict(zip(option_ids, option_logits.tolist())), option_ids[probabilities.argmax().item()], length, time.perf_counter() - started))
        return decisions

    def decide(self, state: str | dict | list, question: str, options: list[str] | dict[str, str]) -> Decision:
        return self.decide_many(state, [(question, options)])[0]

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()
