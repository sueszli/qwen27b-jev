# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub"]
# ///
import atexit
import importlib
import json
import math
import os
import socket
import subprocess
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Self

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def set_storage(weights_dir: Path) -> Path:
    # all downloads and caches live in weights_dir.
    (weights_dir / "tmp").mkdir(parents=True, exist_ok=True)
    hf, xdg = weights_dir / "hf", weights_dir / "cache"
    os.environ.update({"HF_HOME": str(hf), "HF_HUB_CACHE": str(hf), "HF_XET_CACHE": str(hf / "xet"), "HF_HUB_DISABLE_TELEMETRY": "1", "LLAMA_CACHE": str(weights_dir / "llama"), "XDG_CACHE_HOME": str(xdg), "XDG_DATA_HOME": str(xdg), "XDG_CONFIG_HOME": str(xdg), "TMPDIR": str(weights_dir / "tmp")})
    return weights_dir


def download_llama_server(weights_dir: Path, tag: str = "b10908") -> Path:
    # prebuilt llama.cpp release. downloaded once.
    root = weights_dir / f"llama.cpp-{tag}"
    if not (root / "llama-server").exists():
        root.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz", root / "llama.tar.gz")
        with tarfile.open(root / "llama.tar.gz") as tar:
            tar.extractall(root, filter=lambda m, _: m.replace(name=m.name.split("/", 1)[1]) if "/" in m.name else None)
        (root / "llama.tar.gz").unlink()
    assert os.access(root / "llama-server", os.X_OK), f"{root / 'llama-server'} is missing or not executable"
    return root / "llama-server"


def download_gguf(weights_dir: Path, repo: str, filename: str) -> Path:
    # quantized weights. downloaded once.
    path = Path(importlib.import_module("huggingface_hub").hf_hub_download(repo, filename, local_dir=weights_dir / repo.split("/")[1], cache_dir=weights_dir / "hf"))
    assert path.is_relative_to(weights_dir), f"{path} escaped {weights_dir}"
    return path


def start_llama_server(weights_dir: Path, repo: str, model: str, ctx: int, seed: int, port: int) -> subprocess.Popen:
    # start llama-server on the gpu. block until /health answers.
    cmd = [str(download_llama_server(weights_dir)), "-m", str(download_gguf(weights_dir, repo, model)), "-ngl", "99", "-c", str(ctx), "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-np", "1", "--seed", str(seed), "--host", "127.0.0.1", "--port", str(port)]
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0, f"port {port} is already taken, a stale llama-server would answer instead of ours"
    proc = subprocess.Popen(cmd, stdout=(weights_dir / "llama-server.log").open("w"), stderr=subprocess.STDOUT)
    atexit.register(proc.terminate)
    for _ in range(600):
        assert proc.poll() is None, f"llama-server exited, see {weights_dir / 'llama-server.log'}"
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return proc
        except OSError:
            time.sleep(1)
    assert False, f"llama-server not healthy after 600s, see {weights_dir / 'llama-server.log'}"


@dataclass
class Decision:
    probabilities: dict[str, float]
    argmax: str
    reasoning: str
    input_tokens: int
    cached_tokens: int
    seconds: float


class JevV1:
    def __init__(self, weights_dir: str | Path = WEIGHTS_DIR, repo: str = "unsloth/Qwen3.8-27B-GGUF", model: str = "Qwen3.8-27B-UD-Q5_K_XL.gguf", ctx: int = 32768, seed: int = 41, port: int = 8080, max_think_tokens: int = 81920):
        # start the server. assert each answer letter is a single token.
        weights_dir = set_storage(Path(weights_dir))
        self.port, self.ctx, self.max_think_tokens = port, ctx, max_think_tokens
        self.proc = start_llama_server(weights_dir, repo, model, ctx, seed, port)
        self.letters = "ABCDEFGHIJKLMNOP"
        assert all(len(self.post("/tokenize", {"content": letter, "add_special": False})["tokens"]) == 1 for letter in self.letters), "an answer letter is not one token"

    def post(self, path: str, body: dict) -> dict:
        # json request, json response.
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(body).encode(), {"content-type": "application/json"})
        with urllib.request.urlopen(request, timeout=24 * 3600) as response:
            return json.load(response)

    def prompt(self, state: str | dict | list, question: str, options: list[str] | dict[str, str], think: bool, candidate: str | None = None) -> tuple[list[str], str]:
        # chat prompt. ends right before the answer letter, or before the thinking block.
        pairs = [(o, o) for o in options] if isinstance(options, list) else list(options.items())
        assert state and question and 2 <= len(pairs) <= len(self.letters) and len({i for i, _ in pairs}) == len(pairs), f"need a nonempty state and question and 2..{len(self.letters)} unique options"
        payload = {"evidence": state, "criterion": question, **({"candidate": candidate} if candidate is not None else {}), "options": [{"letter": letter, "description": description} for letter, (_, description) in zip(self.letters, pairs)]}
        system = "Apply the supplied criterion to the supplied evidence. If a candidate is supplied, judge that candidate against the criterion. Choose exactly one listed option and answer with only its uppercase letter."
        messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
        return [i for i, _ in pairs], self.post("/apply-template", {"messages": messages, "chat_template_kwargs": {"enable_thinking": think}})["prompt"]

    def read(self, state: str | dict | list, question: str, options: list[str] | dict[str, str], think: bool, candidate: str | None = None) -> Decision:
        # think if asked. read the odds of each letter off the next token. softmax over the letters.
        started = time.perf_counter()
        ids, prompt = self.prompt(state, question, options, think, candidate)
        reasoning, input_tokens, cached_tokens = "", 0, 0
        if think:
            thought = self.post("/completion", {"prompt": prompt, "n_predict": self.max_think_tokens, "stop": ["</think>"], "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repeat_last_n": self.ctx, "repeat_penalty": 1.0, "samplers": ["penalties", "top_k", "temperature", "top_p", "min_p"], "cache_prompt": True})
            assert thought.get("stopping_word") == "</think>", f"thinking did not finish inside {self.max_think_tokens} tokens"
            reasoning, input_tokens, cached_tokens = thought["content"].strip(), thought["timings"]["prompt_n"], thought["timings"]["cache_n"]
            prompt = f"{prompt}{reasoning}\n</think>\n\n"
        letters = self.letters[: len(ids)]
        answer = self.post("/completion", {"prompt": prompt, "n_predict": 1, "n_probs": 16, "temperature": 0, "cache_prompt": True})
        logprobs = {t["token"]: t["logprob"] for t in answer["completion_probabilities"][0]["top_logprobs"]}
        assert all(letter in logprobs for letter in letters), f"answer letters {[l for l in letters if l not in logprobs]} fell out of the top 16 next tokens, the prompt is not being read as a multiple-choice question"
        odds = [math.exp(logprobs[letter]) for letter in letters]
        probabilities = dict(zip(ids, (o / sum(odds) for o in odds)))
        return Decision(probabilities, max(probabilities, key=probabilities.__getitem__), reasoning, input_tokens + answer["timings"]["prompt_n"], cached_tokens + answer["timings"]["cache_n"], time.perf_counter() - started)

    def decide(self, state: str | dict | list, question: str, options: list[str] | dict[str, str], think: bool = False) -> Decision:
        # one question, one option.
        return self.read(state, question, options, think)

    def warm(self, state: str | dict | list, think: bool) -> None:
        # prefill the state once. later requests resume from its end.
        prompt = self.prompt(state, "placeholder", ["yes", "no"], think)[1]
        self.post("/completion", {"prompt": prompt[: prompt.index(', "criterion"')], "n_predict": 1, "cache_prompt": True})

    def decide_many(self, state: str | dict | list, questions: list[tuple[str, list[str] | dict[str, str]]], think: bool = False) -> list[Decision]:
        # many questions, one option each. state is prefilled once.
        self.warm(state, think)
        return [self.read(state, question, options, think) for question, options in questions]

    def select(self, state: str | dict | list, question: str, options: list[str] | dict[str, str], think: bool = False) -> dict[str, Decision]:
        # all options that apply. one yes/no per option.
        pairs = [(o, o) for o in options] if isinstance(options, list) else list(options.items())
        verdict = {"yes": "The candidate satisfies the criterion.", "no": "The candidate does not satisfy the criterion."}
        self.warm(state, think)
        return {i: self.read(state, question, verdict, think, candidate=description) for i, description in pairs}

    def close(self) -> None:
        # stop the server.
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()
