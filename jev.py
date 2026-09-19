# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub"]
# ///
import atexit
import importlib
import json
import os
import subprocess
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Self

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def set_storage(weights_dir: Path) -> Path:
    # keep every download and cache under one directory
    (weights_dir / "tmp").mkdir(parents=True, exist_ok=True)
    hf, xdg = weights_dir / "hf", weights_dir / "cache"
    os.environ.update({"HF_HOME": str(hf), "HF_HUB_CACHE": str(hf), "HF_XET_CACHE": str(hf / "xet"), "HF_HUB_DISABLE_TELEMETRY": "1", "LLAMA_CACHE": str(weights_dir / "llama"), "XDG_CACHE_HOME": str(xdg), "XDG_DATA_HOME": str(xdg), "XDG_CONFIG_HOME": str(xdg), "TMPDIR": str(weights_dir / "tmp")})
    return weights_dir


def download_llama_server(weights_dir: Path, tag: str = "b10908") -> Path:
    # fetch a prebuilt llama.cpp release once
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
    # fetch the quantized weights once
    path = Path(importlib.import_module("huggingface_hub").hf_hub_download(repo, filename, local_dir=weights_dir / repo.split("/")[1], cache_dir=weights_dir / "hf"))
    assert path.is_relative_to(weights_dir), f"{path} escaped {weights_dir}"
    return path


def start_llama_server(weights_dir: Path, repo: str, model: str, ctx: int, seed: int, port: int) -> subprocess.Popen:
    # run the model on the gpu behind a local http server and wait until it answers
    cmd = [str(download_llama_server(weights_dir)), "-m", str(download_gguf(weights_dir, repo, model)), "-ngl", "99", "-c", str(ctx), "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-np", "1", "--seed", str(seed), "--host", "127.0.0.1", "--port", str(port)]
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


class Jev:
    def __init__(self, weights_dir: str | Path = WEIGHTS_DIR, repo: str = "unsloth/Qwen3.8-27B-GGUF", model: str = "Qwen3.8-27B-UD-Q5_K_XL.gguf", ctx: int = 32768, seed: int = 41, port: int = 8080, max_think_tokens: int = 4096):
        # start the server and check that every answer letter is one token
        weights_dir = set_storage(Path(weights_dir))
        self.port, self.max_think_tokens = port, max_think_tokens
        self.proc = start_llama_server(weights_dir, repo, model, ctx, seed, port)
        self.letters = "ABCDEFGHIJKLMNOP"
        assert all(len(self.post("/tokenize", {"content": letter, "add_special": False})["tokens"]) == 1 for letter in self.letters), "an answer letter is not one token"

    def post(self, path: str, body: dict) -> dict:
        # one json request to the server
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(body).encode(), {"content-type": "application/json"})
        with urllib.request.urlopen(request, timeout=24 * 3600) as response:
            return json.load(response)

    def prompt(self, state: str | dict | list, question: str, options: list[str] | dict[str, str], think: bool, candidate: str | None = None) -> tuple[list[str], str]:
        # render one chat prompt whose next token is the answer letter, or the start of a thinking block
        pairs = [(o, o) for o in options] if isinstance(options, list) else list(options.items())
        assert state and question and 2 <= len(pairs) <= len(self.letters) and len({i for i, _ in pairs}) == len(pairs), f"need a nonempty state and question and 2..{len(self.letters)} unique options"
        payload = {"evidence": state, "criterion": question, **({"candidate": candidate} if candidate else {}), "options": [{"letter": letter, "description": description} for letter, (_, description) in zip(self.letters, pairs)]}
        system = "Apply the supplied criterion to the supplied evidence. If a candidate is supplied, judge that candidate against the criterion. Choose exactly one listed option and answer with only its uppercase letter."
        messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
        return [i for i, _ in pairs], self.post("/apply-template", {"messages": messages, "chat_template_kwargs": {"enable_thinking": think}})["prompt"]

    def read(self, state: str | dict | list, question: str, options: list[str] | dict[str, str], think: bool, candidate: str | None = None) -> Decision:
        # optionally think, then read the odds of each answer letter off the next token, the server caches the shared prefix between calls
        started = time.perf_counter()
        ids, prompt = self.prompt(state, question, options, think, candidate)
        reasoning, input_tokens, cached_tokens = "", 0, 0
        if think:
            thought = self.post("/completion", {"prompt": prompt, "n_predict": self.max_think_tokens, "stop": ["</think>"], "temperature": 1.0, "top_p": 0.95, "top_k": 20, "cache_prompt": True})
            reasoning, input_tokens, cached_tokens = thought["content"].strip(), thought["timings"]["prompt_n"], thought["timings"]["cache_n"]
            prompt = f"{prompt}{reasoning}\n</think>\n\n"
        letters = self.letters[: len(ids)]
        answer = self.post("/completion", {"prompt": prompt, "n_predict": 1, "grammar": "root ::= " + " | ".join(f'"{letter}"' for letter in letters), "n_probs": len(ids), "post_sampling_probs": True, "temperature": 1.0, "samplers": ["temperature"], "cache_prompt": True})  # the grammar leaves only the answer letters, so their post-sampling probabilities are the softmax over exactly those
        probabilities = {letter: 0.0 for letter in letters} | {t["token"]: t["prob"] for t in answer["completion_probabilities"][0]["top_probs"]}
        probabilities = dict(zip(ids, (probabilities[letter] for letter in letters)))
        return Decision(probabilities, max(probabilities, key=probabilities.__getitem__), reasoning, input_tokens + answer["timings"]["prompt_n"], cached_tokens + answer["timings"]["cache_n"], time.perf_counter() - started)

    def decide(self, state: str | dict | list, question: str, options: list[str] | dict[str, str], think: bool = False) -> Decision:
        # exactly one option
        return self.read(state, question, options, think)

    def warm(self, state: str | dict | list, think: bool) -> None:
        # prefill the prompt up to the end of the state, the recurrent layers can only resume from where an earlier request stopped
        prompt = self.prompt(state, "placeholder", ["yes", "no"], think)[1]
        self.post("/completion", {"prompt": prompt[: prompt.index(', "criterion"')], "n_predict": 1, "cache_prompt": True})

    def decide_many(self, state: str | dict | list, questions: list[tuple[str, list[str] | dict[str, str]]], think: bool = False) -> list[Decision]:
        # exactly one option per question, all questions over the same state
        self.warm(state, think)
        return [self.read(state, question, options, think) for question, options in questions]

    def select(self, state: str | dict | list, question: str, options: list[str] | dict[str, str], think: bool = False) -> dict[str, Decision]:
        # all options that apply: one independent yes/no decision per option
        pairs = [(o, o) for o in options] if isinstance(options, list) else list(options.items())
        verdict = {"yes": "The candidate satisfies the criterion.", "no": "The candidate does not satisfy the criterion."}
        self.warm(state, think)
        return {i: self.read(state, question, verdict, think, candidate=description) for i, description in pairs}

    def close(self) -> None:
        # stop the server
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()
