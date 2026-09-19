# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub"]
# ///
import re
import time

from jev_v1 import Decision, JevV1


class Plain(JevV1):
    def read(self, state: str | dict | list, question: str, options: list[str] | dict[str, str], candidate: str | None = None) -> Decision:
        # no constraint. the model writes, we regex the letter out.
        started = time.perf_counter()
        ids, prompt = self.prompt(state, question, options, candidate)
        letters = self.letters[: len(ids)]
        answer = self.post("/completion", {"prompt": prompt, "n_predict": 256, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "cache_prompt": True})  # qwen defaults.
        found = re.findall(rf"\b([{letters}])\b", answer["content"])
        written = found[0] if found else None  # first standalone letter. none counts as wrong.
        return Decision({i: float(letter == written) for i, letter in zip(ids, letters)}, ids[letters.index(written)] if written else "", answer["timings"]["prompt_n"], answer["timings"]["cache_n"], time.perf_counter() - started)
