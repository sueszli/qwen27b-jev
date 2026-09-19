# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub"]
# ///
import time

from jev_v1 import Decision, JevV1


class JevV2(JevV1):
    def read(self, state: str | dict | list, question: str, options: list[str] | dict[str, str], think: bool, candidate: str | None = None) -> Decision:
        # same forward pass as jev, but a grammar allows only the answer letters and temperature 0 picks the likeliest, no odds
        started = time.perf_counter()
        ids, prompt = self.prompt(state, question, options, think, candidate)
        reasoning, input_tokens, cached_tokens = "", 0, 0
        if think:
            thought = self.post("/completion", {"prompt": prompt, "n_predict": self.max_think_tokens, "stop": ["</think>"], "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repeat_last_n": self.ctx, "repeat_penalty": 1.0, "samplers": ["penalties", "top_k", "temperature", "top_p", "min_p"], "cache_prompt": True})
            assert thought.get("stopping_word") == "</think>", f"thinking did not finish inside {self.max_think_tokens} tokens"
            reasoning, input_tokens, cached_tokens = thought["content"].strip(), thought["timings"]["prompt_n"], thought["timings"]["cache_n"]
            prompt = f"{prompt}{reasoning}\n</think>\n\n"
        letters = self.letters[: len(ids)]
        answer = self.post("/completion", {"prompt": prompt, "n_predict": 1, "grammar": "root ::= " + " | ".join(f'"{letter}"' for letter in letters), "temperature": 0, "cache_prompt": True})
        written = answer["content"].strip()
        assert written in letters, f"the model wrote {written!r} instead of one of {letters}"
        return Decision({i: float(letter == written) for i, letter in zip(ids, letters)}, ids[letters.index(written)], reasoning, input_tokens + answer["timings"]["prompt_n"], cached_tokens + answer["timings"]["cache_n"], time.perf_counter() - started)
