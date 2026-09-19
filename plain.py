# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub"]
# ///
import time

from jev import Decision, Jev


class Plain(Jev):
    def read(self, state: str | dict | list, question: str, options: list[str] | dict[str, str], think: bool, candidate: str | None = None) -> Decision:
        # the ordinary way: let the model write an answer letter and take the one it wrote, no odds
        started = time.perf_counter()
        ids, prompt = self.prompt(state, question, options, think, candidate)
        reasoning, input_tokens, cached_tokens = "", 0, 0
        if think:
            thought = self.post("/completion", {"prompt": prompt, "n_predict": self.max_think_tokens, "stop": ["</think>"], "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repeat_last_n": self.ctx, "repeat_penalty": 1.0, "samplers": ["penalties", "top_k", "temperature", "top_p", "min_p"], "cache_prompt": True})
            reasoning, input_tokens, cached_tokens = thought["content"].strip(), thought["timings"]["prompt_n"], thought["timings"]["cache_n"]
            prompt = f"{prompt}{reasoning}\n</think>\n\n"
        letters = self.letters[: len(ids)]
        sampling = {"temperature": 1.0, "top_p": 0.95} if think else {"temperature": 0.7, "top_p": 0.8}  # qwen samples a thought-out answer hotter than a direct one
        answer = self.post("/completion", {"prompt": prompt, "n_predict": 1, "grammar": "root ::= " + " | ".join(f'"{letter}"' for letter in letters), "top_k": 20, "min_p": 0.0, "cache_prompt": True, **sampling})
        written = answer["content"].strip()
        assert written in letters, f"the model wrote {written!r} instead of one of {letters}"
        return Decision({i: float(letter == written) for i, letter in zip(ids, letters)}, ids[letters.index(written)], reasoning, input_tokens + answer["timings"]["prompt_n"], cached_tokens + answer["timings"]["cache_n"], time.perf_counter() - started)
