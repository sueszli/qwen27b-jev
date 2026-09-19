# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub"]
# ///
import time

from jev_v1 import Decision, JevV1


class JevV2(JevV1):
    def read(self, state: str | dict | list, question: str, options: list[str] | dict[str, str], candidate: str | None = None) -> Decision:
        # same forward pass. grammar restricts the next token to the answer letters. greedy. no odds.
        started = time.perf_counter()
        ids, prompt = self.prompt(state, question, options, candidate)
        letters = self.letters[: len(ids)]
        answer = self.post("/completion", {"prompt": prompt, "n_predict": 1, "grammar": "root ::= " + " | ".join(f'"{letter}"' for letter in letters), "temperature": 0, "cache_prompt": True})
        written = answer["content"].strip()
        assert written in letters, f"the model wrote {written!r} instead of one of {letters}"
        return Decision({i: float(letter == written) for i, letter in zip(ids, letters)}, ids[letters.index(written)], answer["timings"]["prompt_n"], answer["timings"]["cache_n"], time.perf_counter() - started)
