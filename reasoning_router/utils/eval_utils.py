import concurrent.futures
import json
import os
import re
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM

from reasoning_router.utils.llm_utils import get_response, model
from reasoning_router.router import get_router_response


class CustomModel(DeepEvalBaseLLM):
    def __init__(self, routed=False):
        self.routed = routed
        self.counter = 0

    def load_model(self):  # type: ignore
        return True

    def generate(self, prompt: str, *args, **kwargs):
        self.counter += 1
        # print(f"\n Generating response {self.counter}...")

        if self.routed:
            text = get_router_response(prompt, thread_id=str(self.counter))
        else:
            text = get_response(prompt)

        schema = kwargs.get("schema")
        if schema is None:
            return text

        # If DeepEval requests a structured schema, attempt to comply.
        try:
            constructed = self._construct_schema_with_llm(text, schema)
            if not constructed:
                raise ValueError("LLM-assisted schema construction returned None.")
            return constructed
        except Exception as e:
            print("WARNING: LLM-assisted schema construction failed.", e)
            return None


    def _construct_schema_with_llm(self, answer: str, schema: Any):
        """Use the LLM to produce a structured object that follows `schema`."""
        instruction = (
            f"You are an assistant that must produce JSON that strictly follows a provided structure.\n"
            f" Fill the schema fields based on this answer: {answer}.\n"
        )

        # Call the project's LLM; reuse get_response to keep behavior consistent.
        try:
            llm_output = get_response(instruction, schema=schema)
        except Exception as e:
            print("Error calling LLM for schema construction:", e)
            return None

        # Validate/convert parsed JSON into the schema model
        try:
            return schema.model_validate(llm_output)
        except Exception as e:
            print("Parsed JSON failed schema validation:", e)
            return None

    def _parse_json_from_text(self, t: str):
        t = (t or "").strip()
        # If the text is already just JSON, parse directly.
        try:
            return json.loads(t)
        except Exception:
            # Try to find a JSON object inside the text
            m = re.search(r"(\{.*\})", t, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    return None
            return None

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return self.generate(prompt, *args, **kwargs)

    def batch_generate(self, prompts: list[str]) -> list[str]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            responses = list(executor.map(self.generate, prompts))
        return responses

    def get_model_name(self):
        return model

default_model = CustomModel(routed=False)
routed_model = CustomModel(routed=True)


ROWS_LIMIT = int(os.getenv("ROWS_LIMIT", "100"))


def evaluate_benchmark(benchmark, router_benchmark, eval_name):
    print("\n ----- Evaluation of Base model ----- \n")
    benchmark.evaluate(model=default_model)
    print("Evaluation:", eval_name)
    print("Standard Score:", benchmark.overall_score)
    if not benchmark.overall_score or benchmark.overall_score < 0.01:
        raise ValueError(f"Standard model score is too low, aborting evaluation for {eval_name}.")

    print("\n ----- Evaluation with Reasoning Router ----- \n")
    router_benchmark.evaluate(model=routed_model)

    if not benchmark.overall_score:
        print("WARNING: Standard benchmark overall score is None or 0. Setting to 0.0 to avoid errors.")
        benchmark.overall_score = 0.0
    if not router_benchmark.overall_score:
        print("WARNING: Router benchmark overall score is None or 0. Setting to 0.0 to avoid errors.")
        router_benchmark.overall_score = 0.0

    print("-------- Overall Scores: --------")
    print("Evaluation:", eval_name)
    print("Standard Score:", benchmark.overall_score)
    print("Router Score:", router_benchmark.overall_score)

    change = router_benchmark.overall_score - benchmark.overall_score
    print(f"Absolute change: {change:.4f}")
