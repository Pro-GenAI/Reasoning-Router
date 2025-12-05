from deepeval.models.base_model import DeepEvalBaseLLM
import concurrent.futures

from reasoning_router.utils.llm_utils import get_response, model
from reasoning_router.router import get_router_response


class CustomModel(DeepEvalBaseLLM):
    def __init__(self, routed=False):
        self.routed = routed
        self.counter = 0

    def load_model(self):  # type: ignore
        return True

    def generate(self, prompt: str) -> str:
        self.counter += 1
        print(f"Generating response {self.counter}...")
        if self.routed:
            return get_router_response(prompt)
        return get_response(prompt)

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: list[str]) -> list[str]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            responses = list(executor.map(self.generate, prompts))
        return responses

    def get_model_name(self):
        return model

default_model = CustomModel()
routed_model = CustomModel(routed=True)

ROWS_LIMIT = 100


def evaluate_benchmark(benchmark, router_benchmark):
    print("\n ----- Standard Evaluation ----- \n")
    benchmark.evaluate(model=default_model)
    print("\n ----- Router Evaluation ----- \n")
    router_benchmark.evaluate(model=routed_model)

    if not benchmark.overall_score:
        benchmark.overall_score = 0.0
    if not router_benchmark.overall_score:
        router_benchmark.overall_score = 0.0

    print("-------- Overall Scores: --------")
    print("Standard Score:", benchmark.overall_score)
    print("Router Score:", router_benchmark.overall_score)
    change = router_benchmark.overall_score - benchmark.overall_score
    print(f"Improvement: +{change:.2f}%")


# from deepeval.benchmarks.base_benchmark import (
#     DeepEvalBaseBenchmark,
#     DeepEvalBaseBenchmarkResult,
# )
# def evaluate_benchmark(
#     benchmark: DeepEvalBaseBenchmark, router_benchmark: DeepEvalBaseBenchmark
# ):
#     print("\n ----- Standard Evaluation ----- \n")
#     result: DeepEvalBaseBenchmarkResult = benchmark.evaluate(model=default_model)
#     print("\n ----- Router Evaluation ----- \n")
#     router_result: DeepEvalBaseBenchmarkResult = router_benchmark.evaluate(
#         model=routed_model
#     )

#     if not result.overall_accuracy:
#         result.overall_accuracy = 0.0
#     if not router_result.overall_accuracy:
#         router_result.overall_accuracy = 0.0

#     print("-------- Overall Scores: --------")
#     print("Standard Score:", result.overall_accuracy)
#     print("Router Score:", router_result.overall_accuracy)
#     change = router_result.overall_accuracy - result.overall_accuracy
#     print(f"Improvement: +{change:.2f}%")
