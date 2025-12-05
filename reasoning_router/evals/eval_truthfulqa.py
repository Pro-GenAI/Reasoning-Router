# Evaluates Self-Reflection on TruthfulQA benchmark

from deepeval.benchmarks import TruthfulQA
from deepeval.benchmarks.truthful_qa.task import TruthfulQATask
from deepeval.benchmarks.modes import TruthfulQAMode

from reasoning_router.utils.eval_utils import evaluate_benchmark, ROWS_LIMIT


# Define benchmark with specific tasks and shots
tasks = [TruthfulQATask.LOGICAL_FALSEHOOD, TruthfulQATask.MISCONCEPTIONS,
         TruthfulQATask.PSYCHOLOGY, TruthfulQATask.SUBJECTIVE]

ROWS_LIMIT_PER_TASK = ROWS_LIMIT // len(tasks)

benchmark = TruthfulQA(
    tasks=tasks, mode=TruthfulQAMode.MC2, n_problems_per_task=ROWS_LIMIT_PER_TASK
)
router_benchmark = TruthfulQA(
    tasks=tasks, mode=TruthfulQAMode.MC2, n_problems_per_task=ROWS_LIMIT_PER_TASK
)


if __name__ == "__main__":
    evaluate_benchmark(benchmark, router_benchmark)
