# Evaluates Debate on ARC benchmark

from deepeval.benchmarks import LogiQA
from deepeval.benchmarks.logi_qa.task import LogiQATask

from reasoning_router.utils.eval_utils import evaluate_benchmark, ROWS_LIMIT


tasks = [
    LogiQATask.SUFFICIENT_CONDITIONAL_REASONING,
    # LogiQATask.NECESSARY_CONDITIONAL_REASONING,
]

ROWS_LIMIT_PER_TASK = ROWS_LIMIT // len(tasks)

benchmark = LogiQA(tasks=tasks, n_problems_per_task=ROWS_LIMIT_PER_TASK)
router_benchmark = LogiQA(tasks=tasks, n_problems_per_task=ROWS_LIMIT_PER_TASK)


if __name__ == "__main__":
    evaluate_benchmark(benchmark, router_benchmark)
