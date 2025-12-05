# Evaluates Tree-of-Thought on ARC benchmark

from deepeval.benchmarks import ARC
from deepeval.benchmarks.modes import ARCMode

from reasoning_router.utils.eval_utils import evaluate_benchmark, ROWS_LIMIT


benchmark = ARC(n_problems=ROWS_LIMIT, mode=ARCMode.CHALLENGE)
router_benchmark = ARC(n_problems=ROWS_LIMIT, mode=ARCMode.CHALLENGE)


if __name__ == "__main__":
    evaluate_benchmark(benchmark, router_benchmark)
