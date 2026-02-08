# Evaluates Chain-of-Thought on GSM8K benchmark

from deepeval.benchmarks import GSM8K

from reasoning_router.utils.eval_utils import evaluate_benchmark, ROWS_LIMIT


benchmark = GSM8K(n_problems=ROWS_LIMIT)
router_benchmark = GSM8K(n_problems=ROWS_LIMIT)


def main():
    evaluate_benchmark(benchmark, router_benchmark, eval_name="GSM8K")


if __name__ == "__main__":
    main()
