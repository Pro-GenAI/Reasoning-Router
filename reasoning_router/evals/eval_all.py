"""Run all benchmark evals in this directory.

This script sequentially runs the existing per-benchmark eval entrypoints.
"""

from __future__ import annotations

import traceback

from reasoning_router.evals import eval_arc, eval_gsm8k, eval_logiqa, eval_truthfulqa


EVALS = {
    "GSM8K": eval_gsm8k.main,
    "ARC": eval_arc.main,
    "LogiQA": eval_logiqa.main,
    "TruthfulQA": eval_truthfulqa.main,
}


def main() -> None:
    failures: list[tuple[str, BaseException]] = []

    for name, fn in EVALS.items():
        print("\n" + "=" * 60)
        print(f"Running {name}...")
        print("=" * 60 + "\n")

        try:
            fn()
        except BaseException as exc:  # keep summary even for unexpected failures
            failures.append((name, exc))
            print(f"\nERROR: {name} failed with {type(exc).__name__}: {exc}\n")
            traceback.print_exc()

    if failures:
        print("\n" + "=" * 60)
        print("Completed with failures:")
        for name, exc in failures:
            print(f"- {name}: {type(exc).__name__}: {exc}")
        print("=" * 60 + "\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
