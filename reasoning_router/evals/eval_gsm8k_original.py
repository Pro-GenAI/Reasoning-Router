# Evaluates CoT reasoning on GSM8K dataset
# Does not use DeepEval framework for evaluation

import json
import os
import re
from typing import Any, Dict, List

from datasets import load_dataset
from langchain_community.cache import SQLiteCache
from langchain_core.caches import RETURN_VAL_TYPE
from langchain_core.globals import set_llm_cache
from langchain_core.prompts import ChatPromptTemplate

from reasoning_router.router import get_router_output
from reasoning_router.utils.llm_utils import database_path, llm


class ValidatingSQLiteCache(SQLiteCache):
    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        # Validate before inserting
        if not extract_final_answer(llm_string):
            raise ValueError("Invalid response, not caching.")

        print(f"Caching valid response for prompt {prompt[:30]}...")
        super().update(prompt, llm_string, return_val)


validated_cache = ValidatingSQLiteCache(database_path=database_path)
set_llm_cache(validated_cache)


def load_gsm8k_dataset(limit: int, split: str = "test") -> List[Dict[str, Any]]:
    """Load GSM8K dataset and return a list of problems."""
    cache_file = f"gsm8k_cache_{split}.json"
    problems = []
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            problems = json.load(f)
        problems = problems[:limit]
    if len(problems) < limit:
        dataset = load_dataset("gsm8k", "main", split=split)
        problems = []
        for i, item in enumerate(dataset):
            if i >= limit:
                break
            problems.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "problem_id": i,
                }
            )
        with open(cache_file, "w") as f:
            json.dump(problems, f)
    return problems


def extract_final_answer(text: str) -> float | str | None:
    """Extract the final numerical answer from model response."""
    # Look for patterns like "#### 42" or "42" at the end
    lines = text.strip().split("\n")
    line = lines[-1].strip()
    if "####" in line:
        num = line.split("####")[-1].strip()
    else:
        # Return the last number in the line (might include commas and decimal points)
        match = re.findall(r"[-+]?\d[\d,]*\.?\d*", line)
        if match:
            num = match[-1]
        else:
            return None

    num = num.replace(",", "").rstrip(".00").rstrip(".0")
    num = num.rstrip(".").strip("*")
    print("\t -", num)
    # try:
    #     return float(num)
    # except ValueError:
    #     return num
    return num


# Direct answer prompt for baseline
DIRECT_ANSWER_PROMPT_GSM8K = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful math assistant. Solve the following math problem and provide the final answer.",
        ),
        ("human", "{problem_base}\n{problem}\n{output_format}"),
    ]
)

direct_chain = DIRECT_ANSWER_PROMPT_GSM8K | llm


def evaluate_direct_llm(problem: str) -> Dict[str, Any]:
    """Evaluate a single problem using direct LLM call."""

    response = direct_chain.invoke(
        {
            "problem": problem,
            "problem_base": "Solve this complex math problem:",
            "output_format": "Provide the final answer as a number at the end like #### 4",
        }
    )
    final_answer = extract_final_answer(str(response.content))
    if not final_answer:
        print("Could not extract final answer from direct LLM response.")

    return {
        "method": "direct_llm",
        "problem": problem,
        "predicted_answer": final_answer,
        "reasoning_steps": [],
        "strategy_used": "direct",
    }


def evaluate_reasoning_router(problem: str) -> Dict[str, Any]:
    """Evaluate a single problem using the reasoning router."""
    result = get_router_output(
        problem_base="Solve this complex math problem:",
        problem=problem,
        output_format="Provide the final answer as a number at the end like #### 4",
        thread_id=f"eval_{hash(problem)}",
    )
    final_answer = extract_final_answer(result["final_answer"])
    if final_answer is None:
        raise ValueError(
            "Could not extract final answer from reasoning router response."
        )

    return {
        "method": "reasoning_router",
        "problem": problem,
        "predicted_answer": final_answer,
        "reasoning_steps": result["reasoning_steps"],
        "strategy_used": result["strategy"],
    }


def evaluate_accuracy(predicted: str, actual: str) -> bool:
    """Check if predicted answer matches actual answer."""
    if predicted is None:
        raise ValueError("Predicted answer is None.")
    if actual is None:
        raise ValueError("Actual answer is None.")

    # Extract numbers from both
    pred_nums = re.findall(r"\d+", predicted)
    actual_nums = re.findall(r"\d+", actual)
    if not pred_nums or not actual_nums:
        return False

    # Compare the last numbers found
    return pred_nums[-1] == actual_nums[-1]


def run_evaluation(num_problems: int):
    """Run full evaluation comparing reasoning router vs direct LLM."""

    print(f"Loading GSM8K dataset ({num_problems} problems)...")
    problems = load_gsm8k_dataset(limit=num_problems)

    print("Evaluating with Direct LLM...")
    direct_results = []
    for i, problem_data in enumerate(problems):
        print(f"Evaluating problem {i+1}/{len(problems)}")
        # Evaluate with direct LLM
        try:
            direct_result = evaluate_direct_llm(problem_data["question"])
            direct_result["is_correct"] = evaluate_accuracy(
                direct_result["predicted_answer"], problem_data["answer"]
            )
            direct_result["actual_answer"] = problem_data["answer"]
            direct_results.append(direct_result)
        except Exception as e:
            print(f"Error with direct LLM: {e}")
            continue
    direct_accuracy = (
        sum(1 for res in direct_results if res["is_correct"]) / len(direct_results)
        if direct_results
        else 0
    )

    print("Evaluating with Reasoning Router...")
    reasoning_results = []
    for i, problem_data in enumerate(problems):
        print(f"Evaluating problem {i+1}/{len(problems)}")
        # Evaluate with reasoning router
        try:
            reasoning_result = evaluate_reasoning_router(problem_data["question"])
            reasoning_result["is_correct"] = evaluate_accuracy(
                reasoning_result["predicted_answer"], problem_data["answer"]
            )
            reasoning_result["actual_answer"] = problem_data["answer"]
            reasoning_results.append(reasoning_result)
        except Exception as e:
            print(f"Error with reasoning router: {e}")
            continue
    reasoning_accuracy = (
        sum(1 for res in reasoning_results if res["is_correct"]) / len(reasoning_results)
        if reasoning_results
        else 0
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Problems evaluated: {len(problems)}")
    # Display percentages rather than fraction values
    improvement_pct = (reasoning_accuracy - direct_accuracy) * 100
    print(f"Direct LLM accuracy: {direct_accuracy * 100:.1f}%")
    print(f"Reasoning accuracy: {reasoning_accuracy * 100:.1f}%")
    if improvement_pct >= 0:
        print(f"Accuracy improvement: +{improvement_pct:.1f}%")
    else:
        print(f"Accuracy DECREASE: {improvement_pct:.1f}%")

    # Strategy analysis
    print("\nStrategy Usage:")
    strategy_counts = {}
    for result in reasoning_results:
        strategy = result["strategy_used"]
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    for strategy, count in strategy_counts.items():
        print(f"  {strategy}: {count} times")

    return direct_results, reasoning_results


if __name__ == "__main__":
    # Run evaluation
    run_evaluation(num_problems=100)
