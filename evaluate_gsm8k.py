from reasoning_router import graph

from datasets import load_dataset
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI

from typing import Any, Dict, List
import json
import os
import re

load_dotenv()

model = os.getenv("OPENAI_MODEL")
if not model:
    raise ValueError("OPENAI_MODEL not set in environment variables")

llm = ChatOpenAI(model=model, temperature=0.7)

# Direct answer prompt for baseline
DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful math assistant. Solve the following math problem and provide the final answer.",
        ),
        ("human", "{problem_base}\n{problem}\n{output_format}"),
    ]
)


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


direct_chain = DIRECT_ANSWER_PROMPT | llm


def evaluate_direct_llm(problem: str) -> Dict[str, Any]:
    """Evaluate a single problem using direct LLM call."""

    response = direct_chain.invoke(
        {
            "problem": problem,
            "problem_base": "Solve this complex math problem:",
            "output_format": "Provide the final answer as a number at the end like #### 4",
        }
    )

    return {
        "method": "direct_llm",
        "problem": problem,
        "predicted_answer": extract_final_answer(str(response.content)),
        "reasoning_steps": [],
        "strategy_used": "direct",
    }


def evaluate_reasoning_router(problem: str) -> Dict[str, Any]:
    """Evaluate a single problem using the reasoning router."""
    config = RunnableConfig(configurable={"thread_id": f"eval_{hash(problem)}"})

    result = graph.invoke(
        {
            "problem": problem,
            "problem_base": "Solve this complex math problem:",
            "output_format": "Provide the final answer as a number at the end like #### 4",
            "strategy": "",
            "reasoning_steps": [],
            "final_answer": "",
            "messages": [],
        },
        config,
    )

    return {
        "method": "reasoning_router",
        "problem": problem,
        "predicted_answer": extract_final_answer(result["final_answer"]),
        "reasoning_steps": result["reasoning_steps"],
        "strategy_used": result["strategy"],
    }


def evaluate_accuracy(predicted: str, actual: str) -> bool:
    """Check if predicted answer matches actual answer."""
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
            direct_result["correct"] = evaluate_accuracy(
                direct_result["predicted_answer"], problem_data["answer"]
            )
            direct_result["actual_answer"] = problem_data["answer"]
            direct_results.append(direct_result)
        except Exception as e:
            print(f"Error with direct LLM: {e}")
            continue
    direct_accuracy = (
        sum(1 for res in direct_results if res["correct"]) / len(direct_results)
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
            reasoning_result["correct"] = evaluate_accuracy(
                reasoning_result["predicted_answer"], problem_data["answer"]
            )
            reasoning_result["actual_answer"] = problem_data["answer"]
            reasoning_results.append(reasoning_result)
        except Exception as e:
            print(f"Error with reasoning router: {e}")
            continue
    reasoning_accuracy = (
        sum(1 for res in reasoning_results if res["correct"]) / len(reasoning_results)
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
    print(f"Accuracy improvement: {improvement_pct:.1f}%")

    # # Strategy analysis
    # print("\nStrategy Usage:")
    # strategy_counts = {}
    # for result in reasoning_results:
    #     strategy = result["strategy_used"]
    #     strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    # for strategy, count in strategy_counts.items():
    #     print(f"  {strategy}: {count} times")

    return direct_results, reasoning_results


if __name__ == "__main__":
    # Run evaluation
    run_evaluation(num_problems=150)
