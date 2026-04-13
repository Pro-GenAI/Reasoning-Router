from typing import Any, Literal, TypedDict, List
import time

from langchain_core.globals import set_llm_cache
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph, START, END

from reasoning_router.utils.llm_utils import llm, cache
from reasoning_router.utils.strategy_selector import classifier


def _invoke_with_retries(chain, payload: dict, retries: int = 2, delay_s: float = 0.01):
    """Invoke an LLM chain with basic retries for transient network/timeouts."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return chain.invoke(payload)
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            set_llm_cache(None)
            time.sleep(delay_s)
            set_llm_cache(cache)
    raise last_exc  # type: ignore[misc]


# Define the state
class ReasoningState(TypedDict):
    problem: str
    problem_base: str
    output_format: str
    strategy: str
    reasoning_steps: List[str]
    final_answer: str
    messages: List[dict]


# Prompt templates for different reasoning strategies (focus on generating thoughts)
CHAIN_OF_THOUGHT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert problem solver.
Write reasoning notes based on the user's problem.
Think step-by-step and write your thoughts.
Do NOT provide the final answer in this step.""",
        ),
        ("human", "{problem}"),
    ]
)

TREE_OF_THOUGHT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are using Tree-of-Thought reasoning.
Consider multiple possible approaches and branches of reasoning.
Explore different paths, evaluate their feasibility, and choose the best one.
Write short internal notes (max 8 lines). Do NOT provide the final answer.""",
        ),
        ("human", "{problem}"),
    ]
)

SELF_REFLECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Use self-reflection reasoning.
First, note a tentative answer (without committing), then critically examine it.
Identify potential flaws, biases, or missing information.
Write short internal notes (max 8 lines). Do NOT provide the final answer.""",
        ),
        ("human", "{problem}"),
    ]
)

DEBATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Engage in internal debate reasoning.
Present arguments for different perspectives on the problem.
Have a 'pro' and 'con' discussion internally.
Write short internal notes (max 10 lines). Do NOT provide the final answer.""",
        ),
        ("human", "{problem}"),
    ]
)

SIX_HATS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Use the Six Thinking Hats method.
Consider the problem from six different perspectives:
1. White Hat: Facts and information
2. Red Hat: Emotions and feelings
3. Black Hat: Risks and cautions
4. Yellow Hat: Benefits and optimism
5. Green Hat: Creativity and alternatives
6. Blue Hat: Process and overview
Write very short notes (1 line per hat). Do NOT provide the final answer.""",
        ),
        ("human", "{problem}"),
    ]
)

# Create a prompt that uses the reasoning steps to generate final answer
RESPONDER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a synthesizer that takes intermediate reasoning notes and produces the final answer.
Return only the final answer (no extra words, no explanation).
Follow the output format instructions if provided.
""",
        ),
        (
            "human",
            """Original Problem: {problem}

Reasoning Steps:
{steps}

Based on these reasoning steps, provide a clear and concise final answer.

{output_format}
""",
        ),
    ]
)


def router_node(state: ReasoningState) -> ReasoningState:
    """Analyze the problem and select the most appropriate reasoning strategy."""

    problem = state["problem_base"] + state["problem"]

    # Use classifier to predict strategy
    state["strategy"] = classifier.predict(problem)
    return state


def apply_strategy(state: ReasoningState) -> ReasoningState:
    """Apply the selected reasoning strategy to generate thoughts."""
    strategy = state["strategy"]
    problem = state["problem"]

    prompts = {
        "chain_of_thought": CHAIN_OF_THOUGHT_PROMPT,
        "tree_of_thought": TREE_OF_THOUGHT_PROMPT,
        "self_reflection": SELF_REFLECTION_PROMPT,
        "debate": DEBATE_PROMPT,
        "six_hats": SIX_HATS_PROMPT,
    }

    prompt = prompts.get(strategy, CHAIN_OF_THOUGHT_PROMPT)
    chain = prompt | llm

    response = _invoke_with_retries(chain, {"problem": problem})
    if not response.content:
        set_llm_cache(None)
        response = _invoke_with_retries(chain, {"problem": problem})
        set_llm_cache(cache)
    content = str(response.content) if response.content else ""

    # Extract reasoning steps (thoughts)
    state["reasoning_steps"] = [
        line.strip() for line in content.split("\n") if line.strip()
    ]

    return state


def responder_node(state: ReasoningState) -> ReasoningState:
    """Generate the final answer based on the reasoning steps."""
    problem = state["problem"]
    reasoning_steps = state["reasoning_steps"]

    chain = RESPONDER_PROMPT | llm
    steps_text = "\n".join(reasoning_steps)
    response = _invoke_with_retries(chain, {
        "problem": problem,
        "steps": steps_text,
        "output_format": state["output_format"],
    })

    if not response.content:
        set_llm_cache(None)
        response = _invoke_with_retries(chain, {
            "problem": problem,
            "steps": steps_text,
            "output_format": state["output_format"],
        })
        set_llm_cache(cache)
    content = str(response.content) if response.content else ""

    state["final_answer"] = content.strip()
    return state


def route_to_strategy(
    state: ReasoningState,
) -> Literal[
    "chain_of_thought", "tree_of_thought", "self_reflection", "debate", "six_hats"
]:
    """Route to the appropriate strategy node."""
    strategy = state["strategy"]
    if strategy in [
        "chain_of_thought",
        "tree_of_thought",
        "self_reflection",
        "debate",
        "six_hats",
    ]:
        return strategy  # type: ignore
    return "chain_of_thought"


# Build the graph
builder = StateGraph(ReasoningState)  # type: ignore

builder.add_node("router", router_node)
builder.add_node("chain_of_thought", apply_strategy)
builder.add_node("tree_of_thought", apply_strategy)
builder.add_node("self_reflection", apply_strategy)
builder.add_node("debate", apply_strategy)
builder.add_node("six_hats", apply_strategy)
builder.add_node("responder", responder_node)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", route_to_strategy)
builder.add_edge("chain_of_thought", "responder")
builder.add_edge("tree_of_thought", "responder")
builder.add_edge("self_reflection", "responder")
builder.add_edge("debate", "responder")
builder.add_edge("six_hats", "responder")
builder.add_edge("responder", END)

graph = builder.compile()


def get_router_output(
    problem_base: str,
    problem: str,
    output_format: str,
    thread_id: str = "1",
) -> dict[str, Any]:
    """Get the reasoning router response for a given problem."""
    initial_state: ReasoningState = {
        "problem_base": problem_base,
        "problem": problem,
        "output_format": output_format,
        "strategy": "",
        "reasoning_steps": [],
        "final_answer": "",
        "messages": [],
    }

    config = RunnableConfig(configurable={"thread_id": thread_id})
    result = graph.invoke(initial_state, config)
    return result


def get_router_response(
    problem: str,
    problem_base: str = "",
    output_format: str = "",
    thread_id: str = "1",
) -> str:
    response = get_router_output(
        problem_base=problem_base,
        problem=problem,
        output_format=output_format,
        thread_id=thread_id,
    )
    return response["final_answer"]


if __name__ == "__main__":
    # Test the reasoning router
    result1 = get_router_response(
        problem_base="Solve this complex math problem:",
        problem=(
            "If a train travels at 60 mph for 2 hours, "
            "then 40 mph for 3 hours, what is the average speed?"
        ),
        # GSM8K format
        output_format="Provide the final answer as a number at the end like #### 4",
        thread_id="test_thread_1",
    )

    print(f"Final Answer: \n{result1} \n")
