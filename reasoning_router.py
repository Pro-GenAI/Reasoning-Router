from langgraph.graph import StateGraph, START, END
from langchain_core.globals import set_llm_cache
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_community.cache import SQLiteCache
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from typing import Literal, TypedDict, List
import os

load_dotenv()

model = os.getenv("OPENAI_MODEL")
if not model:
    raise ValueError("OPENAI_MODEL not set in environment variables")

# Set up persistent caching for LLM responses using SQLite
cache = SQLiteCache(database_path="langchain_cache.db")
set_llm_cache(cache)

llm = ChatOpenAI(model=model, temperature=0.7)


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
CHAIN_OF_THOUGHT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert problem solver using Chain-of-Thought reasoning.
Break down the problem step by step, showing your reasoning process clearly.
For each step, explain your thought process and how you arrived at that conclusion.
Generate detailed reasoning steps without providing the final answer."""),
    ("human", "{problem}")
])

TREE_OF_THOUGHT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are using Tree-of-Thought reasoning.
Consider multiple possible approaches and branches of reasoning.
Explore different paths, evaluate their feasibility, and choose the best one.
Show the tree structure of your thinking as detailed reasoning steps."""),
    ("human", "{problem}")
])

SELF_REFLECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Use self-reflection reasoning.
First, provide an initial answer, then critically examine your own reasoning.
Identify potential flaws, biases, or missing information.
Generate detailed thoughts about the reflection process."""),
    ("human", "{problem}")
])

DEBATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Engage in internal debate reasoning.
Present arguments for different perspectives on the problem.
Have a 'pro' and 'con' discussion internally.
Generate detailed thoughts from the debate without final synthesis."""),
    ("human", "{problem}")
])

SIX_HATS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Use the Six Thinking Hats method.
Consider the problem from six different perspectives:
1. White Hat: Facts and information
2. Red Hat: Emotions and feelings
3. Black Hat: Risks and cautions
4. Yellow Hat: Benefits and optimism
5. Green Hat: Creativity and alternatives
6. Blue Hat: Process and overview
Generate detailed thoughts from each hat's perspective."""),
    ("human", "{problem}")
])

# Create a prompt that uses the reasoning steps to generate final answer
RESPONDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a synthesizer that takes detailed reasoning thoughts and generates a clear, concise final answer.
Based on the reasoning steps provided, synthesize the information and provide the final answer to the original problem."""),
    ("human", """Original Problem: {problem}

Reasoning Steps:
{steps}

Based on these reasoning steps, provide a clear and concise final answer.

{output_format}
""")
])


def router_node(state: ReasoningState) -> ReasoningState:
    """Analyze the problem and select the most appropriate reasoning strategy."""

    problem = state["problem_base"] + state["problem"]

    # Simple heuristic-based routing (could be enhanced with LLM)
    if "complex" in problem.lower() or "multiple" in problem.lower():
        strategy = "tree_of_thought"
    elif "reflect" in problem.lower() or "critique" in problem.lower():
        strategy = "self_reflection"
    elif "debate" in problem.lower() or "arguments" in problem.lower():
        strategy = "debate"
    elif "perspectives" in problem.lower() or "hats" in problem.lower():
        strategy = "six_hats"
    else:
        strategy = "chain_of_thought"

    state["strategy"] = strategy
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
        "six_hats": SIX_HATS_PROMPT
    }

    prompt = prompts.get(strategy, CHAIN_OF_THOUGHT_PROMPT)
    chain = prompt | llm

    response = chain.invoke({"problem": problem})
    content = str(response.content)  # Ensure it's a string

    # Extract reasoning steps (thoughts)
    state["reasoning_steps"] = [line.strip() for line in content.split('\n') if line.strip()]

    return state

def responder_node(state: ReasoningState) -> ReasoningState:
    """Generate the final answer based on the reasoning steps."""
    problem = state["problem"]
    reasoning_steps = state["reasoning_steps"]

    chain = RESPONDER_PROMPT | llm
    steps_text = "\n".join(reasoning_steps)
    response = chain.invoke({"problem": problem, "steps": steps_text,
                             "output_format": state["output_format"]})

    state["final_answer"] = str(response.content).strip()
    return state

def route_to_strategy(state: ReasoningState) -> Literal["chain_of_thought", "tree_of_thought", "self_reflection", "debate", "six_hats"]:
    """Route to the appropriate strategy node."""
    strategy = state["strategy"]
    if strategy in ["chain_of_thought", "tree_of_thought", "self_reflection", "debate", "six_hats"]:
        return strategy  # type: ignore
    return "chain_of_thought"


# Build the graph
builder = StateGraph(ReasoningState) # type: ignore

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


if __name__ == "__main__":
    # Test the reasoning router
    test_problem = "If a train travels at 60 mph for 2 hours, then 40 mph for 3 hours, what is the average speed?"
    config = RunnableConfig(configurable={"thread_id": "1"})

    result1 = graph.invoke({
        "problem": test_problem,
        "problem_base": "Solve this complex math problem:",
        "output_format": "Provide the final answer as a number at the end like #### 4",
        "strategy": "",
        "reasoning_steps": [],
        "final_answer": "",
        "messages": []
    }, config)

    print()
    print(f"Problem: \n{result1['problem']} \n")
    print(f"Selected Strategy: \n{result1['strategy']} \n")
    print(f"Reasoning Steps: \n{result1['reasoning_steps']} \n")
    print(f"Final Answer: \n{result1['final_answer']} \n")
