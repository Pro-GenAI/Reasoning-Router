import os
import time

import openai
from dotenv import load_dotenv
from langchain_community.cache import SQLiteCache
from langchain_openai import ChatOpenAI
import numpy as np


load_dotenv()

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "")
if not EMBED_MODEL_NAME:
    raise ValueError("EMBED_MODEL_NAME environment variable not set.")

embed_client = openai.OpenAI(
    base_url=os.getenv("EMBEDDING_BASE_URL"),
)

def encode(texts: list[str] | str) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    responses = embed_client.embeddings.create(
        model=EMBED_MODEL_NAME,
        input=texts,
    )
    return np.array([data.embedding for data in responses.data])



model = os.getenv("OPENAI_MODEL", "")
if not model:
    raise ValueError("OPENAI_MODEL not set in environment variables")

# Set up persistent caching for LLM responses using SQLite
database_path = ".langchain_cache.db"
cache = SQLiteCache(database_path=database_path)

llm = ChatOpenAI(
    model=model,
    temperature=0.7,
    cache=cache,
    max_completion_tokens=250,
    timeout=30,
)


def get_response(messages: str | list[dict[str, str]]) -> str:
    """Simple response generator for demo purposes.

    If the model's response contains a JSON object with an `action` key, we will
    route it to the local action classifier and return a short classification
    summary. Otherwise, return the model's response.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    for attempt in range(1):
        try:
            response = llm.invoke(messages)
            response = response.content
            if isinstance(response, list):
                response = " ".join([str(r) for r in response])
            return response
        except openai.BadRequestError as ex:
            print(f"Error getting response (attempt {attempt + 1}): {ex}")
            time.sleep(1)
    return "Error: Unable to get response."


if __name__ == "__main__":
	response = get_response("Do you know the capital of France?")
	print("Response:", response)
