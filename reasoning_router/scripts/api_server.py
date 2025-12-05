import os
import time
from typing import Optional

from fastapi import FastAPI
import httpx
import openai
from pydantic import BaseModel
import uvicorn

from reasoning_router.router import get_router_response, model
from reasoning_router.utils.llm_utils import get_response


BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL")

model_routed = f"{model}-ROUTED"

app = FastAPI()


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7



@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # Generate response
    if request.model == model:
        response = get_response(request.messages)
    elif request.model == model_routed:
        response = get_router_response(problem=request.messages[-1]["content"])
    else:
        raise openai.NotFoundError(
            f"Unknown model name: {request.model}",
            response=httpx.Response(openai.NotFoundError.status_code),
            body={ "error": { "message": f"Model '{request.model}' not found." } },
        )

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response,
                },
                "finish_reason": "stop",
            }
        ],
    }


@app.get("/v1/models")
def list_models():
    return {"data": [
        {"id": model_routed, "object": "model"},
        {"id": model, "object": "model"},
    ]}


@app.get("/")
def read_root():
    return {"message": "API Server is running."}


if __name__ == "__main__":
    # extract port from BACKEND_BASE_URL
    from urllib.parse import urlparse
    parsed_url = urlparse(BACKEND_BASE_URL)
    port = parsed_url.port or 8000

    uvicorn.run(app, host="0.0.0.0", port=port)
