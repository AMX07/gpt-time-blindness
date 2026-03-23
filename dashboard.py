"""Web dashboard with live chat, eval results, and conversation history."""

import os
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from prompts import get_system_prompt

app = FastAPI(title="Time-Aware Chatbot Dashboard")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

BEDROCK_MODELS = {
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
    "claude-sonnet-4-6": "arn:aws:bedrock:us-east-1:744741211997:inference-profile/us.anthropic.claude-sonnet-4-6",
    "claude-haiku-4-5-20251001": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}

THINKING_BUDGET = 10000

# Lazy-init client
_client = None
_use_bedrock = None


def _get_client():
    global _client, _use_bedrock
    if _client is None:
        has_aws = bool(os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
        if has_aws:
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            _client = anthropic.AnthropicBedrock(aws_region=region)
            _use_bedrock = True
        else:
            _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            _use_bedrock = False
    return _client, _use_bedrock


def _get_model_id(model: str) -> str:
    _, use_bedrock = _get_client()
    if not use_bedrock:
        return model
    return BEDROCK_MODELS.get(model, f"us.anthropic.{model}-v1:0")


class ChatRequest(BaseModel):
    messages: list[dict]
    model: str = "claude-opus-4-6"
    use_timestamps: bool = True
    use_thinking: bool = True


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    client, _ = _get_client()
    model_id = _get_model_id(req.model)
    system = get_system_prompt(req.use_timestamps)

    kwargs = {
        "model": model_id,
        "max_tokens": 16384,
        "system": system,
        "messages": req.messages,
    }

    if req.use_thinking:
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": THINKING_BUDGET,
        }

    response = client.messages.create(**kwargs)

    # Extract text, skip thinking blocks
    text = ""
    for block in response.content:
        if block.type == "text":
            text = block.text
            break

    return {"response": text}


@app.get("/api/evals")
async def list_evals():
    try:
        from db import EvalStore
        return EvalStore().list_eval_runs()
    except Exception:
        return []


@app.get("/api/evals/{eval_id}")
async def get_eval(eval_id: str):
    from db import EvalStore
    return EvalStore().get_eval_run(eval_id)


@app.get("/api/conversations")
async def list_conversations():
    try:
        from db import ConversationStore
        return ConversationStore().list_sessions()
    except Exception:
        return []


@app.get("/api/conversations/{session_id}")
async def get_conversation(session_id: str):
    from db import ConversationStore
    return ConversationStore().get_session(session_id)


if __name__ == "__main__":
    from env import load_env
    load_env()

    port = int(os.environ.get("PORT", 8000))
    print(f"Starting dashboard at http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
