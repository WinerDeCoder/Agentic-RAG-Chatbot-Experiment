from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

from chat_history.postgres_store import PostgresChatHistoryStore, SessionRecord


MODEL_NAME = "gpt-5-nano"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
load_dotenv(".env")


class CreateSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    external_user_id: str | None = None
    user_display_name: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class SessionSummary(BaseModel):
    session_id: str
    title: str
    user_id: str
    external_user_id: str
    user_display_name: str
    created_at: str
    updated_at: str
    message_count: int


class SessionResponse(BaseModel):
    session_id: str
    title: str
    user: dict[str, Any]
    created_at: str
    updated_at: str
    messages: list[dict[str, Any]]
    message_count: int


class ChatResponse(BaseModel):
    session: SessionResponse
    assistant_message: dict[str, Any]


store = PostgresChatHistoryStore()
app = FastAPI(title="Simple Chat Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def session_summary(record: SessionRecord) -> SessionSummary:
    return SessionSummary(
        session_id=record.session_id,
        title=record.title,
        user_id=record.user.user_id,
        external_user_id=record.user.external_user_id,
        user_display_name=record.user.display_name,
        created_at=record.created_at,
        updated_at=record.updated_at,
        message_count=record.message_count,
    )


def session_response(record: SessionRecord) -> SessionResponse:
    return SessionResponse(**record.to_dict())


def build_messages(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    system_prompt = (
        "You are a concise helpful assistant for a simple chat application. "
        "Answer directly and keep context from the conversation history."
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return messages


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before calling the chat endpoint")
    return OpenAI(api_key=api_key)


def generate_assistant_reply(history: list[dict[str, Any]]) -> str:
    response = get_openai_client().responses.create(model=MODEL_NAME, input=build_messages(history))
    text = getattr(response, "output_text", "") or ""
    if text:
        return text.strip()

    output = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                output.append(value)
    return "\n".join(part.strip() for part in output if part.strip())


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        **store.health(),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.get("/sessions", response_model=list[SessionSummary])
def list_sessions() -> list[SessionSummary]:
    return [session_summary(record) for record in store.list_sessions()]


@app.post("/sessions", response_model=SessionResponse)
def create_session(request: CreateSessionRequest) -> SessionResponse:
    record = store.create_session(
        title=request.title,
        external_user_id=request.external_user_id,
        display_name=request.user_display_name,
    )
    return session_response(record)


@app.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    try:
        record = store.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    return session_response(record)


@app.post("/sessions/{session_id}/messages", response_model=ChatResponse)
def chat(session_id: str, request: ChatRequest) -> ChatResponse:
    try:
        record = store.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc

    user_message = store.append_message(
        session_id=session_id,
        user_id=record.user.user_id,
        role="user",
        content=request.message.strip(),
        metadata={"source": "api"},
    )
    record.messages.append(user_message)

    try:
        assistant_text = generate_assistant_reply(record.messages)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {exc}") from exc

    assistant_message = store.append_message(
        session_id=session_id,
        user_id=None,
        role="assistant",
        content=assistant_text,
        model_name=MODEL_NAME,
        metadata={"source": "openai-responses"},
    )
    saved = store.get_session(session_id)
    return ChatResponse(session=session_response(saved), assistant_message=assistant_message)
