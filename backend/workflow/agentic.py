from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import START, StateGraph
from mlflow.entities import SpanType
from pydantic import BaseModel, Field

from agentic_core import AgentState, build_plan_execute_components
from chat_history.postgres_store import PostgresChatHistoryStore
from observability.tracing import (
    build_langgraph_trace_config,
    configure_mlflow_tracing,
    flush_traces,
    trace_function,
    update_chat_trace_context,
)
from utils import (
    build_completed_steps_payload,
    build_session_response_payload,
    build_session_summary_payload,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
load_dotenv(BACKEND_DIR / '.env')
configure_mlflow_tracing()


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


class AgentTraceResponse(BaseModel):
    final_plan: list[str]
    completed_steps: list[dict[str, str]]
    replan_count: int


class ChatResponse(BaseModel):
    session: SessionResponse
    assistant_message: dict[str, Any]
    trace: AgentTraceResponse


store = PostgresChatHistoryStore()
components = build_plan_execute_components(max_replans=int(os.getenv('AGENTIC_MAX_REPLANS', '3')))


def build_agentic_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node('planner', components.plan_step)
    workflow.add_node('agent', components.execute_step)
    workflow.add_node('replan', components.replan_step)

    workflow.add_edge(START, 'planner')
    workflow.add_conditional_edges('planner', components.route_after_plan, ['agent', '__end__'])
    workflow.add_edge('agent', 'replan')
    workflow.add_conditional_edges('replan', components.route_after_replan, ['agent', '__end__'])

    return workflow.compile()


agentic_graph = build_agentic_graph()

app = FastAPI(title='Agentic Chat Backend', version='0.1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/health')
def health() -> dict[str, Any]:
    return {
        'status': 'ok',
        'planner_model': os.getenv('AGENTIC_PLANNER_MODEL', 'gpt-4.1-mini'),
        'executor_model': os.getenv('AGENTIC_EXECUTOR_MODEL', 'gpt-5-nano'),
        'replanner_model': os.getenv('AGENTIC_REPLANNER_MODEL', 'gpt-4.1-mini'),
        **store.health(),
        'has_openai_key': bool(os.getenv('OPENAI_API_KEY')),
        'has_tavily_key': bool(os.getenv('TAVILY_API_KEY')),
    }


@app.get('/sessions', response_model=list[SessionSummary])
def list_sessions() -> list[SessionSummary]:
    return [SessionSummary(**build_session_summary_payload(record)) for record in store.list_sessions()]


@app.post('/sessions', response_model=SessionResponse)
def create_session(request: CreateSessionRequest) -> SessionResponse:
    record = store.create_session(
        title=request.title,
        external_user_id=request.external_user_id,
        display_name=request.user_display_name,
    )
    return SessionResponse(**build_session_response_payload(record))


@app.get('/sessions/{session_id}', response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    try:
        record = store.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='Session not found') from exc
    return SessionResponse(**build_session_response_payload(record))


@app.post('/sessions/{session_id}/messages', response_model=ChatResponse)
@trace_function(
    name='agentic_chat_request',
    span_type=SpanType.CHAIN,
    attributes={'workflow': 'agentic', 'endpoint': '/sessions/{session_id}/messages'},
)
async def chat(session_id: str, request: ChatRequest) -> ChatResponse:
    try:
        record = store.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='Session not found') from exc

    user_message = store.append_message(
        session_id=session_id,
        user_id=record.user.user_id,
        role='user',
        content=request.message.strip(),
        metadata={'source': 'api', 'mode': 'agentic-plan-execute'},
    )

    history = [*record.messages, user_message]
    update_chat_trace_context(
        workflow_name='agentic',
        session_id=session_id,
        external_user_id=record.user.external_user_id,
        user_display_name=record.user.display_name,
        request_text=request.message.strip(),
        extra_tags={
            'workflow_mode': 'plan_execute',
            'message_count_before': len(record.messages),
        },
    )

    try:
        result = await agentic_graph.ainvoke(
            {
                'input': request.message.strip(),
                'history': history,
                'plan': [],
                'past_steps': [],
                'response': '',
                'last_error': None,
                'replan_count': 0,
            },
            config=build_langgraph_trace_config(
                session_id=session_id,
                recursion_limit=20,
                workflow_name='agentic',
                external_user_id=record.user.external_user_id,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Agentic flow failed: {exc}') from exc

    assistant_text = (result.get('response') or '').strip()
    if not assistant_text:
        completed_steps = result.get('past_steps', [])
        if completed_steps:
            assistant_text = completed_steps[-1][1]
        else:
            assistant_text = 'I could not produce a final answer.'

    completed_steps = build_completed_steps_payload(result.get('past_steps', []))
    assistant_metadata = {
        'source': 'agentic-plan-execute',
        'final_plan': result.get('plan', []),
        'completed_steps': completed_steps,
        'replan_count': result.get('replan_count', 0),
    }
    if result.get('last_error'):
        assistant_metadata['last_error'] = result['last_error']

    assistant_message = store.append_message(
        session_id=session_id,
        user_id=None,
        role='assistant',
        content=assistant_text,
        model_name=os.getenv('AGENTIC_EXECUTOR_MODEL', 'gpt-5-nano'),
        metadata=assistant_metadata,
    )

    saved = store.get_session(session_id)
    update_chat_trace_context(
        workflow_name='agentic',
        session_id=session_id,
        external_user_id=record.user.external_user_id,
        user_display_name=record.user.display_name,
        request_text=request.message.strip(),
        response_text=assistant_text,
        extra_tags={
            'workflow_mode': 'plan_execute',
            'completed_steps_count': len(completed_steps),
            'replan_count': result.get('replan_count', 0),
        },
    )
    flush_traces()
    trace = AgentTraceResponse(
        final_plan=result.get('plan', []),
        completed_steps=completed_steps,
        replan_count=result.get('replan_count', 0),
    )
    return ChatResponse(
        session=SessionResponse(**build_session_response_payload(saved)),
        assistant_message=assistant_message,
        trace=trace,
    )