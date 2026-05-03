from __future__ import annotations

from dataclasses import asdict
import json
import logging
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
from config import get_plan_execute_config
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


def _configure_application_logging() -> None:
    level_name = os.getenv('AGENTIC_LOG_LEVEL', 'INFO').upper()
    level = getattr(logging, level_name, logging.INFO)
    log_dir = BACKEND_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(os.getenv('AGENTIC_LOG_FILE', str(log_dir / 'agentic.log')))

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    for logger_name in ('workflow', 'agentic_core'):
        package_logger = logging.getLogger(logger_name)
        package_logger.setLevel(level)
        package_logger.propagate = False

        existing_handler_keys = {
            (type(handler), getattr(handler, 'baseFilename', None))
            for handler in package_logger.handlers
        }
        desired_handlers = [console_handler, file_handler]
        for handler in desired_handlers:
            handler_key = (type(handler), getattr(handler, 'baseFilename', None))
            if handler_key not in existing_handler_keys:
                package_logger.addHandler(handler)


_configure_application_logging()
configure_mlflow_tracing()


LOGGER = logging.getLogger(__name__)
PLAN_EXECUTE_CONFIG = get_plan_execute_config()


def _workflow_state_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    return {
        'input_preview': str(state.get('input', ''))[:160],
        'goal': state.get('goal', {}),
        'current_step': state.get('execute_track', {}).get('current_step', {}),
        'control': state.get('control', {}),
        'plans_count': len(state.get('plans', [])),
        'execution_log_count': len(state.get('execute_track', {}).get('track', [])),
        'replan_history_count': len(state.get('replan_history', [])),
        'has_response': bool(state.get('response')),
        'last_error': state.get('last_error'),
    }


def _log_workflow_event(event: str, payload: dict[str, Any]) -> None:
    LOGGER.info('workflow_event %s', json.dumps({'event': event, **payload}, default=str))


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
    goal: dict[str, Any]
    plans: list[dict[str, Any]]
    completed_steps: list[dict[str, Any]]
    execute_track: dict[str, Any]
    control: dict[str, Any]
    replan_history: list[dict[str, Any]]
    last_error: str | None = None


class ChatResponse(BaseModel):
    session: SessionResponse
    assistant_message: dict[str, Any]
    trace: AgentTraceResponse


store = PostgresChatHistoryStore()
components = build_plan_execute_components(max_replans=PLAN_EXECUTE_CONFIG.flow.max_replans)


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
        'planner_model': PLAN_EXECUTE_CONFIG.planner_model.name,
        'executor_model': PLAN_EXECUTE_CONFIG.executor_model.name,
        'replanner_model': PLAN_EXECUTE_CONFIG.replanner_model.name,
        'reasoning_effort': PLAN_EXECUTE_CONFIG.executor_model.reasoning_effort,
        'plan_execute_config': asdict(PLAN_EXECUTE_CONFIG),
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

    initial_state = {
        'input': request.message.strip(),
        'history': history,
        'goal': {},
        'plans': [],
        'execute_track': {
            'current_step': {'plan_version': None, 'step_id': None},
            'track': [],
        },
        'control': {
            'step_count': 0,
            'replan_time': 0,
            'status': 'running',
        },
        'replan_history': [],
        'response': '',
        'last_error': None,
    }
    _log_workflow_event(
        'chat_invoke_start',
        {
            'session_id': session_id,
            'request_text': request.message.strip(),
            'state': _workflow_state_snapshot(initial_state),
        },
    )

    try:
        result = await agentic_graph.ainvoke(
            initial_state,
            config=build_langgraph_trace_config(
                session_id=session_id,
                recursion_limit=PLAN_EXECUTE_CONFIG.flow.graph_recursion_limit,
                workflow_name='agentic',
                external_user_id=record.user.external_user_id,
            ),
        )
    except Exception as exc:
        _log_workflow_event(
            'chat_invoke_error',
            {
                'session_id': session_id,
                'error': str(exc),
            },
        )
        raise HTTPException(status_code=500, detail=f'Agentic flow failed: {exc}') from exc

    _log_workflow_event(
        'chat_invoke_done',
        {
            'session_id': session_id,
            'state': _workflow_state_snapshot(result),
        },
    )

    assistant_text = (result.get('response') or '').strip()
    if not assistant_text:
        execution_logs = result.get('execute_track', {}).get('track', [])
        if execution_logs:
            assistant_text = execution_logs[-1].get('output', '')
        else:
            assistant_text = 'I could not produce a final answer.'

    completed_steps = build_completed_steps_payload(result.get('plans', []))
    assistant_metadata = {
        'source': 'agentic-plan-execute',
        'goal': result.get('goal', {}),
        'plans': result.get('plans', []),
        'completed_steps': completed_steps,
        'execute_track': result.get('execute_track', {}),
        'control': result.get('control', {}),
        'replan_history': result.get('replan_history', []),
    }
    if result.get('last_error'):
        assistant_metadata['last_error'] = result['last_error']

    assistant_message = store.append_message(
        session_id=session_id,
        user_id=None,
        role='assistant',
        content=assistant_text,
        model_name=PLAN_EXECUTE_CONFIG.executor_model.name,
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
            'replan_count': result.get('control', {}).get('replan_time', 0),
        },
    )
    flush_traces()
    trace = AgentTraceResponse(
        goal=result.get('goal', {}),
        plans=result.get('plans', []),
        completed_steps=completed_steps,
        execute_track=result.get('execute_track', {}),
        control=result.get('control', {}),
        replan_history=result.get('replan_history', []),
        last_error=result.get('last_error'),
    )
    return ChatResponse(
        session=SessionResponse(**build_session_response_payload(saved)),
        assistant_message=assistant_message,
        trace=trace,
    )