from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import mlflow
from dotenv import load_dotenv
from mlflow.entities import SpanType
from mlflow.entities.trace_location import MlflowExperimentLocation


TRACE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TRACE_DIR.parent.parent
load_dotenv(BACKEND_DIR / '.env')


@dataclass(frozen=True)
class MlflowTracingConfig:
    enabled: bool
    tracking_uri: str
    experiment_name: str
    run_tracer_inline: bool
    langchain_autolog_enabled: bool


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


@lru_cache(maxsize=1)
def get_tracing_config() -> MlflowTracingConfig:
    return MlflowTracingConfig(
        enabled=_env_bool('MLFLOW_TRACING_ENABLED', True),
        tracking_uri=os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000').strip(),
        experiment_name=os.getenv('MLFLOW_EXPERIMENT_NAME', 'agent_test').strip(),
        run_tracer_inline=_env_bool('MLFLOW_LANGCHAIN_RUN_TRACER_INLINE', True),
        langchain_autolog_enabled=_env_bool('MLFLOW_LANGCHAIN_AUTOLOG_ENABLED', True),
    )


@lru_cache(maxsize=1)
def configure_mlflow_tracing() -> MlflowTracingConfig:
    config = get_tracing_config()
    if not config.enabled:
        return config

    mlflow.set_tracking_uri(config.tracking_uri)
    experiment = mlflow.set_experiment(config.experiment_name)
    mlflow.tracing.enable()
    mlflow.tracing.set_destination(
        MlflowExperimentLocation(experiment_id=experiment.experiment_id)
    )
    mlflow.langchain.autolog(
        log_traces=True,
        run_tracer_inline=config.run_tracer_inline,
        silent=True,
        disable=not config.langchain_autolog_enabled,
    )
    return config


def trace_function(
    *,
    name: str | None = None,
    span_type: SpanType | str = SpanType.CHAIN,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    config = configure_mlflow_tracing()
    if not config.enabled:
        return lambda func: func
    return mlflow.trace(name=name, span_type=span_type, attributes=attributes or {})


def set_current_span_attributes(attributes: dict[str, Any]) -> None:
    span = mlflow.get_current_active_span()
    if span is not None and attributes:
        span.set_attributes(attributes)


def set_current_span_outputs(outputs: Any) -> None:
    span = mlflow.get_current_active_span()
    if span is not None:
        span.set_outputs(outputs)


def update_chat_trace_context(
    *,
    workflow_name: str,
    session_id: str,
    external_user_id: str,
    user_display_name: str,
    request_text: str,
    response_text: str | None = None,
    extra_tags: dict[str, Any] | None = None,
) -> None:
    tags = {
        'workflow': workflow_name,
        'session_id': session_id,
        'external_user_id': external_user_id,
        'user_display_name': user_display_name,
    }
    if extra_tags:
        tags.update(extra_tags)

    mlflow.update_current_trace(
        tags=tags,
        metadata={
            'mlflow.trace.session': session_id,
            'mlflow.trace.user': external_user_id,
        },
        request_preview=request_text[:200],
        response_preview=(response_text or '')[:200] or None,
    )


def flush_traces() -> None:
    config = get_tracing_config()
    if not config.enabled:
        return
    mlflow.flush_trace_async_logging()


def build_langgraph_trace_config(
    *,
    session_id: str,
    recursion_limit: int,
    workflow_name: str,
    external_user_id: str,
) -> dict[str, Any]:
    return {
        'recursion_limit': recursion_limit,
        'configurable': {
            'thread_id': session_id,
        },
        'metadata': {
            'workflow': workflow_name,
            'session_id': session_id,
            'external_user_id': external_user_id,
        },
    }
