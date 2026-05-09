from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_PLAN_EXECUTE_CONFIG_PATH = CONFIG_DIR / 'plan_execute.yaml'


@dataclass(frozen=True)
class ModelSettings:
    name: str
    temperature: float
    reasoning_effort: str


@dataclass(frozen=True)
class FlowSettings:
    max_replans: int
    executor_max_turns: int
    graph_recursion_limit: int


@dataclass(frozen=True)
class QueryQdrantSettings:
    num_query: int
    score_threshold: float | None
    collection_name: str | None


@dataclass(frozen=True)
class SearchWebSettings:
    num_results: int
    topic: str | None
    search_depth: str | None
    include_answer: bool | None
    include_raw_content: bool | None
    include_images: bool | None


@dataclass(frozen=True)
class ToolSettings:
    query_qdrant: QueryQdrantSettings
    search_web: SearchWebSettings


@dataclass(frozen=True)
class PlanExecuteConfig:
    path: str
    planner_model: ModelSettings
    executor_model: ModelSettings
    replanner_model: ModelSettings
    flow: FlowSettings
    tools: ToolSettings

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'Plan-execute config file not found: {path}')

    loaded = yaml.safe_load(path.read_text(encoding='utf-8'))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError('Plan-execute config must be a YAML mapping at the top level.')
    return loaded


def _section(root: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = root
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, dict) else {}


def _string(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _int(value: Any, default: int, *, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        return max(parsed, minimum)
    return parsed


def _float_or_none(value: Any) -> float | None:
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'on'}:
        return True
    if text in {'0', 'false', 'no', 'off'}:
        return False
    return None


def _reasoning_effort(value: Any, default: str = 'low') -> str:
    text = _string(value, default).lower()
    if text not in {'low', 'medium', 'high'}:
        return default
    return text


def _model_settings(root: dict[str, Any], key: str, env_name: str, default_name: str) -> ModelSettings:
    section = _section(root, 'models', key)
    return ModelSettings(
        name=_string(os.getenv(env_name), _string(section.get('name'), default_name)),
        temperature=_float(section.get('temperature'), 0.2),
        reasoning_effort=_reasoning_effort(
            os.getenv('AGENTIC_REASONING_EFFORT'),
            _reasoning_effort(section.get('reasoning_effort'), 'low'),
        ),
    )


@lru_cache(maxsize=1)
def get_plan_execute_config() -> PlanExecuteConfig:
    raw_path = os.getenv('AGENTIC_PLAN_EXECUTE_CONFIG')
    config_path = Path(raw_path).expanduser() if raw_path else DEFAULT_PLAN_EXECUTE_CONFIG_PATH
    data = _read_yaml_config(config_path)

    flow = _section(data, 'flow')
    query_qdrant = _section(data, 'tools', 'query_qdrant')
    search_web = _section(data, 'tools', 'search_web')

    return PlanExecuteConfig(
        path=str(config_path),
        planner_model=_model_settings(data, 'planner', 'AGENTIC_PLANNER_MODEL', 'gpt-5.4-mini'),
        executor_model=_model_settings(data, 'executor', 'AGENTIC_EXECUTOR_MODEL', 'gpt-5.4-mini'),
        replanner_model=_model_settings(data, 'replanner', 'AGENTIC_REPLANNER_MODEL', 'gpt-5.4-mini'),
        flow=FlowSettings(
            max_replans=_int(os.getenv('AGENTIC_MAX_REPLANS'), _int(flow.get('max_replans'), 3, minimum=0), minimum=0),
            executor_max_turns=_int(
                os.getenv('AGENTIC_EXECUTOR_MAX_TURNS'),
                _int(flow.get('executor_max_turns'), 3, minimum=1),
                minimum=1,
            ),
            graph_recursion_limit=_int(flow.get('graph_recursion_limit'), 20, minimum=3),
        ),
        tools=ToolSettings(
            query_qdrant=QueryQdrantSettings(
                num_query=_int(query_qdrant.get('num_query'), 5, minimum=1),
                score_threshold=_float_or_none(query_qdrant.get('score_threshold')),
                collection_name=_string(query_qdrant.get('collection_name'), '') or None,
            ),
            search_web=SearchWebSettings(
                num_results=_int(search_web.get('num_results'), 5, minimum=1),
                topic=_string(search_web.get('topic'), '') or None,
                search_depth=_string(search_web.get('search_depth'), '') or None,
                include_answer=_bool_or_none(search_web.get('include_answer')),
                include_raw_content=_bool_or_none(search_web.get('include_raw_content')),
                include_images=_bool_or_none(search_web.get('include_images')),
            ),
        ),
    )