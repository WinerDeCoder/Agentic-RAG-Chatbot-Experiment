from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from mlflow.entities import SpanType

from observability.tracing import set_current_span_attributes, set_current_span_outputs, trace_function


TOOLS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TOOLS_DIR.parent
TAVILY_SEARCH_URL = 'https://api.tavily.com/search'


@dataclass(frozen=True)
class TavilyConfig:
    api_key: str
    default_topic: str
    default_search_depth: str
    default_max_results: int
    default_include_answer: bool
    default_include_raw_content: bool
    default_include_images: bool
    timeout_seconds: int


def _load_environment() -> None:
    load_dotenv(BACKEND_DIR / '.env')


def _parse_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


@lru_cache(maxsize=1)
def get_tavily_config() -> TavilyConfig:
    _load_environment()

    api_key = os.getenv('TAVILY_API_KEY')
    if not api_key or not api_key.strip():
        raise RuntimeError('Set TAVILY_API_KEY in backend/.env')

    return TavilyConfig(
        api_key=api_key.strip(),
        default_topic=os.getenv('TAVILY_DEFAULT_TOPIC', 'general').strip() or 'general',
        default_search_depth=os.getenv('TAVILY_DEFAULT_SEARCH_DEPTH', 'basic').strip() or 'basic',
        default_max_results=int(os.getenv('TAVILY_DEFAULT_MAX_RESULTS', '5')),
        default_include_answer=_parse_bool('TAVILY_DEFAULT_INCLUDE_ANSWER', True),
        default_include_raw_content=_parse_bool('TAVILY_DEFAULT_INCLUDE_RAW_CONTENT', False),
        default_include_images=_parse_bool('TAVILY_DEFAULT_INCLUDE_IMAGES', False),
        timeout_seconds=int(os.getenv('TAVILY_TIMEOUT_SECONDS', '30')),
    )


def _clean_topic(topic: str) -> str:
    cleaned_topic = topic.strip().lower()
    if cleaned_topic not in {'general', 'news'}:
        raise ValueError("topic must be either 'general' or 'news'")
    return cleaned_topic


def _clean_search_depth(search_depth: str) -> str:
    cleaned_search_depth = search_depth.strip().lower()
    if cleaned_search_depth not in {'basic', 'advanced'}:
        raise ValueError("search_depth must be either 'basic' or 'advanced'")
    return cleaned_search_depth


def _format_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        'title': item.get('title', ''),
        'url': item.get('url', ''),
        'content': item.get('content', ''),
        'score': item.get('score'),
        'raw_content': item.get('raw_content'),
    }


@trace_function(
    name='search_web_tool',
    span_type=SpanType.TOOL,
    attributes={'tool.name': 'search_web', 'workflow_capability': 'web_search'},
)
def search_web(
    query: str,
    num_results: int | None = None,
    topic: str | None = None,
    search_depth: str | None = None,
    include_answer: bool | None = None,
    include_raw_content: bool | None = None,
    include_images: bool | None = None,
) -> dict[str, Any]:
    """Search the web with Tavily using a simple, agent-friendly interface.

    Main parameters:
    - `query`: natural-language web search query.
    - `num_results`: number of results to return.
    - `topic`: Tavily search topic, usually `general` or `news`.
    - `search_depth`: `basic` for speed, `advanced` for broader search.

    Keep this tool small first. It returns the most useful fields for downstream agents.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError('query must not be empty')

    set_current_span_attributes(
        {
            'query_length': len(cleaned_query),
            'requested_num_results': num_results,
            'requested_topic': topic,
            'requested_search_depth': search_depth,
        }
    )

    config = get_tavily_config()

    resolved_num_results = num_results if num_results is not None else config.default_max_results
    if resolved_num_results < 1:
        raise ValueError('num_results must be at least 1')

    resolved_topic = _clean_topic(topic or config.default_topic)
    resolved_search_depth = _clean_search_depth(search_depth or config.default_search_depth)
    resolved_include_answer = (
        include_answer if include_answer is not None else config.default_include_answer
    )
    resolved_include_raw_content = (
        include_raw_content
        if include_raw_content is not None
        else config.default_include_raw_content
    )
    resolved_include_images = (
        include_images if include_images is not None else config.default_include_images
    )

    payload = {
        'api_key': config.api_key,
        'query': cleaned_query,
        'topic': resolved_topic,
        'search_depth': resolved_search_depth,
        'max_results': resolved_num_results,
        'include_answer': resolved_include_answer,
        'include_raw_content': resolved_include_raw_content,
        'include_images': resolved_include_images,
    }

    response = requests.post(
        TAVILY_SEARCH_URL,
        json=payload,
        timeout=config.timeout_seconds,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text.strip()
        raise RuntimeError(f'Tavily search failed: {detail or exc}') from exc

    data = response.json()
    results = [_format_result(item) for item in data.get('results', [])]

    result = {
        'query': cleaned_query,
        'topic': resolved_topic,
        'search_depth': resolved_search_depth,
        'num_results': resolved_num_results,
        'answer': data.get('answer'),
        'images': data.get('images', []),
        'results': results,
    }
    set_current_span_outputs(
        {
            'topic': resolved_topic,
            'search_depth': resolved_search_depth,
            'results_count': len(results),
        }
    )
    return result


def get_search_web_tool() -> Any:
    """Return a LangGraph-friendly tool when langchain-core is installed."""
    try:
        from langchain_core.tools import tool
    except ImportError as exc:
        raise RuntimeError(
            'langchain-core is not installed. Use search_web directly or install langchain-core.'
        ) from exc

    return tool(search_web)