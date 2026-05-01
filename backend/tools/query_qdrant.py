from __future__ import annotations

import atexit
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from mlflow.entities import SpanType
from qdrant_client import QdrantClient
from qdrant_client.http import models

from observability.tracing import set_current_span_attributes, set_current_span_outputs, trace_function


TOOLS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TOOLS_DIR.parent
_QDRANT_CLIENT: QdrantClient | None = None


@dataclass(frozen=True)
class QdrantQueryConfig:
    google_api_key: str
    embedding_model: str
    embedding_dimension: int
    qdrant_url: str | None
    qdrant_api_key: str | None
    qdrant_local_path: str | None
    collection_prefix: str
    collection_version: str
    collection_name_override: str | None


def _load_environment() -> None:
    load_dotenv(BACKEND_DIR / '.env')


@lru_cache(maxsize=1)
def get_qdrant_query_config() -> QdrantQueryConfig:
    _load_environment()

    google_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not google_api_key:
        raise RuntimeError('Set GOOGLE_API_KEY or GEMINI_API_KEY in backend/.env')

    qdrant_url = os.getenv('QDRANT_URL') or None
    qdrant_local_path = os.getenv('QDRANT_LOCAL_PATH') or None
    if not qdrant_url and not qdrant_local_path:
        raise RuntimeError('Set QDRANT_URL or QDRANT_LOCAL_PATH in backend/.env')

    return QdrantQueryConfig(
        google_api_key=google_api_key,
        embedding_model=os.getenv('GOOGLE_EMBEDDING_MODEL', 'gemini-embedding-001'),
        embedding_dimension=int(os.getenv('GOOGLE_EMBEDDING_DIMENSION', '1536')),
        qdrant_url=qdrant_url,
        qdrant_api_key=os.getenv('QDRANT_API_KEY') or None,
        qdrant_local_path=qdrant_local_path,
        collection_prefix=os.getenv('QDRANT_COLLECTION_PREFIX', 'gcp2_1536_page_level'),
        collection_version=os.getenv('QDRANT_COLLECTION_VERSION', 'v1'),
        collection_name_override=os.getenv('QDRANT_COLLECTION_NAME') or None,
    )


@lru_cache(maxsize=1)
def get_embedding_client() -> genai.Client:
    config = get_qdrant_query_config()
    return genai.Client(api_key=config.google_api_key)


def get_qdrant_client() -> QdrantClient:
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is not None:
        return _QDRANT_CLIENT

    config = get_qdrant_query_config()
    if config.qdrant_url:
        _QDRANT_CLIENT = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
        return _QDRANT_CLIENT

    _QDRANT_CLIENT = QdrantClient(path=config.qdrant_local_path)
    return _QDRANT_CLIENT


def close_qdrant_client() -> None:
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        return

    try:
        _QDRANT_CLIENT.close()
    except Exception:
        pass
    finally:
        _QDRANT_CLIENT = None


atexit.register(close_qdrant_client)


def resolve_collection_name(collection_name: str | None = None) -> str:
    config = get_qdrant_query_config()

    if collection_name:
        return collection_name
    if config.collection_name_override:
        return config.collection_name_override

    client = get_qdrant_client()
    collections = sorted(item.name for item in client.get_collections().collections)
    matching_collections = [
        name
        for name in collections
        if name.startswith(config.collection_prefix) and name.endswith(config.collection_version)
    ]
    if not matching_collections:
        raise RuntimeError(
            f'No Qdrant collection found for prefix={config.collection_prefix!r} '
            f'and version={config.collection_version!r}'
        )
    return matching_collections[-1]


def embed_query_text(query: str) -> list[float]:
    config = get_qdrant_query_config()
    response = get_embedding_client().models.embed_content(
        model=config.embedding_model,
        contents=query,
        config={
            'task_type': 'RETRIEVAL_QUERY',
            'output_dimensionality': config.embedding_dimension,
        },
    )
    if not response.embeddings:
        raise RuntimeError('Embedding response was empty')
    return response.embeddings[0].values


def build_metadata_filter(metadata: dict[str, Any] | None) -> models.Filter | None:
    if not metadata:
        return None

    conditions: list[models.FieldCondition] = []

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, dict):
            range_kwargs = {
                range_key: value[range_key]
                for range_key in ('gt', 'gte', 'lt', 'lte')
                if range_key in value and value[range_key] is not None
            }
            if range_kwargs:
                conditions.append(
                    models.FieldCondition(key=key, range=models.Range(**range_kwargs))
                )
                continue

            if 'value' in value:
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value['value']))
                )
                continue

            raise ValueError(
                f'Unsupported metadata filter for {key!r}. Use a scalar, list, or range dict.'
            )

        if isinstance(value, (list, tuple, set)):
            values = list(value)
            if values:
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchAny(any=values))
                )
            continue

        conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))

    return models.Filter(must=conditions) if conditions else None


def _format_hit(hit: Any) -> dict[str, Any]:
    payload = hit.payload or {}
    return {
        'id': str(hit.id),
        'score': getattr(hit, 'score', None),
        'text': payload.get('text', ''),
        'metadata': {key: value for key, value in payload.items() if key != 'text'},
    }


@trace_function(
    name='query_qdrant_tool',
    span_type=SpanType.TOOL,
    attributes={'tool.name': 'query_qdrant', 'workflow_capability': 'retrieval'},
)
def query_qdrant(
    query: str,
    num_query: int = 5,
    metadata: dict[str, Any] | None = None,
    collection_name: str | None = None,
    score_threshold: float | None = None,
) -> dict[str, Any]:
    """Retrieve page-level chunks from Qdrant.

    Keep this tool intentionally small:
    - `query` is the natural-language search text.
    - `num_query` is the number of top results to return.
    - `metadata` is an optional payload filter.

    Supported metadata filter values:
    - scalar exact match: `{'folder_name': 'MyDoc'}`
    - list match-any: `{'page_number': [1, 2, 3]}`
    - numeric range: `{'page_number': {'gte': 3, 'lte': 7}}`
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError('query must not be empty')
    if num_query < 1:
        raise ValueError('num_query must be at least 1')

    set_current_span_attributes(
        {
            'query_length': len(cleaned_query),
            'num_query': num_query,
            'has_metadata_filter': bool(metadata),
            'score_threshold': score_threshold,
        }
    )

    resolved_collection_name = resolve_collection_name(collection_name)
    query_vector = embed_query_text(cleaned_query)
    query_filter = build_metadata_filter(metadata)
    client = get_qdrant_client()

    if hasattr(client, 'query_points'):
        response = client.query_points(
            collection_name=resolved_collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=num_query,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        points = response.points
    else:
        points = client.search(
            collection_name=resolved_collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=num_query,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )

    result = {
        'query': cleaned_query,
        'collection_name': resolved_collection_name,
        'num_query': num_query,
        'metadata_filter': metadata or {},
        'score_threshold': score_threshold,
        'results': [_format_hit(hit) for hit in points],
    }
    set_current_span_outputs(
        {
            'collection_name': resolved_collection_name,
            'results_count': len(result['results']),
        }
    )
    return result


def get_query_qdrant_tool() -> Any:
    """Return a LangGraph-friendly tool when langchain-core is installed.

    The plain `query_qdrant` function already works directly.
    This wrapper keeps the upgrade path small when you switch to LangGraph.
    """
    try:
        from langchain_core.tools import tool
    except ImportError as exc:
        raise RuntimeError(
            'langchain-core is not installed. Use query_qdrant directly or install langchain-core.'
        ) from exc

    return tool(query_qdrant)
