from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.http import models


SCRIPT_DIR = Path(__file__).resolve().parent
QDRAND_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE_ROOT = (
    QDRAND_DIR.parent.parent / "database" / "cleaned_data_remove_image"
)

PAGE_MARKER_PATTERN = re.compile(
    r"<!--\s*page\s+(?P<page>\d+)\s*-->\s*", re.IGNORECASE
)


@dataclass
class Config:
    source_root: Path
    markdown_glob: str
    page_overlap_pages: int
    page_overlap_characters: int
    minimum_chunk_characters: int
    embedding_model: str
    embedding_dimension: int
    google_api_key: str
    qdrant_url: str | None
    qdrant_api_key: str | None
    qdrant_local_path: str | None
    collection_prefix: str
    collection_version: str
    qdrant_distance: str
    batch_size: int
    collection_name_override: str | None


@dataclass
class PageChunk:
    point_id: str
    text: str
    metadata: dict


def extended_path(path: Path) -> str:
    path_str = str(path)
    if os.name != "nt":
        return path_str
    if path_str.startswith("\\\\?\\"):
        return path_str
    if path_str.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path_str.lstrip("\\")
    return "\\\\?\\" + path_str


def read_text_file(path: Path) -> str:
    with open(extended_path(path), "r", encoding="utf-8") as handle:
        return handle.read()


def load_config() -> Config:
    load_dotenv(QDRAND_DIR / ".env")

    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not google_api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in Qdrand/.env")

    source_root = Path(os.getenv("SOURCE_MARKDOWN_ROOT", str(DEFAULT_SOURCE_ROOT)))
    if not source_root.exists():
        raise RuntimeError(f"SOURCE_MARKDOWN_ROOT does not exist: {source_root}")

    return Config(
        source_root=source_root,
        markdown_glob=os.getenv("SOURCE_MARKDOWN_GLOB", "**/*.md"),
        page_overlap_pages=int(os.getenv("PAGE_OVERLAP_PAGES", "1")),
        page_overlap_characters=int(os.getenv("PAGE_OVERLAP_CHARACTERS", "400")),
        minimum_chunk_characters=int(os.getenv("MINIMUM_CHUNK_CHARACTERS", "80")),
        embedding_model=os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001"),
        embedding_dimension=int(os.getenv("GOOGLE_EMBEDDING_DIMENSION", "1536")),
        google_api_key=google_api_key,
        qdrant_url=os.getenv("QDRANT_URL") or None,
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        qdrant_local_path=os.getenv("QDRANT_LOCAL_PATH") or None,
        collection_prefix=os.getenv("QDRANT_COLLECTION_PREFIX", "gcp2_1536_page_level"),
        collection_version=os.getenv("QDRANT_COLLECTION_VERSION", "v1"),
        qdrant_distance=os.getenv("QDRANT_DISTANCE", "Cosine"),
        batch_size=int(os.getenv("QDRANT_BATCH_SIZE", "32")),
        collection_name_override=os.getenv("QDRANT_COLLECTION_NAME") or None,
    )


def discover_markdown_files(config: Config) -> list[Path]:
    return sorted(path for path in config.source_root.glob(config.markdown_glob) if path.is_file())


def split_pages(markdown_text: str) -> list[tuple[int, str]]:
    matches = list(PAGE_MARKER_PATTERN.finditer(markdown_text))
    if not matches:
        cleaned = markdown_text.strip()
        return [(1, cleaned)] if cleaned else []

    pages: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown_text)
        page_number = int(match.group("page"))
        page_text = markdown_text[start:end].strip()
        if page_text:
            pages.append((page_number, page_text))
    return pages


def page_overlap_suffix(
    pages: list[tuple[int, str]],
    page_index: int,
    overlap_pages: int,
    overlap_characters: int,
) -> tuple[str, list[int]]:
    suffix_parts: list[str] = []
    overlap_page_numbers: list[int] = []

    for next_index in range(page_index + 1, min(len(pages), page_index + 1 + overlap_pages)):
        next_page_number, next_page_text = pages[next_index]
        if not next_page_text:
            continue
        suffix = next_page_text[:overlap_characters].strip()
        if suffix:
            suffix_parts.append(suffix)
            overlap_page_numbers.append(next_page_number)

    return "\n\n".join(suffix_parts).strip(), overlap_page_numbers


def build_page_chunks(file_path: Path, source_root: Path, config: Config) -> list[PageChunk]:
    markdown_text = read_text_file(file_path)
    pages = split_pages(markdown_text)
    relative_path = file_path.relative_to(source_root)
    folder_name = file_path.parent.name
    parent_folder_name = file_path.parent.parent.name if file_path.parent.parent != source_root.parent else ""
    document_name = file_path.stem

    chunks: list[PageChunk] = []
    for page_index, (page_number, page_text) in enumerate(pages):
        overlap_text, overlap_page_numbers = page_overlap_suffix(
            pages,
            page_index=page_index,
            overlap_pages=config.page_overlap_pages,
            overlap_characters=config.page_overlap_characters,
        )
        chunk_text = page_text.strip()
        if overlap_text:
            chunk_text = f"{chunk_text}\n\n[Next page overlap]\n{overlap_text}".strip()

        if len(chunk_text) < config.minimum_chunk_characters:
            continue

        stable_key = f"{relative_path.as_posix()}::page::{page_number}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
        metadata = {
            "chunk_type": "page_level",
            "page_number": page_number,
            "page_index": page_index,
            "overlap_page_numbers": overlap_page_numbers,
            "overlap_pages": config.page_overlap_pages,
            "overlap_characters": config.page_overlap_characters,
            "document_name": document_name,
            "file_name": file_path.name,
            "folder_name": folder_name,
            "parent_folder_name": parent_folder_name,
            "relative_folder": file_path.parent.relative_to(source_root).as_posix(),
            "relative_path": relative_path.as_posix(),
            "source_root": str(source_root),
        }
        chunks.append(PageChunk(point_id=point_id, text=chunk_text, metadata=metadata))

    return chunks


def embed_texts(client: genai.Client, texts: list[str], config: Config) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        response = client.models.embed_content(
            model=config.embedding_model,
            contents=text,
            config={
                "task_type": "RETRIEVAL_DOCUMENT",
                "output_dimensionality": config.embedding_dimension,
            },
        )
        if not response.embeddings:
            raise RuntimeError("Google embedding response did not contain any embeddings")
        vectors.append(response.embeddings[0].values)
    return vectors


def build_collection_name(config: Config) -> str:
    if config.collection_name_override:
        return config.collection_name_override
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{config.collection_prefix}_{timestamp}_{config.collection_version}"


def qdrant_distance(name: str) -> models.Distance:
    normalized = name.strip().lower()
    mapping = {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclid": models.Distance.EUCLID,
        "manhattan": models.Distance.MANHATTAN,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported QDRANT_DISTANCE: {name}")
    return mapping[normalized]


def build_qdrant_client(config: Config) -> QdrantClient:
    if config.qdrant_url:
        return QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    if config.qdrant_local_path:
        return QdrantClient(path=config.qdrant_local_path)
    raise RuntimeError("Set QDRANT_URL or QDRANT_LOCAL_PATH in Qdrand/.env")


def create_collection(client: QdrantClient, collection_name: str, config: Config) -> None:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=config.embedding_dimension,
            distance=qdrant_distance(config.qdrant_distance),
        ),
    )


def batch_iterable(items: list[PageChunk], batch_size: int) -> Iterable[list[PageChunk]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    config = load_config()
    markdown_files = discover_markdown_files(config)
    if not markdown_files:
        raise RuntimeError(f"No markdown files found under {config.source_root}")

    all_chunks: list[PageChunk] = []
    for file_path in markdown_files:
        file_chunks = build_page_chunks(file_path, config.source_root, config)
        all_chunks.extend(file_chunks)
        print(f"Prepared {len(file_chunks)} page chunks from {file_path.relative_to(config.source_root).as_posix()}")

    if not all_chunks:
        raise RuntimeError("No page chunks were created. Check the markdown structure and page markers.")

    embedding_client = genai.Client(api_key=config.google_api_key)
    qdrant_client = build_qdrant_client(config)
    collection_name = build_collection_name(config)
    create_collection(qdrant_client, collection_name, config)

    for batch in batch_iterable(all_chunks, config.batch_size):
        texts = [item.text for item in batch]
        vectors = embed_texts(embedding_client, texts, config)
        points = [
            models.PointStruct(
                id=item.point_id,
                vector=vector,
                payload={**item.metadata, "text": item.text},
            )
            for item, vector in zip(batch, vectors, strict=True)
        ]
        qdrant_client.upsert(collection_name=collection_name, points=points)
        print(f"Upserted {len(points)} points into {collection_name}")

    print(
        {
            "collection_name": collection_name,
            "files": len(markdown_files),
            "chunks": len(all_chunks),
            "embedding_model": config.embedding_model,
            "embedding_dimension": config.embedding_dimension,
        }
    )


if __name__ == "__main__":
    main()