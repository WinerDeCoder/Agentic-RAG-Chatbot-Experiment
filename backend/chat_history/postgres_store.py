from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.strip()


@dataclass
class UserRecord:
    user_id: str
    external_user_id: str
    display_name: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "external_user_id": self.external_user_id,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


@dataclass
class SessionRecord:
    session_id: str
    user: UserRecord
    title: str
    created_at: str
    updated_at: str
    messages: list[dict[str, Any]]
    message_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user": self.user.to_dict(),
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
            "message_count": self.message_count,
        }


class PostgresChatHistoryStore:
    def __init__(self) -> None:
        self.host = require_env("CHAT_HISTORY_DB_HOST")
        self.port = int(require_env("CHAT_HISTORY_DB_PORT"))
        self.database = require_env("CHAT_HISTORY_DB_NAME")
        self.user = require_env("CHAT_HISTORY_DB_USER")
        self.password = require_env("CHAT_HISTORY_DB_PASSWORD")
        self.schema = require_env("CHAT_HISTORY_DB_SCHEMA")
        self.default_external_user_id = require_env("CHAT_HISTORY_DEFAULT_USER_ID")
        self.default_display_name = require_env("CHAT_HISTORY_DEFAULT_USER_NAME")
        self._conninfo = (
            f"host={self.host} port={self.port} dbname={self.database} "
            f"user={self.user} password={self.password}"
        )
        self.ensure_schema()

    def connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self._conninfo, row_factory=dict_row)

    def ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.chat_users (
                    user_id UUID PRIMARY KEY,
                    external_user_id TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.chat_sessions (
                    session_id UUID PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES {self.schema}.chat_users(user_id),
                    title TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.chat_messages (
                    message_id UUID PRIMARY KEY,
                    session_id UUID NOT NULL REFERENCES {self.schema}.chat_sessions(session_id) ON DELETE CASCADE,
                    user_id UUID REFERENCES {self.schema}.chat_users(user_id),
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    model_name TEXT,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT chat_messages_role_check CHECK (role IN ('system', 'user', 'assistant', 'tool'))
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_chat_users_external_user_id ON {self.schema}.chat_users (external_user_id)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated_at ON {self.schema}.chat_sessions (user_id, updated_at DESC)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_chat_sessions_title_trgm ON {self.schema}.chat_sessions USING GIN (title gin_trgm_ops)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created_at ON {self.schema}.chat_messages (session_id, created_at ASC)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_chat_messages_user_created_at ON {self.schema}.chat_messages (user_id, created_at DESC)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_chat_messages_role_created_at ON {self.schema}.chat_messages (role, created_at DESC)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_chat_messages_content_trgm ON {self.schema}.chat_messages USING GIN (content gin_trgm_ops)"
            )
            conn.commit()

    def health(self) -> dict[str, Any]:
        return {
            "db_host": self.host,
            "db_port": self.port,
            "db_name": self.database,
            "db_schema": self.schema,
        }

    def _row_to_user(self, row: dict[str, Any]) -> UserRecord:
        return UserRecord(
            user_id=str(row["user_id"]),
            external_user_id=row["external_user_id"],
            display_name=row["display_name"],
            created_at=row["created_at"].isoformat(),
            updated_at=row["updated_at"].isoformat(),
            metadata=row.get("metadata") or {},
        )

    def get_or_create_user(
        self,
        external_user_id: str | None = None,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UserRecord:
        external_user_id = (external_user_id or self.default_external_user_id).strip()
        display_name = (display_name or self.default_display_name).strip()
        metadata = metadata or {}

        with self.connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {self.schema}.chat_users WHERE external_user_id = %s",
                (external_user_id,),
            ).fetchone()
            if row:
                return self._row_to_user(row)

            user_id = str(uuid.uuid4())
            conn.execute(
                f"""
                INSERT INTO {self.schema}.chat_users (
                    user_id,
                    external_user_id,
                    display_name,
                    metadata,
                    created_at,
                    updated_at
                ) VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                """,
                (user_id, external_user_id, display_name, json.dumps(metadata), utc_now(), utc_now()),
            )
            conn.commit()
            row = conn.execute(
                f"SELECT * FROM {self.schema}.chat_users WHERE user_id = %s",
                (user_id,),
            ).fetchone()
            if not row:
                raise RuntimeError("Failed to create chat user")
            return self._row_to_user(row)

    def create_session(
        self,
        title: str,
        external_user_id: str | None = None,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionRecord:
        user = self.get_or_create_user(external_user_id=external_user_id, display_name=display_name)
        session_id = str(uuid.uuid4())
        timestamp = utc_now()

        with self.connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema}.chat_sessions (
                    session_id,
                    user_id,
                    title,
                    metadata,
                    created_at,
                    updated_at
                ) VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                """,
                (session_id, user.user_id, title.strip(), json.dumps(metadata or {}), timestamp, timestamp),
            )
            conn.commit()

        return SessionRecord(
            session_id=session_id,
            user=user,
            title=title.strip(),
            created_at=timestamp,
            updated_at=timestamp,
            messages=[],
            message_count=0,
        )

    def list_sessions(self) -> list[SessionRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    s.session_id,
                    s.title,
                    s.created_at,
                    s.updated_at,
                    u.user_id,
                    u.external_user_id,
                    u.display_name,
                    u.metadata AS user_metadata,
                    u.created_at AS user_created_at,
                    u.updated_at AS user_updated_at,
                    COUNT(m.message_id) AS message_count
                FROM {self.schema}.chat_sessions s
                JOIN {self.schema}.chat_users u ON u.user_id = s.user_id
                LEFT JOIN {self.schema}.chat_messages m ON m.session_id = s.session_id
                GROUP BY
                    s.session_id,
                    s.title,
                    s.created_at,
                    s.updated_at,
                    u.user_id,
                    u.external_user_id,
                    u.display_name,
                    u.metadata,
                    u.created_at,
                    u.updated_at
                ORDER BY s.updated_at DESC
                """
            ).fetchall()

        records: list[SessionRecord] = []
        for row in rows:
            user = UserRecord(
                user_id=str(row["user_id"]),
                external_user_id=row["external_user_id"],
                display_name=row["display_name"],
                created_at=row["user_created_at"].isoformat(),
                updated_at=row["user_updated_at"].isoformat(),
                metadata=row.get("user_metadata") or {},
            )
            records.append(
                SessionRecord(
                    session_id=str(row["session_id"]),
                    user=user,
                    title=row["title"],
                    created_at=row["created_at"].isoformat(),
                    updated_at=row["updated_at"].isoformat(),
                    messages=[],
                    message_count=int(row["message_count"]),
                )
            )
        return records

    def get_session(self, session_id: str) -> SessionRecord:
        with self.connect() as conn:
            session_row = conn.execute(
                f"""
                SELECT
                    s.session_id,
                    s.title,
                    s.created_at,
                    s.updated_at,
                    u.user_id,
                    u.external_user_id,
                    u.display_name,
                    u.metadata AS user_metadata,
                    u.created_at AS user_created_at,
                    u.updated_at AS user_updated_at
                FROM {self.schema}.chat_sessions s
                JOIN {self.schema}.chat_users u ON u.user_id = s.user_id
                WHERE s.session_id = %s
                """,
                (session_id,),
            ).fetchone()
            if not session_row:
                raise KeyError(session_id)

            message_rows = conn.execute(
                f"""
                SELECT message_id, user_id, role, content, model_name, metadata, created_at
                FROM {self.schema}.chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC, message_id ASC
                """,
                (session_id,),
            ).fetchall()

        user = UserRecord(
            user_id=str(session_row["user_id"]),
            external_user_id=session_row["external_user_id"],
            display_name=session_row["display_name"],
            created_at=session_row["user_created_at"].isoformat(),
            updated_at=session_row["user_updated_at"].isoformat(),
            metadata=session_row.get("user_metadata") or {},
        )
        messages = [
            {
                "id": str(row["message_id"]),
                "user_id": str(row["user_id"]) if row.get("user_id") else None,
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"].isoformat(),
                "model": row.get("model_name"),
                "metadata": row.get("metadata") or {},
            }
            for row in message_rows
        ]
        return SessionRecord(
            session_id=str(session_row["session_id"]),
            user=user,
            title=session_row["title"],
            created_at=session_row["created_at"].isoformat(),
            updated_at=session_row["updated_at"].isoformat(),
            messages=messages,
            message_count=len(messages),
        )

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: str | None = None,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        message_id = str(uuid.uuid4())
        created_at = utc_now()
        with self.connect() as conn:
            updated = conn.execute(
                f"UPDATE {self.schema}.chat_sessions SET updated_at = %s WHERE session_id = %s",
                (created_at, session_id),
            )
            if updated.rowcount == 0:
                raise KeyError(session_id)
            conn.execute(
                f"""
                INSERT INTO {self.schema}.chat_messages (
                    message_id,
                    session_id,
                    user_id,
                    role,
                    content,
                    model_name,
                    metadata,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                """,
                (message_id, session_id, user_id, role, content, model_name, json.dumps(metadata or {}), created_at),
            )
            conn.commit()
        return {
            "id": message_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "created_at": created_at,
            "model": model_name,
            "metadata": metadata or {},
        }
