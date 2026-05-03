from __future__ import annotations

from typing import Any

from chat_history.postgres_store import SessionRecord


def build_session_summary_payload(record: SessionRecord) -> dict[str, Any]:
    return {
        'session_id': record.session_id,
        'title': record.title,
        'user_id': record.user.user_id,
        'external_user_id': record.user.external_user_id,
        'user_display_name': record.user.display_name,
        'created_at': record.created_at,
        'updated_at': record.updated_at,
        'message_count': record.message_count,
    }


def build_session_response_payload(record: SessionRecord) -> dict[str, Any]:
    return record.to_dict()


def build_completed_steps_payload(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for plan in plans:
        for step in plan.get('steps', []):
            if step.get('status') == 'done':
                completed.append(
                    {
                        'plan_version': plan.get('version'),
                        'step_id': step.get('step_id'),
                        'title': step.get('title'),
                        'detail': step.get('detail'),
                    }
                )
    return completed