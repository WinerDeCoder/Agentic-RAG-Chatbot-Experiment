from __future__ import annotations

import ast
import json
from typing import Any


def format_goal(goal: dict[str, Any] | None) -> str:
    if not goal:
        return 'No goal defined yet.'

    constraints = goal.get('constraints') or []
    constraint_text = '\n'.join(f'- {item}' for item in constraints) if constraints else '- None'
    return (
        f"Goal: {goal.get('goal', 'Not specified')}\n"
        f"Expected outcome: {goal.get('expected_outcome', 'Not specified')}\n"
        f'Constraints:\n{constraint_text}'
    )


def format_history(history: list[dict[str, Any]] | None) -> str:
    if not history:
        return 'No prior conversation history.'

    lines: list[str] = []
    for index, item in enumerate(history, start=1):
        role = item.get('role', 'unknown')
        content = str(item.get('content', '') or '').strip()
        if not content:
            continue
        lines.append(f'{index}. {role}: {content}')

    return '\n'.join(lines) if lines else 'No prior conversation history.'


def format_plan_steps(steps: list[dict[str, Any]] | None) -> str:
    if not steps:
        return 'No plan steps defined.'

    lines: list[str] = []
    for index, step in enumerate(steps, start=1):
        lines.append(
            f"{index}. [{step.get('status', 'unknown')}] {step.get('title', 'Untitled step')}"
        )
        lines.append(f"   Detail: {step.get('detail', '')}")

    return '\n'.join(lines)


def format_execution_track(track: list[dict[str, Any]] | None) -> str:
    if not track:
        return 'No execution logs yet.'

    return '\n'.join(
        (
            f"{index}. Plan v{entry.get('plan_version', '?')} | "
            f"{entry.get('step_title', 'Unknown step')}\n"
            f"Status: {entry.get('status', 'unknown')}\n"
            f"Output: {entry.get('output', '')}\n"
            f"Carry-forward notes: "
            f"{'; '.join(entry.get('carry_forward_notes', [])) or 'None'}\n"
            f"Skip-stop recommended: "
            f"{'yes' if entry.get('skip_stop_recommended') else 'no'}\n"
            f"Skip-stop reason: {entry.get('skip_stop_reason', '') or 'None'}\n"
            f"Evaluation: {entry.get('evaluation', {}).get('last_status', 'unknown')} | "
            f"Confidence: {entry.get('evaluation', {}).get('confidence', 0.0)}\n"
            f"Reason: {entry.get('evaluation', {}).get('reason', '')}"
        )
        for index, entry in enumerate(track, start=1)
    )


def format_replan_history(replan_history: list[dict[str, Any]] | None) -> str:
    if not replan_history:
        return 'No replan history yet.'

    return '\n'.join(
        (
            f"{index}. v{item.get('from_version', '?')} -> v{item.get('to_version', '?')}\n"
            f"Reason: {item.get('reason', '')}\n"
            f"Failed step: {item.get('failed_step_id', 'None')}"
        )
        for index, item in enumerate(replan_history, start=1)
    )


def extract_last_message_text(agent_result: dict[str, Any]) -> str:
    messages = agent_result.get('messages', [])
    if not messages:
        return 'No response returned by executor.'

    last_message = messages[-1]
    content = getattr(last_message, 'content', '')
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get('text') or item.get('content') or ''
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return '\n'.join(parts).strip() or 'Executor returned an empty response.'

    return str(content).strip() or 'Executor returned an empty response.'


def _parse_tool_message_content(content: Any) -> dict[str, Any] | None:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        combined = '\n'.join(str(item) for item in content if item)
    else:
        combined = str(content).strip()
    if not combined:
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(combined)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_sources_from_tool_payload(tool_name: str, payload: dict[str, Any] | None) -> list[dict[str, str]]:
    if not payload:
        return []

    sources: list[dict[str, str]] = []
    if tool_name == 'query_qdrant':
        for item in payload.get('results', [])[:5]:
            metadata = item.get('metadata', {}) or {}
            document_name = str(metadata.get('document_name') or metadata.get('file_name') or '').strip()
            page_number = metadata.get('page_number')
            relative_path = str(metadata.get('relative_path') or '').strip()
            label_parts = [part for part in [document_name, f'page {page_number}' if page_number else ''] if part]
            label = ', '.join(label_parts) or relative_path or 'internal source'
            location_parts = [part for part in [relative_path, f'page {page_number}' if page_number else ''] if part]
            sources.append(
                {
                    'type': 'internal',
                    'label': label,
                    'location': ' | '.join(location_parts) if location_parts else label,
                }
            )
    elif tool_name == 'search_web':
        for item in payload.get('results', [])[:5]:
            title = str(item.get('title') or 'web source').strip()
            url = str(item.get('url') or '').strip()
            if not url:
                continue
            sources.append(
                {
                    'type': 'web',
                    'label': title,
                    'location': url,
                }
            )
    return sources


def _dedupe_sources(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, str]] = []
    for source in sources:
        key = (
            source.get('type', ''),
            source.get('label', ''),
            source.get('location', ''),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def format_sources_section(sources: list[dict[str, str]] | None) -> str:
    if not sources:
        return 'Sources:\n- None explicitly captured.'

    lines = ['Sources:']
    for source in sources:
        label = source.get('label', 'source')
        location = source.get('location', '')
        if location and location != label:
            lines.append(f'- {label} | {location}')
        else:
            lines.append(f'- {label}')
    return '\n'.join(lines)


def ensure_response_has_sources(response_text: str, sources: list[dict[str, str]] | None) -> str:
    cleaned = response_text.strip() or 'Executor returned an empty response.'
    if 'Sources:' in cleaned:
        return cleaned
    return f"{cleaned}\n\n{format_sources_section(sources)}"


def extract_tool_use_logs(agent_result: dict[str, Any]) -> list[dict[str, Any]]:
    messages = agent_result.get('messages', [])
    if not messages:
        return []

    pending_calls: dict[str, dict[str, Any]] = {}
    ordered_logs: list[dict[str, Any]] = []

    for message in messages:
        tool_calls = getattr(message, 'tool_calls', None) or []
        for tool_call in tool_calls:
            tool_call_id = tool_call.get('id') or f"tool_call_{len(ordered_logs) + 1}"
            log = {
                'tool_name': tool_call.get('name', 'unknown_tool'),
                'tool_input': tool_call.get('args', {}),
                'tool_output_summary': '',
                'sources': [],
                'status': 'success',
            }
            pending_calls[tool_call_id] = log
            ordered_logs.append(log)

        message_type = getattr(message, 'type', '')
        if message_type != 'tool':
            continue

        tool_call_id = getattr(message, 'tool_call_id', None)
        content = getattr(message, 'content', '')
        if isinstance(content, list):
            content = '\n'.join(str(item) for item in content if item)

        summary = str(content).strip()
        if len(summary) > 400:
            summary = summary[:400].rstrip() + '...'
        parsed_payload = _parse_tool_message_content(content)

        if tool_call_id and tool_call_id in pending_calls:
            pending_calls[tool_call_id]['tool_output_summary'] = summary
            pending_calls[tool_call_id]['sources'] = _extract_sources_from_tool_payload(
                pending_calls[tool_call_id].get('tool_name', ''),
                parsed_payload,
            )
            pending_calls[tool_call_id]['status'] = 'fail' if summary.startswith('ERROR:') else 'success'
        else:
            ordered_logs.append(
                {
                    'tool_name': getattr(message, 'name', 'unknown_tool'),
                    'tool_input': {},
                    'tool_output_summary': summary,
                    'sources': _extract_sources_from_tool_payload(
                        getattr(message, 'name', 'unknown_tool'),
                        parsed_payload,
                    ),
                    'status': 'fail' if summary.startswith('ERROR:') else 'success',
                }
            )

    return ordered_logs


def collect_sources_from_tool_use_logs(tool_use_logs: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    if not tool_use_logs:
        return []

    sources: list[dict[str, str]] = []
    for log in tool_use_logs:
        for source in log.get('sources', []) or []:
            if isinstance(source, dict):
                sources.append(
                    {
                        'type': str(source.get('type', '')),
                        'label': str(source.get('label', '')),
                        'location': str(source.get('location', '')),
                    }
                )
    return _dedupe_sources(sources)


def build_execution_task_prompt(
    *,
    goal_text: str,
    full_plan_text: str,
    future_steps_text: str,
    current_step_title: str,
    current_step_detail: str,
    current_plan_version: int,
) -> str:
    return (
        f'Goal context:\n{goal_text}\n\n'
        f'Current plan version: {current_plan_version}\n\n'
        f'Full current plan:\n{full_plan_text}\n\n'
        f'Later pending steps:\n{future_steps_text}\n\n'
        f'Execute only this step:\n'
        f'Title: {current_step_title}\n'
        f'Detail: {current_step_detail}'
    )