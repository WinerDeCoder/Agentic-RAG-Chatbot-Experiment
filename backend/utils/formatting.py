from __future__ import annotations

from typing import Any


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


def format_past_steps(past_steps: list[tuple[str, str]] | None) -> str:
    if not past_steps:
        return 'No steps completed yet.'

    return '\n'.join(
        f'{index}. Step: {step}\nResult: {result}'
        for index, (step, result) in enumerate(past_steps, start=1)
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


def build_execution_task_prompt(
    *,
    objective: str,
    history_text: str,
    plan_text: str,
    past_steps_text: str,
    current_step: str,
) -> str:
    return (
        f'Objective:\n{objective}\n\n'
        f'Conversation history:\n{history_text}\n\n'
        f'Current plan:\n{plan_text}\n\n'
        f'Completed steps so far:\n{past_steps_text}\n\n'
        f'Execute only this step:\n{current_step}'
    )