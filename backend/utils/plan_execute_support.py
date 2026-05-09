from __future__ import annotations

import ast
from copy import deepcopy
import json
import logging
from typing import Any

from config import get_plan_execute_config
from tools import query_qdrant, search_web
from utils.formatting import (
    build_execution_task_prompt,
    format_goal,
    format_plan_steps,
    format_sources_section,
)


def state_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    plans = state.get('plans', [])
    return {
        'input_preview': state.get('input', '')[:160],
        'goal': state.get('goal', {}),
        'current_step': state.get('execute_track', {}).get('current_step', {}),
        'control': state.get('control', {}),
        'plans': [
            {
                'version': plan.get('version'),
                'created_by': plan.get('created_by'),
                'reason': plan.get('reason'),
                'steps': [
                    {
                        'step_id': step.get('step_id'),
                        'title': step.get('title'),
                        'status': step.get('status'),
                    }
                    for step in plan.get('steps', [])
                ],
            }
            for plan in plans
        ],
        'execution_log_count': len(state.get('execute_track', {}).get('track', [])),
        'replan_history_count': len(state.get('replan_history', [])),
        'has_response': bool(state.get('response')),
        'last_error': state.get('last_error'),
    }


def log_node_event(
    logger: logging.Logger,
    node_name: str,
    phase: str,
    state: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        'node': node_name,
        'phase': phase,
        'state': state_snapshot(state),
    }
    if extra:
        payload['extra'] = extra
    logger.info('workflow_node_event %s', json.dumps(payload, default=str))


def copy_plan_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [deepcopy(step) for step in steps]


def copy_plans(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            'version': plan['version'],
            'created_by': plan['created_by'],
            'reason': plan['reason'],
            'steps': copy_plan_steps(plan.get('steps', [])),
        }
        for plan in plans
    ]


def ensure_control(state: dict[str, Any]) -> dict[str, Any]:
    return {
        'step_count': state.get('control', {}).get('step_count', 0),
        'replan_time': state.get('control', {}).get('replan_time', 0),
        'status': state.get('control', {}).get('status', 'running'),
    }


def increment_control(
    state: dict[str, Any],
    *,
    increment_replan: bool = False,
    status: str | None = None,
) -> dict[str, Any]:
    control = ensure_control(state)
    control['step_count'] = control.get('step_count', 0) + 1
    if increment_replan:
        control['replan_time'] = control.get('replan_time', 0) + 1
    if status is not None:
        control['status'] = status
    return control


def build_step_id(version: int, index: int) -> str:
    return f'v{version}_step_{index}'


def build_plan_version(
    *,
    version: int,
    created_by: str,
    reason: str,
    steps: list[Any],
) -> dict[str, Any]:
    return {
        'version': version,
        'created_by': created_by,
        'reason': reason,
        'steps': [
            {
                'step_id': build_step_id(version, index),
                'title': step.title,
                'detail': step.detail,
                'status': 'pending',
                'depends_on': [],
            }
            for index, step in enumerate(steps, start=1)
        ],
    }


def latest_plan(plans: list[dict[str, Any]]) -> dict[str, Any] | None:
    return plans[-1] if plans else None


def find_plan(plans: list[dict[str, Any]], version: int | None) -> dict[str, Any] | None:
    if version is None:
        return None
    for plan in plans:
        if plan.get('version') == version:
            return plan
    return None


def find_step(plan: dict[str, Any] | None, step_id: str | None) -> dict[str, Any] | None:
    if not plan or not step_id:
        return None
    for step in plan.get('steps', []):
        if step.get('step_id') == step_id:
            return step
    return None


def first_pending_step(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    if not plan:
        return None
    for step in plan.get('steps', []):
        if step.get('status') == 'pending':
            return step
    return None


def current_pointer(state: dict[str, Any]) -> dict[str, Any]:
    return state.get('execute_track', {}).get(
        'current_step', {'plan_version': None, 'step_id': None}
    )


def build_execute_track(
    state: dict[str, Any],
    *,
    current_plan_version: int | None,
    current_step_id: str | None,
    track: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    existing_track = state.get('execute_track', {}).get('track', [])
    return {
        'current_step': {
            'plan_version': current_plan_version,
            'step_id': current_step_id,
        },
        'track': track if track is not None else list(existing_track),
    }


def append_execution_log(state: dict[str, Any], log: dict[str, Any]) -> dict[str, Any]:
    track = list(state.get('execute_track', {}).get('track', []))
    track.append(log)
    return {
        'current_step': current_pointer(state),
        'track': track,
    }


def latest_execution_log(state: dict[str, Any]) -> dict[str, Any] | None:
    track = state.get('execute_track', {}).get('track', [])
    return track[-1] if track else None


def mark_step_status(
    plans: list[dict[str, Any]],
    *,
    version: int | None,
    step_id: str | None,
    status: str,
) -> list[dict[str, Any]]:
    updated = copy_plans(plans)
    plan = find_plan(updated, version)
    step = find_step(plan, step_id)
    if step:
        step['status'] = status
    return updated


def build_replan_version(
    *,
    previous_plan: dict[str, Any],
    new_version: int,
    update_reason: str,
    revised_steps: list[Any],
    failed_step_id: str | None,
) -> dict[str, Any]:
    preserved_done_steps = [
        deepcopy(step) for step in previous_plan.get('steps', []) if step.get('status') == 'done'
    ]
    new_steps = preserved_done_steps + [
        {
            'step_id': build_step_id(new_version, offset),
            'title': step.title,
            'detail': step.detail,
            'status': 'pending',
            'depends_on': [],
        }
        for offset, step in enumerate(revised_steps, start=len(preserved_done_steps) + 1)
    ]
    return {
        'version': new_version,
        'created_by': 'replanner',
        'reason': update_reason,
        'steps': new_steps,
    }


def goal_dict(goal: Any) -> dict[str, Any]:
    return {
        'goal': goal.goal,
        'expected_outcome': goal.expected_outcome,
        'constraints': list(goal.constraints),
    }


def fallback_from_latest_execution(state: dict[str, Any]) -> str:
    latest_execution = latest_execution_log(state)
    if latest_execution and latest_execution.get('output'):
        return latest_execution['output']
    return 'I could not produce a final answer.'


def prefers_internal_search(goal: dict[str, Any] | None, step: dict[str, Any]) -> bool:
    combined_text = ' '.join(
        [
            str(goal.get('goal', '') if goal else ''),
            str(goal.get('expected_outcome', '') if goal else ''),
            step.get('title', ''),
            step.get('detail', ''),
        ]
    ).lower()
    keywords = (
        'internal',
        'knowledge base',
        'qdrant',
        'policy',
        'process',
        'guideline',
        'document',
        'kb',
    )
    return any(keyword in combined_text for keyword in keywords)


def prefers_web_search(goal: dict[str, Any] | None, step: dict[str, Any]) -> bool:
    combined_text = ' '.join(
        [
            str(goal.get('goal', '') if goal else ''),
            str(goal.get('expected_outcome', '') if goal else ''),
            step.get('title', ''),
            step.get('detail', ''),
        ]
    ).lower()
    keywords = (
        'web',
        'public',
        'internet',
        'online',
        'news',
        'recent',
        'current',
        'external',
        'tavily',
    )
    return any(keyword in combined_text for keyword in keywords)


def build_executor_query(state: dict[str, Any], step: dict[str, Any]) -> str:
    goal = state.get('goal', {})
    step_detail = step.get('detail', '') or ''
    metadata_marker = 'metadata filter:'
    detail_text = step_detail
    marker_index = step_detail.lower().find(metadata_marker)
    if marker_index >= 0:
        detail_text = step_detail[:marker_index].strip()
    return ' '.join(
        part.strip()
        for part in [
            str(goal.get('goal', '') or ''),
            step.get('title', '') or '',
            detail_text,
        ]
        if part and str(part).strip()
    )


def extract_metadata_filter(step: dict[str, Any]) -> dict[str, Any] | None:
    detail_text = step.get('detail', '') or ''
    marker = 'Metadata filter:'
    marker_index = detail_text.lower().find(marker.lower())
    if marker_index < 0:
        return None

    raw_filter = detail_text[marker_index + len(marker):].strip()
    if not raw_filter:
        return None

    if raw_filter[0] == '{':
        depth = 0
        end_index = None
        for index, character in enumerate(raw_filter):
            if character == '{':
                depth += 1
            elif character == '}':
                depth -= 1
                if depth == 0:
                    end_index = index + 1
                    break
        if end_index is not None:
            raw_filter = raw_filter[:end_index]

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw_filter)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError('Metadata filter must be a valid JSON or Python dict literal.')


def summarize_tool_output(tool_output: dict[str, Any]) -> str:
    summary = json.dumps(tool_output, default=str)
    if len(summary) > 400:
        return summary[:400].rstrip() + '...'
    return summary


def clip_text(value: str, limit: int = 1200) -> str:
    cleaned = ' '.join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + '...'


def render_tool_output_for_prompt(log: dict[str, Any]) -> str:
    tool_name = log.get('tool_name')
    tool_output = log.get('tool_output', {}) or {}

    if tool_name == 'query_qdrant':
        blocks: list[str] = []
        for index, item in enumerate(tool_output.get('results', [])[:5], start=1):
            metadata = item.get('metadata', {}) or {}
            label = str(
                metadata.get('document_name')
                or metadata.get('file_name')
                or metadata.get('relative_path')
                or f'internal_result_{index}'
            ).strip()
            page_number = metadata.get('page_number')
            prefix = f'Retrieved chunk {index}: {label}'
            if page_number is not None:
                prefix += f' | page {page_number}'
            chunk_text = clip_text(str(item.get('text', '') or ''))
            if chunk_text:
                blocks.append(f'{prefix}\nText: {chunk_text}')
        if blocks:
            return '\n\n'.join(blocks)

    if tool_name == 'search_web':
        blocks = []
        for index, item in enumerate(tool_output.get('results', [])[:5], start=1):
            title = str(item.get('title') or f'web_result_{index}').strip()
            url = str(item.get('url') or '').strip()
            content = clip_text(str(item.get('content', '') or ''))
            pieces = [f'Web result {index}: {title}']
            if url:
                pieces.append(f'URL: {url}')
            if content:
                pieces.append(f'Content: {content}')
            blocks.append('\n'.join(pieces))
        if blocks:
            return '\n\n'.join(blocks)

    return log.get('tool_output_summary', '')


def sources_from_qdrant_output(tool_output: dict[str, Any]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    for item in tool_output.get('results', [])[:5]:
        metadata = item.get('metadata', {}) or {}
        document_name = str(metadata.get('document_name') or metadata.get('file_name') or 'internal source').strip()
        page_number = metadata.get('page_number')
        relative_path = str(metadata.get('relative_path') or '').strip()
        label = document_name + (f', page {page_number}' if page_number else '')
        location_parts = [part for part in [relative_path, f'page {page_number}' if page_number else ''] if part]
        sources.append(
            {
                'type': 'internal',
                'label': label,
                'location': ' | '.join(location_parts) if location_parts else label,
            }
        )
    return sources


def sources_from_web_output(tool_output: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            'type': 'web',
            'label': str(item.get('title') or 'web source'),
            'location': str(item.get('url') or ''),
        }
        for item in tool_output.get('results', [])[:5]
        if item.get('url')
    ]


def run_executor_tools(state: dict[str, Any], step: dict[str, Any]) -> list[dict[str, Any]]:
    tool_logs: list[dict[str, Any]] = []
    config = get_plan_execute_config()
    query_text = build_executor_query(state, step)
    metadata_filter = extract_metadata_filter(step)
    use_internal = prefers_internal_search(state.get('goal'), step)
    use_web = prefers_web_search(state.get('goal'), step)

    if not use_internal and not use_web:
        use_internal = True

    if use_internal:
        tool_output = query_qdrant(
            query=query_text,
            num_query=config.tools.query_qdrant.num_query,
            metadata=metadata_filter,
            collection_name=config.tools.query_qdrant.collection_name,
            score_threshold=config.tools.query_qdrant.score_threshold,
        )
        tool_logs.append(
            {
                'tool_name': 'query_qdrant',
                'tool_input': {
                    'query': query_text,
                    'num_query': config.tools.query_qdrant.num_query,
                    'metadata': metadata_filter,
                    'collection_name': config.tools.query_qdrant.collection_name,
                    'score_threshold': config.tools.query_qdrant.score_threshold,
                },
                'tool_output': tool_output,
                'tool_output_summary': summarize_tool_output(tool_output),
                'sources': sources_from_qdrant_output(tool_output),
                'status': 'success',
            }
        )

    if use_web:
        tool_output = search_web(
            query=query_text,
            num_results=config.tools.search_web.num_results,
            topic=config.tools.search_web.topic,
            search_depth=config.tools.search_web.search_depth,
            include_answer=config.tools.search_web.include_answer,
            include_raw_content=config.tools.search_web.include_raw_content,
            include_images=config.tools.search_web.include_images,
        )
        tool_logs.append(
            {
                'tool_name': 'search_web',
                'tool_input': {
                    'query': query_text,
                    'num_results': config.tools.search_web.num_results,
                    'topic': config.tools.search_web.topic,
                    'search_depth': config.tools.search_web.search_depth,
                    'include_answer': config.tools.search_web.include_answer,
                    'include_raw_content': config.tools.search_web.include_raw_content,
                    'include_images': config.tools.search_web.include_images,
                },
                'tool_output': tool_output,
                'tool_output_summary': summarize_tool_output(tool_output),
                'sources': sources_from_web_output(tool_output),
                'status': 'success',
            }
        )

    return tool_logs


def build_executor_synthesis_input(
    *,
    state: dict[str, Any],
    step: dict[str, Any],
    current_plan_version: int,
    tool_use_logs: list[dict[str, Any]],
) -> str:
    current_plan = find_plan(state.get('plans', []), current_plan_version)
    full_plan_text = format_plan_steps(current_plan.get('steps', []) if current_plan else [])
    future_steps = []
    seen_current_step = False
    for plan_step in current_plan.get('steps', []) if current_plan else []:
        if plan_step.get('step_id') == step.get('step_id'):
            seen_current_step = True
            continue
        if seen_current_step and plan_step.get('status') != 'done':
            future_steps.append(plan_step)
    future_steps_text = format_plan_steps(future_steps) if future_steps else 'No later pending steps.'

    evidence_blocks: list[str] = []
    for index, log in enumerate(tool_use_logs, start=1):
        explicit_output = render_tool_output_for_prompt(log)
        evidence_blocks.append(
            '\n'.join(
                [
                    f'Tool {index}: {log.get("tool_name", "unknown_tool")}',
                    f'Input: {json.dumps(log.get("tool_input", {}), default=str)}',
                    f'Output summary: {log.get("tool_output_summary", "")}',
                    f'Retrieved content:\n{explicit_output}',
                    format_sources_section(log.get('sources', [])),
                ]
            )
        )

    task_prompt = build_execution_task_prompt(
        goal_text=format_goal(state.get('goal')),
        full_plan_text=full_plan_text,
        future_steps_text=future_steps_text,
        current_step_title=step.get('title', ''),
        current_step_detail=step.get('detail', ''),
        current_plan_version=current_plan_version,
    )
    evidence_text = '\n\n'.join(evidence_blocks) if evidence_blocks else 'No evidence collected.'
    return (
        f'{task_prompt}\n\n'
        'Use only the evidence below to answer this step.\n\n'
        f'Evidence:\n{evidence_text}'
    )


def extract_named_section(text: str, section_name: str) -> str:
    marker = f'{section_name}:'
    start_index = text.find(marker)
    if start_index < 0:
        return ''

    remaining = text[start_index + len(marker):]
    next_markers = [
        remaining.find('\nCarry-forward notes:'),
        remaining.find('\nSkip-stop recommendation:'),
        remaining.find('\nSkip-stop reason:'),
        remaining.find('\nSources:'),
    ]
    next_positions = [position for position in next_markers if position >= 0]
    end_index = min(next_positions) if next_positions else len(remaining)
    return remaining[:end_index].strip()


def parse_carry_forward_notes(text: str) -> list[str]:
    section = extract_named_section(text, 'Carry-forward notes')
    if not section:
        return []

    notes: list[str] = []
    for line in section.splitlines():
        cleaned = line.strip()
        if cleaned.startswith('- '):
            cleaned = cleaned[2:].strip()
        if not cleaned or cleaned.lower() == 'none':
            continue
        notes.append(cleaned)
    return notes


def parse_skip_stop_recommendation(text: str) -> tuple[bool, str | None]:
    recommendation_section = extract_named_section(text, 'Skip-stop recommendation').lower()
    reason_section = extract_named_section(text, 'Skip-stop reason') or None
    return recommendation_section.startswith('yes'), reason_section


def parse_step_result_text(text: str) -> str:
    section = extract_named_section(text, 'Step result')
    cleaned = section or text.strip()
    if '\nCarry-forward notes:' in cleaned:
        cleaned = cleaned.split('\nCarry-forward notes:', 1)[0].strip()
    if '\nSkip-stop recommendation:' in cleaned:
        cleaned = cleaned.split('\nSkip-stop recommendation:', 1)[0].strip()
    return cleaned or 'Executor returned an empty response.'