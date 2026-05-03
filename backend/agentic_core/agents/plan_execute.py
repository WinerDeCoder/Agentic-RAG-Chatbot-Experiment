from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
import json
import logging
from typing import Annotated, Any, Literal

from langgraph.constants import END
from mlflow.entities import SpanType
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from config import get_plan_execute_config
from utils import (
    build_execution_task_prompt,
    build_plan_execute_runnables,
    collect_sources_from_tool_use_logs,
    ensure_response_has_sources,
    format_execution_track,
    format_goal,
    format_history,
    format_plan_steps,
    format_replan_history,
    format_sources_section,
)
from observability.tracing import set_current_span_attributes, set_current_span_outputs, trace_function
from tools import query_qdrant, search_web


LOGGER = logging.getLogger(__name__)


class GoalSpec(TypedDict, total=False):
    goal: str
    expected_outcome: str
    constraints: list[str]


class PlanStepState(TypedDict, total=False):
    step_id: str
    title: str
    detail: str
    status: Literal['pending', 'running', 'done', 'failed']
    depends_on: list[str]


class PlanVersionState(TypedDict, total=False):
    version: int
    created_by: Literal['planner', 'replanner']
    reason: str
    steps: list[PlanStepState]


class ToolUseLogState(TypedDict, total=False):
    tool_name: str
    tool_input: str | dict[str, Any]
    tool_output: dict[str, Any]
    tool_output_summary: str
    sources: list[dict[str, str]]
    status: Literal['success', 'fail']


class ExecutionEvaluationState(TypedDict, total=False):
    last_status: Literal['pass', 'fail', 'partial']
    confidence: float
    reason: str


class ExecutionLogState(TypedDict, total=False):
    plan_version: int
    step_id: str
    step_title: str
    step_detail: str
    output: str
    status: Literal['success', 'fail']
    error: str | None
    tool_use_logs: list[ToolUseLogState]
    sources: list[dict[str, str]]
    carry_forward_notes: list[str]
    skip_stop_recommended: bool
    skip_stop_reason: str | None
    evaluation: ExecutionEvaluationState
    possible_solutions: list[str]


def _state_snapshot(state: AgentState) -> dict[str, Any]:
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


def _log_node_event(node_name: str, phase: str, state: AgentState, extra: dict[str, Any] | None = None) -> None:
    payload = {
        'node': node_name,
        'phase': phase,
        'state': _state_snapshot(state),
    }
    if extra:
        payload['extra'] = extra
    LOGGER.info('workflow_node_event %s', json.dumps(payload, default=str))


class CurrentStepPointer(TypedDict, total=False):
    plan_version: int | None
    step_id: str | None


class ExecuteTrackState(TypedDict, total=False):
    current_step: CurrentStepPointer
    track: list[ExecutionLogState]


class ControlState(TypedDict, total=False):
    step_count: int
    replan_time: int
    status: Literal['running', 'done', 'failed']


class ReplanHistoryState(TypedDict, total=False):
    from_version: int
    to_version: int
    failed_step_id: str | None
    old_step_snapshot: dict[str, Any] | None
    new_step_snapshot: dict[str, Any] | None
    reason: str


class AgentState(TypedDict, total=False):
    input: str
    history: list[dict[str, Any]]
    goal: GoalSpec
    plans: list[PlanVersionState]
    execute_track: ExecuteTrackState
    control: ControlState
    replan_history: list[ReplanHistoryState]
    response: str
    last_error: str | None


class GoalSpecModel(BaseModel):
    goal: str = Field(description='The mission the workflow should accomplish.')
    expected_outcome: str = Field(description='The expected final output to deliver to the user.')
    constraints: list[str] = Field(
        default_factory=list,
        description='Explicit user constraints, preferences, or scope limits.',
    )


class PlanStepDraft(BaseModel):
    title: str = Field(description='Short step title.')
    detail: str = Field(description='Execution-ready detail for this step.')


class PlannedResponse(BaseModel):
    kind: Literal['plan'] = 'plan'
    goal: GoalSpecModel
    steps: list[PlanStepDraft] = Field(description='Ordered steps to solve the user request.')



class DirectResponse(BaseModel):
    kind: Literal['direct'] = 'direct'
    goal: GoalSpecModel
    response: str = Field(description='Final response for the user.')


class EvaluationModel(BaseModel):
    last_status: Literal['pass', 'fail', 'partial']
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class FinalResponseDecision(BaseModel):
    kind: Literal['final'] = 'final'
    evaluation: EvaluationModel
    response: str


class PassDecision(BaseModel):
    kind: Literal['pass'] = 'pass'
    evaluation: EvaluationModel


class ReviseDecision(BaseModel):
    kind: Literal['revise'] = 'revise'
    evaluation: EvaluationModel
    update_reason: str
    steps: list[PlanStepDraft]


class PlannerDecision(BaseModel):
    """Planner output: either respond directly or produce goal plus plan."""

    action: DirectResponse | PlannedResponse = Field(
        description='Return DirectResponse if no tool use is needed, otherwise return PlannedResponse.'
    )


class ReplannerDecision(BaseModel):
    """Replanner output after evaluating one execution step."""

    action: FinalResponseDecision | PassDecision | ReviseDecision = Field(
        description='Return a final response, pass decision, or revised remaining steps.'
    )


def _copy_plan_steps(steps: list[PlanStepState]) -> list[PlanStepState]:
    return [deepcopy(step) for step in steps]


def _copy_plans(plans: list[PlanVersionState]) -> list[PlanVersionState]:
    return [
        {
            'version': plan['version'],
            'created_by': plan['created_by'],
            'reason': plan['reason'],
            'steps': _copy_plan_steps(plan.get('steps', [])),
        }
        for plan in plans
    ]


def _ensure_control(state: AgentState) -> ControlState:
    return {
        'step_count': state.get('control', {}).get('step_count', 0),
        'replan_time': state.get('control', {}).get('replan_time', 0),
        'status': state.get('control', {}).get('status', 'running'),
    }


def _increment_control(
    state: AgentState,
    *,
    increment_replan: bool = False,
    status: Literal['running', 'done', 'failed'] | None = None,
) -> ControlState:
    control = _ensure_control(state)
    control['step_count'] = control.get('step_count', 0) + 1
    if increment_replan:
        control['replan_time'] = control.get('replan_time', 0) + 1
    if status is not None:
        control['status'] = status
    return control


def _build_step_id(version: int, index: int) -> str:
    return f'v{version}_step_{index}'


def _build_plan_version(
    *,
    version: int,
    created_by: Literal['planner', 'replanner'],
    reason: str,
    steps: list[PlanStepDraft],
) -> PlanVersionState:
    return {
        'version': version,
        'created_by': created_by,
        'reason': reason,
        'steps': [
            {
                'step_id': _build_step_id(version, index),
                'title': step.title,
                'detail': step.detail,
                'status': 'pending',
                'depends_on': [],
            }
            for index, step in enumerate(steps, start=1)
        ],
    }


def _latest_plan(plans: list[PlanVersionState]) -> PlanVersionState | None:
    return plans[-1] if plans else None


def _find_plan(plans: list[PlanVersionState], version: int | None) -> PlanVersionState | None:
    if version is None:
        return None
    for plan in plans:
        if plan.get('version') == version:
            return plan
    return None


def _find_step(plan: PlanVersionState | None, step_id: str | None) -> PlanStepState | None:
    if not plan or not step_id:
        return None
    for step in plan.get('steps', []):
        if step.get('step_id') == step_id:
            return step
    return None


def _first_pending_step(plan: PlanVersionState | None) -> PlanStepState | None:
    if not plan:
        return None
    for step in plan.get('steps', []):
        if step.get('status') == 'pending':
            return step
    return None


def _current_pointer(state: AgentState) -> CurrentStepPointer:
    return state.get('execute_track', {}).get(
        'current_step', {'plan_version': None, 'step_id': None}
    )


def _build_execute_track(
    state: AgentState,
    *,
    current_plan_version: int | None,
    current_step_id: str | None,
    track: list[ExecutionLogState] | None = None,
) -> ExecuteTrackState:
    existing_track = state.get('execute_track', {}).get('track', [])
    return {
        'current_step': {
            'plan_version': current_plan_version,
            'step_id': current_step_id,
        },
        'track': track if track is not None else list(existing_track),
    }


def _append_execution_log(state: AgentState, log: ExecutionLogState) -> ExecuteTrackState:
    track = list(state.get('execute_track', {}).get('track', []))
    track.append(log)
    current_step = _current_pointer(state)
    return {
        'current_step': current_step,
        'track': track,
    }


def _latest_execution_log(state: AgentState) -> ExecutionLogState | None:
    track = state.get('execute_track', {}).get('track', [])
    return track[-1] if track else None


def _mark_step_status(
    plans: list[PlanVersionState],
    *,
    version: int | None,
    step_id: str | None,
    status: Literal['pending', 'running', 'done', 'failed'],
) -> list[PlanVersionState]:
    updated = _copy_plans(plans)
    plan = _find_plan(updated, version)
    step = _find_step(plan, step_id)
    if step:
        step['status'] = status
    return updated


def _build_replan_version(
    *,
    previous_plan: PlanVersionState,
    new_version: int,
    update_reason: str,
    revised_steps: list[PlanStepDraft],
    failed_step_id: str | None,
) -> PlanVersionState:
    preserved_done_steps = [
        deepcopy(step) for step in previous_plan.get('steps', []) if step.get('status') == 'done'
    ]
    new_steps = preserved_done_steps + [
        {
            'step_id': _build_step_id(new_version, offset),
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


def _goal_dict(goal: GoalSpecModel) -> GoalSpec:
    return {
        'goal': goal.goal,
        'expected_outcome': goal.expected_outcome,
        'constraints': list(goal.constraints),
    }


def _fallback_from_latest_execution(state: AgentState) -> str:
    latest_execution = _latest_execution_log(state)
    if latest_execution and latest_execution.get('output'):
        return latest_execution['output']
    return 'I could not produce a final answer.'


def _prefers_internal_search(goal: GoalSpec | None, step: PlanStepState) -> bool:
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


def _prefers_web_search(goal: GoalSpec | None, step: PlanStepState) -> bool:
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


def _build_executor_query(state: AgentState, step: PlanStepState) -> str:
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


def _extract_metadata_filter(step: PlanStepState) -> dict[str, Any] | None:
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


def _summarize_tool_output(tool_output: dict[str, Any]) -> str:
    summary = json.dumps(tool_output, default=str)
    if len(summary) > 400:
        return summary[:400].rstrip() + '...'
    return summary


def _clip_text(value: str, limit: int = 1200) -> str:
    cleaned = ' '.join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + '...'


def _render_tool_output_for_prompt(log: ToolUseLogState) -> str:
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
            chunk_text = _clip_text(str(item.get('text', '') or ''))
            if chunk_text:
                blocks.append(f'{prefix}\nText: {chunk_text}')
        if blocks:
            return '\n\n'.join(blocks)

    if tool_name == 'search_web':
        blocks = []
        for index, item in enumerate(tool_output.get('results', [])[:5], start=1):
            title = str(item.get('title') or f'web_result_{index}').strip()
            url = str(item.get('url') or '').strip()
            content = _clip_text(str(item.get('content', '') or ''))
            pieces = [f'Web result {index}: {title}']
            if url:
                pieces.append(f'URL: {url}')
            if content:
                pieces.append(f'Content: {content}')
            blocks.append('\n'.join(pieces))
        if blocks:
            return '\n\n'.join(blocks)

    return log.get('tool_output_summary', '')


def _sources_from_qdrant_output(tool_output: dict[str, Any]) -> list[dict[str, str]]:
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


def _sources_from_web_output(tool_output: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            'type': 'web',
            'label': str(item.get('title') or 'web source'),
            'location': str(item.get('url') or ''),
        }
        for item in tool_output.get('results', [])[:5]
        if item.get('url')
    ]


def _run_executor_tools(state: AgentState, step: PlanStepState) -> list[ToolUseLogState]:
    tool_logs: list[ToolUseLogState] = []
    config = get_plan_execute_config()
    query_text = _build_executor_query(state, step)
    metadata_filter = _extract_metadata_filter(step)
    use_internal = _prefers_internal_search(state.get('goal'), step)
    use_web = _prefers_web_search(state.get('goal'), step)

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
                'tool_output_summary': _summarize_tool_output(tool_output),
                'sources': _sources_from_qdrant_output(tool_output),
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
                'tool_output_summary': _summarize_tool_output(tool_output),
                'sources': _sources_from_web_output(tool_output),
                'status': 'success',
            }
        )

    return tool_logs


def _build_executor_synthesis_input(
    *,
    state: AgentState,
    step: PlanStepState,
    current_plan_version: int,
    tool_use_logs: list[ToolUseLogState],
) -> str:
    current_plan = _find_plan(state.get('plans', []), current_plan_version)
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
        explicit_output = _render_tool_output_for_prompt(log)
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


def _extract_named_section(text: str, section_name: str) -> str:
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


def _parse_carry_forward_notes(text: str) -> list[str]:
    section = _extract_named_section(text, 'Carry-forward notes')
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


def _parse_skip_stop_recommendation(text: str) -> tuple[bool, str | None]:
    recommendation_section = _extract_named_section(text, 'Skip-stop recommendation').lower()
    reason_section = _extract_named_section(text, 'Skip-stop reason') or None
    return recommendation_section.startswith('yes'), reason_section


def _parse_step_result_text(text: str) -> str:
    section = _extract_named_section(text, 'Step result')
    cleaned = section or text.strip()
    if '\nCarry-forward notes:' in cleaned:
        cleaned = cleaned.split('\nCarry-forward notes:', 1)[0].strip()
    if '\nSkip-stop recommendation:' in cleaned:
        cleaned = cleaned.split('\nSkip-stop recommendation:', 1)[0].strip()
    return cleaned or 'Executor returned an empty response.'


@dataclass
class PlanExecuteComponents:
    planner: Any
    executor: Any
    replanner: Any
    max_replans: int
    executor_recursion_limit: int

    @trace_function(
        name='planner_node',
        span_type=SpanType.CHAIN,
        attributes={'workflow': 'agentic', 'langgraph.node': 'planner'},
    )
    async def plan_step(self, state: AgentState) -> dict[str, Any]:
        _log_node_event('planner', 'enter', state)
        control = _increment_control(state)
        set_current_span_attributes(
            {
                'history_length': len(state.get('history', [])),
                'input_length': len(state['input']),
            }
        )
        history_text = format_history(state.get('history'))
        decision = await self.planner.ainvoke(
            {
                'objective': state['input'],
                'history': history_text,
            }
        )
        if isinstance(decision.action, DirectResponse):
            result = {
                'goal': _goal_dict(decision.action.goal),
                'response': decision.action.response,
                'plans': [],
                'execute_track': _build_execute_track(
                    state,
                    current_plan_version=None,
                    current_step_id=None,
                    track=[],
                ),
                'control': {**control, 'status': 'done'},
                'replan_history': [],
                'last_error': None,
            }
            set_current_span_outputs({'response': True, 'plan_steps': 0})
            _log_node_event('planner', 'exit', result, {'decision': 'direct'})
            return result

        initial_plan = _build_plan_version(
            version=1,
            created_by='planner',
            reason='Initial plan from planner',
            steps=decision.action.steps,
        )
        first_step = _first_pending_step(initial_plan)
        result = {
            'goal': _goal_dict(decision.action.goal),
            'plans': [initial_plan],
            'execute_track': _build_execute_track(
                state,
                current_plan_version=initial_plan['version'],
                current_step_id=first_step.get('step_id') if first_step else None,
                track=[],
            ),
            'control': control,
            'replan_history': [],
            'last_error': None,
        }
        set_current_span_outputs({'response': False, 'plan_steps': len(initial_plan['steps'])})
        _log_node_event(
            'planner',
            'exit',
            result,
            {'decision': 'plan', 'plan_version': initial_plan['version'], 'plan_steps': len(initial_plan['steps'])},
        )
        return result

    @trace_function(
        name='agent_node',
        span_type=SpanType.CHAIN,
        attributes={'workflow': 'agentic', 'langgraph.node': 'agent'},
    )
    async def execute_step(self, state: AgentState) -> dict[str, Any]:
        _log_node_event('agent', 'enter', state)
        pointer = _current_pointer(state)
        current_plan_version = pointer.get('plan_version')
        current_step_id = pointer.get('step_id')
        plans = state.get('plans', [])
        plan = _find_plan(plans, current_plan_version)
        step = _find_step(plan, current_step_id)
        if not plan or not step:
            result = {'control': _increment_control(state)}
            _log_node_event('agent', 'exit', {**state, **result}, {'reason': 'missing_plan_or_step'})
            return result

        plans_with_running = _mark_step_status(
            plans,
            version=current_plan_version,
            step_id=current_step_id,
            status='running',
        )

        control = _increment_control(state)
        set_current_span_attributes(
            {
                'plan_version': current_plan_version,
                'current_step': step.get('title', ''),
                'execution_log_count': len(state.get('execute_track', {}).get('track', [])),
            }
        )

        try:
            tool_use_logs = _run_executor_tools(state, step)
            sources = collect_sources_from_tool_use_logs(tool_use_logs)
            synthesis_input = _build_executor_synthesis_input(
                state=state,
                step=step,
                current_plan_version=current_plan_version or 0,
                tool_use_logs=tool_use_logs,
            )
            synthesis_result = await self.executor.ainvoke(
                {'executor_input': synthesis_input},
                config={'recursion_limit': self.executor_recursion_limit},
            )
            raw_result_text = str(getattr(synthesis_result, 'content', '')).strip() or 'Executor returned an empty response.'
            carry_forward_notes = _parse_carry_forward_notes(raw_result_text)
            skip_stop_recommended, skip_stop_reason = _parse_skip_stop_recommendation(raw_result_text)
            result_text = ensure_response_has_sources(_parse_step_result_text(raw_result_text), sources)
            execution_log: ExecutionLogState = {
                'plan_version': current_plan_version or 0,
                'step_id': step.get('step_id', ''),
                'step_title': step.get('title', ''),
                'step_detail': step.get('detail', ''),
                'output': result_text,
                'status': 'success',
                'error': None,
                'tool_use_logs': tool_use_logs,
                'sources': sources,
                'carry_forward_notes': carry_forward_notes,
                'skip_stop_recommended': skip_stop_recommended,
                'skip_stop_reason': skip_stop_reason,
                'evaluation': {
                    'last_status': 'partial',
                    'confidence': 0.6,
                    'reason': 'Executor completed without runtime failure. Final validation is delegated to the replanner.',
                },
                'possible_solutions': [],
            }
            execute_track = _append_execution_log(state, execution_log)
            result = {
                'plans': plans_with_running,
                'execute_track': execute_track,
                'control': control,
                'last_error': None,
            }
            set_current_span_outputs({'last_error': None, 'result_preview': result_text[:200]})
            _log_node_event(
                'agent',
                'exit',
                {**state, **result},
                {
                    'status': 'success',
                    'tool_calls': len(tool_use_logs),
                    'sources': sources,
                    'carry_forward_notes_count': len(carry_forward_notes),
                    'skip_stop_recommended': skip_stop_recommended,
                    'executor_recursion_limit': self.executor_recursion_limit,
                },
            )
            return result
        except Exception as exc:
            error_text = f'ERROR: {exc}'
            execution_log = {
                'plan_version': current_plan_version or 0,
                'step_id': step.get('step_id', ''),
                'step_title': step.get('title', ''),
                'step_detail': step.get('detail', ''),
                'output': error_text,
                'status': 'fail',
                'error': str(exc),
                'tool_use_logs': [],
                'sources': [],
                'carry_forward_notes': [],
                'skip_stop_recommended': False,
                'skip_stop_reason': None,
                'evaluation': {
                    'last_status': 'fail',
                    'confidence': 0.2,
                    'reason': 'Executor raised a runtime error while trying to complete the step.',
                },
                'possible_solutions': [
                    'Revise the retrieval query.',
                    'Try an alternate tool if appropriate.',
                    'Narrow the scope of the failed step.',
                ],
            }
            execute_track = _append_execution_log(state, execution_log)
            result = {
                'plans': plans_with_running,
                'execute_track': execute_track,
                'control': control,
                'last_error': str(exc),
            }
            set_current_span_outputs({'last_error': str(exc), 'result_preview': error_text[:200]})
            _log_node_event(
                'agent',
                'exit',
                {**state, **result},
                {'status': 'fail', 'error': str(exc), 'executor_recursion_limit': self.executor_recursion_limit},
            )
            return result

    @trace_function(
        name='replan_node',
        span_type=SpanType.CHAIN,
        attributes={'workflow': 'agentic', 'langgraph.node': 'replan'},
    )
    async def replan_step(self, state: AgentState) -> dict[str, Any]:
        _log_node_event('replan', 'enter', state)
        latest_execution = _latest_execution_log(state)
        pointer = _current_pointer(state)
        current_plan_version = pointer.get('plan_version')
        current_step_id = pointer.get('step_id')
        plans = state.get('plans', [])
        current_plan = _find_plan(plans, current_plan_version)
        current_step = _find_step(current_plan, current_step_id)
        control = _increment_control(state, increment_replan=True)
        current_replan_count = control.get('replan_time', 0)
        set_current_span_attributes(
            {
                'current_replan_count': current_replan_count,
                'max_replans': self.max_replans,
                'execution_log_count': len(state.get('execute_track', {}).get('track', [])),
            }
        )
        if current_replan_count > self.max_replans:
            fallback = self._build_fallback_response(state)
            result = {
                'response': fallback,
                'execute_track': _build_execute_track(
                    state,
                    current_plan_version=None,
                    current_step_id=None,
                ),
                'control': {**control, 'status': 'failed'},
            }
            set_current_span_outputs({'response': True, 'plan_steps': 0, 'fallback': True})
            _log_node_event('replan', 'exit', {**state, **result}, {'decision': 'fallback_failed'})
            return result

        history_text = format_history(state.get('history'))
        execution_track_text = format_execution_track(state.get('execute_track', {}).get('track', []))
        goal_text = format_goal(state.get('goal'))
        current_plan_text = format_plan_steps(current_plan.get('steps', []) if current_plan else [])
        replan_history_text = format_replan_history(state.get('replan_history', []))

        if not latest_execution:
            result = {'control': control}
            _log_node_event('replan', 'exit', {**state, **result}, {'reason': 'missing_latest_execution'})
            return result

        decision = await self.replanner.ainvoke(
            {
                'objective': state['input'],
                'goal': f'{goal_text}\n\nReplan history:\n{replan_history_text}',
                'history': history_text,
                'plan': current_plan_text,
                'past_steps': execution_track_text,
                'last_error': state.get('last_error') or 'None',
            }
        )

        if isinstance(decision.action, FinalResponseDecision):
            updated_plans = _mark_step_status(
                plans,
                version=current_plan_version,
                step_id=current_step_id,
                status='done',
            )
            final_response = ensure_response_has_sources(
                decision.action.response,
                latest_execution.get('sources', []),
            )
            result = {
                'response': final_response,
                'plans': updated_plans,
                'execute_track': _build_execute_track(
                    state,
                    current_plan_version=None,
                    current_step_id=None,
                ),
                'control': {**control, 'status': 'done'},
                'last_error': None,
            }
            set_current_span_outputs({'response': True, 'plan_steps': 0, 'fallback': False})
            _log_node_event('replan', 'exit', {**state, **result}, {'decision': 'final'})
            return result

        if isinstance(decision.action, PassDecision):
            updated_plans = _mark_step_status(
                plans,
                version=current_plan_version,
                step_id=current_step_id,
                status='done',
            )
            updated_current_plan = _find_plan(updated_plans, current_plan_version)
            next_step = _first_pending_step(updated_current_plan)
            if not next_step:
                final_response = ensure_response_has_sources(
                    latest_execution.get('output') or _fallback_from_latest_execution(state),
                    latest_execution.get('sources', []),
                )
                result = {
                    'response': final_response,
                    'plans': updated_plans,
                    'execute_track': _build_execute_track(
                        state,
                        current_plan_version=None,
                        current_step_id=None,
                    ),
                    'control': {**control, 'status': 'done'},
                    'last_error': None,
                }
                set_current_span_outputs({'response': True, 'plan_steps': 0, 'fallback': False})
                _log_node_event('replan', 'exit', {**state, **result}, {'decision': 'pass_and_finish'})
                return result

            result = {
                'plans': updated_plans,
                'execute_track': _build_execute_track(
                    state,
                    current_plan_version=current_plan_version,
                    current_step_id=next_step.get('step_id'),
                ),
                'control': control,
                'last_error': None,
            }
            set_current_span_outputs(
                {'response': False, 'plan_steps': len(updated_current_plan.get('steps', [])), 'fallback': False}
            )
            _log_node_event('replan', 'exit', {**state, **result}, {'decision': 'pass_and_continue'})
            return result

        updated_old_plans = _mark_step_status(
            plans,
            version=current_plan_version,
            step_id=current_step_id,
            status='failed',
        )
        previous_plan = _find_plan(updated_old_plans, current_plan_version)
        new_version_number = (previous_plan.get('version', 0) if previous_plan else 0) + 1
        revised_plan = _build_replan_version(
            previous_plan=previous_plan or {'version': 0, 'created_by': 'planner', 'reason': '', 'steps': []},
            new_version=new_version_number,
            update_reason=decision.action.update_reason,
            revised_steps=decision.action.steps,
            failed_step_id=current_step_id,
        )
        all_plans = _copy_plans(updated_old_plans)
        all_plans.append(revised_plan)
        next_step = _first_pending_step(revised_plan)
        replan_history = list(state.get('replan_history', []))
        replan_history.append(
            {
                'from_version': current_plan_version or 0,
                'to_version': revised_plan['version'],
                'failed_step_id': current_step_id,
                'old_step_snapshot': deepcopy(current_step) if current_step else None,
                'new_step_snapshot': deepcopy(next_step) if next_step else None,
                'reason': decision.action.update_reason,
            }
        )
        result = {
            'plans': all_plans,
            'execute_track': _build_execute_track(
                state,
                current_plan_version=revised_plan['version'],
                current_step_id=next_step.get('step_id') if next_step else None,
            ),
            'control': control,
            'replan_history': replan_history,
            'last_error': latest_execution.get('error') or decision.action.evaluation.reason,
        }
        set_current_span_outputs({'response': False, 'plan_steps': len(revised_plan['steps']), 'fallback': False})
        _log_node_event(
            'replan',
            'exit',
            {**state, **result},
            {'decision': 'revise', 'from_version': current_plan_version, 'to_version': revised_plan['version']},
        )
        return result

    def route_after_plan(self, state: AgentState) -> Literal['agent'] | str:
        if state.get('response'):
            return END
        current_step = state.get('execute_track', {}).get('current_step', {})
        if current_step.get('step_id'):
            return 'agent'
        return END

    def route_after_replan(self, state: AgentState) -> Literal['agent'] | str:
        if state.get('response'):
            return END
        current_step = state.get('execute_track', {}).get('current_step', {})
        if current_step.get('step_id') and state.get('control', {}).get('status') == 'running':
            return 'agent'
        return END

    def _build_fallback_response(self, state: AgentState) -> str:
        latest_execution = _latest_execution_log(state)
        if latest_execution:
            return (
                'I could not finish the full plan after several retries. '
                f"Last attempted step: {latest_execution.get('step_title', 'Unknown step')}.\n"
                f"Last result: {latest_execution.get('output', '')}"
            )
        return 'I could not complete the request after several replanning attempts.'


def build_plan_execute_components(max_replans: int | None = None) -> PlanExecuteComponents:
    config = get_plan_execute_config()
    planner, executor, replanner, executor_recursion_limit = build_plan_execute_runnables(
        planner_schema=PlannerDecision,
        replanner_schema=ReplannerDecision,
    )

    return PlanExecuteComponents(
        planner=planner,
        executor=executor,
        replanner=replanner,
        max_replans=max_replans if max_replans is not None else config.flow.max_replans,
        executor_recursion_limit=executor_recursion_limit,
    )