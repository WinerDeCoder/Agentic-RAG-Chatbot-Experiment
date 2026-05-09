from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
from typing import Annotated, Any, Literal

from langgraph.constants import END
from mlflow.entities import SpanType
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from config import get_plan_execute_config
from utils import (
    append_execution_log,
    build_execute_track,
    build_executor_synthesis_input,
    build_plan_version,
    build_plan_execute_runnables,
    build_replan_version,
    collect_sources_from_tool_use_logs,
    copy_plans,
    current_pointer,
    ensure_response_has_sources,
    fallback_from_latest_execution,
    find_plan,
    find_step,
    first_pending_step,
    format_execution_track,
    format_goal,
    format_history,
    format_plan_steps,
    format_replan_history,
    goal_dict,
    increment_control,
    latest_execution_log,
    log_node_event,
    mark_step_status,
    parse_carry_forward_notes,
    parse_skip_stop_recommendation,
    parse_step_result_text,
    run_executor_tools,
)
from observability.tracing import set_current_span_attributes, set_current_span_outputs, trace_function


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
        log_node_event(LOGGER, 'planner', 'enter', state)
        control = increment_control(state)
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
                'goal': goal_dict(decision.action.goal),
                'response': decision.action.response,
                'plans': [],
                'execute_track': build_execute_track(
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
            log_node_event(LOGGER, 'planner', 'exit', result, {'decision': 'direct'})
            return result

        initial_plan = build_plan_version(
            version=1,
            created_by='planner',
            reason='Initial plan from planner',
            steps=decision.action.steps,
        )
        first_step = first_pending_step(initial_plan)
        result = {
            'goal': goal_dict(decision.action.goal),
            'plans': [initial_plan],
            'execute_track': build_execute_track(
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
        log_node_event(
            LOGGER,
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
        log_node_event(LOGGER, 'agent', 'enter', state)
        pointer = current_pointer(state)
        current_plan_version = pointer.get('plan_version')
        current_step_id = pointer.get('step_id')
        plans = state.get('plans', [])
        plan = find_plan(plans, current_plan_version)
        step = find_step(plan, current_step_id)
        if not plan or not step:
            result = {'control': increment_control(state)}
            log_node_event(LOGGER, 'agent', 'exit', {**state, **result}, {'reason': 'missing_plan_or_step'})
            return result

        plans_with_running = mark_step_status(
            plans,
            version=current_plan_version,
            step_id=current_step_id,
            status='running',
        )

        control = increment_control(state)
        set_current_span_attributes(
            {
                'plan_version': current_plan_version,
                'current_step': step.get('title', ''),
                'execution_log_count': len(state.get('execute_track', {}).get('track', [])),
            }
        )

        try:
            tool_use_logs = run_executor_tools(state, step)
            sources = collect_sources_from_tool_use_logs(tool_use_logs)
            synthesis_input = build_executor_synthesis_input(
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
            carry_forward_notes = parse_carry_forward_notes(raw_result_text)
            skip_stop_recommended, skip_stop_reason = parse_skip_stop_recommendation(raw_result_text)
            result_text = ensure_response_has_sources(parse_step_result_text(raw_result_text), sources)
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
            execute_track = append_execution_log(state, execution_log)
            result = {
                'plans': plans_with_running,
                'execute_track': execute_track,
                'control': control,
                'last_error': None,
            }
            set_current_span_outputs({'last_error': None, 'result_preview': result_text[:200]})
            log_node_event(
                LOGGER,
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
            execute_track = append_execution_log(state, execution_log)
            result = {
                'plans': plans_with_running,
                'execute_track': execute_track,
                'control': control,
                'last_error': str(exc),
            }
            set_current_span_outputs({'last_error': str(exc), 'result_preview': error_text[:200]})
            log_node_event(
                LOGGER,
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
        log_node_event(LOGGER, 'replan', 'enter', state)
        latest_execution = latest_execution_log(state)
        pointer = current_pointer(state)
        current_plan_version = pointer.get('plan_version')
        current_step_id = pointer.get('step_id')
        plans = state.get('plans', [])
        current_plan = find_plan(plans, current_plan_version)
        current_step = find_step(current_plan, current_step_id)
        control = increment_control(state, increment_replan=True)
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
                'execute_track': build_execute_track(
                    state,
                    current_plan_version=None,
                    current_step_id=None,
                ),
                'control': {**control, 'status': 'failed'},
            }
            set_current_span_outputs({'response': True, 'plan_steps': 0, 'fallback': True})
            log_node_event(LOGGER, 'replan', 'exit', {**state, **result}, {'decision': 'fallback_failed'})
            return result

        history_text = format_history(state.get('history'))
        execution_track_text = format_execution_track(state.get('execute_track', {}).get('track', []))
        goal_text = format_goal(state.get('goal'))
        current_plan_text = format_plan_steps(current_plan.get('steps', []) if current_plan else [])
        replan_history_text = format_replan_history(state.get('replan_history', []))

        if not latest_execution:
            result = {'control': control}
            log_node_event(LOGGER, 'replan', 'exit', {**state, **result}, {'reason': 'missing_latest_execution'})
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
            updated_plans = mark_step_status(
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
                'execute_track': build_execute_track(
                    state,
                    current_plan_version=None,
                    current_step_id=None,
                ),
                'control': {**control, 'status': 'done'},
                'last_error': None,
            }
            set_current_span_outputs({'response': True, 'plan_steps': 0, 'fallback': False})
            log_node_event(LOGGER, 'replan', 'exit', {**state, **result}, {'decision': 'final'})
            return result

        if isinstance(decision.action, PassDecision):
            updated_plans = mark_step_status(
                plans,
                version=current_plan_version,
                step_id=current_step_id,
                status='done',
            )
            updated_current_plan = find_plan(updated_plans, current_plan_version)
            next_step = first_pending_step(updated_current_plan)
            if not next_step:
                final_response = ensure_response_has_sources(
                    latest_execution.get('output') or fallback_from_latest_execution(state),
                    latest_execution.get('sources', []),
                )
                result = {
                    'response': final_response,
                    'plans': updated_plans,
                    'execute_track': build_execute_track(
                        state,
                        current_plan_version=None,
                        current_step_id=None,
                    ),
                    'control': {**control, 'status': 'done'},
                    'last_error': None,
                }
                set_current_span_outputs({'response': True, 'plan_steps': 0, 'fallback': False})
                log_node_event(LOGGER, 'replan', 'exit', {**state, **result}, {'decision': 'pass_and_finish'})
                return result

            result = {
                'plans': updated_plans,
                'execute_track': build_execute_track(
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
            log_node_event(LOGGER, 'replan', 'exit', {**state, **result}, {'decision': 'pass_and_continue'})
            return result

        updated_old_plans = mark_step_status(
            plans,
            version=current_plan_version,
            step_id=current_step_id,
            status='failed',
        )
        previous_plan = find_plan(updated_old_plans, current_plan_version)
        new_version_number = (previous_plan.get('version', 0) if previous_plan else 0) + 1
        revised_plan = build_replan_version(
            previous_plan=previous_plan or {'version': 0, 'created_by': 'planner', 'reason': '', 'steps': []},
            new_version=new_version_number,
            update_reason=decision.action.update_reason,
            revised_steps=decision.action.steps,
            failed_step_id=current_step_id,
        )
        all_plans = copy_plans(updated_old_plans)
        all_plans.append(revised_plan)
        next_step = first_pending_step(revised_plan)
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
            'execute_track': build_execute_track(
                state,
                current_plan_version=revised_plan['version'],
                current_step_id=next_step.get('step_id') if next_step else None,
            ),
            'control': control,
            'replan_history': replan_history,
            'last_error': latest_execution.get('error') or decision.action.evaluation.reason,
        }
        set_current_span_outputs({'response': False, 'plan_steps': len(revised_plan['steps']), 'fallback': False})
        log_node_event(
            LOGGER,
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
        latest_execution = latest_execution_log(state)
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