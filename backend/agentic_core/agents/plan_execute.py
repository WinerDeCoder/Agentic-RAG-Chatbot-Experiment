from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from langgraph.constants import END
from mlflow.entities import SpanType
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from utils import (
    build_execution_task_prompt,
    build_plan_execute_runnables,
    extract_last_message_text,
    format_history,
    format_past_steps,
)
from observability.tracing import set_current_span_attributes, set_current_span_outputs, trace_function


class AgentState(TypedDict, total=False):
    input: str
    history: list[dict[str, Any]]
    plan: list[str]
    past_steps: Annotated[list[tuple[str, str]], operator.add]
    response: str
    last_error: str | None
    replan_count: int


class Plan(BaseModel):
    """A list of execution steps."""

    steps: list[str] = Field(description='Ordered steps to solve the user request.')


class DirectResponse(BaseModel):
    """A direct answer to the user without any more tool use."""

    response: str = Field(description='Final response for the user.')


class PlannerDecision(BaseModel):
    """Planner output: either respond directly or produce a plan."""

    action: DirectResponse | Plan = Field(
        description='Return DirectResponse if no tool use is needed, otherwise return Plan.'
    )


class ReplannerDecision(BaseModel):
    """Replanner output after one execution step."""

    action: DirectResponse | Plan = Field(
        description='Return DirectResponse if enough information exists, otherwise return the remaining Plan.'
    )


@dataclass
class PlanExecuteComponents:
    planner: Any
    executor: Any
    replanner: Any
    max_replans: int

    @trace_function(
        name='planner_node',
        span_type=SpanType.CHAIN,
        attributes={'workflow': 'agentic', 'langgraph.node': 'planner'},
    )
    async def plan_step(self, state: AgentState) -> dict[str, Any]:
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
                'response': decision.action.response,
                'plan': [],
                'replan_count': 0,
                'last_error': None,
            }
            set_current_span_outputs({'response': True, 'plan_steps': 0})
            return result

        result = {
            'plan': decision.action.steps,
            'replan_count': 0,
            'last_error': None,
        }
        set_current_span_outputs({'response': False, 'plan_steps': len(result['plan'])})
        return result

    @trace_function(
        name='agent_node',
        span_type=SpanType.CHAIN,
        attributes={'workflow': 'agentic', 'langgraph.node': 'agent'},
    )
    async def execute_step(self, state: AgentState) -> dict[str, Any]:
        plan = state.get('plan', [])
        if not plan:
            return {}

        current_step = plan[0]
        plan_text = '\n'.join(f'{index}. {step}' for index, step in enumerate(plan, start=1))
        history_text = format_history(state.get('history'))
        past_steps_text = format_past_steps(state.get('past_steps'))

        task_prompt = build_execution_task_prompt(
            objective=state['input'],
            history_text=history_text,
            plan_text=plan_text,
            past_steps_text=past_steps_text,
            current_step=current_step,
        )
        set_current_span_attributes(
            {
                'plan_length': len(plan),
                'current_step': current_step,
                'past_steps_length': len(state.get('past_steps', [])),
            }
        )

        try:
            agent_result = await self.executor.ainvoke({'messages': [('user', task_prompt)]})
            result_text = extract_last_message_text(agent_result)
            result = {
                'past_steps': [(current_step, result_text)],
                'last_error': None,
            }
            set_current_span_outputs({'last_error': None, 'result_preview': result_text[:200]})
            return result
        except Exception as exc:
            error_text = f'ERROR: {exc}'
            result = {
                'past_steps': [(current_step, error_text)],
                'last_error': str(exc),
            }
            set_current_span_outputs({'last_error': str(exc), 'result_preview': error_text[:200]})
            return result

    @trace_function(
        name='replan_node',
        span_type=SpanType.CHAIN,
        attributes={'workflow': 'agentic', 'langgraph.node': 'replan'},
    )
    async def replan_step(self, state: AgentState) -> dict[str, Any]:
        current_replan_count = state.get('replan_count', 0)
        current_plan = state.get('plan', [])
        remaining_plan = current_plan[1:] if current_plan else []
        past_steps = state.get('past_steps', [])
        set_current_span_attributes(
            {
                'current_replan_count': current_replan_count,
                'max_replans': self.max_replans,
                'past_steps_length': len(past_steps),
            }
        )
        if current_replan_count >= self.max_replans:
            fallback = self._build_fallback_response(state)
            result = {
                'response': fallback,
                'plan': [],
                'replan_count': current_replan_count,
            }
            set_current_span_outputs({'response': True, 'plan_steps': 0, 'fallback': True})
            return result

        next_replan_count = current_replan_count + 1

        history_text = format_history(state.get('history'))
        past_steps_text = format_past_steps(past_steps)
        current_plan_text = '\n'.join(remaining_plan) or 'No remaining plan.'

        # If the executed step was the last remaining step and it succeeded,
        # prefer the concrete executor result over another model rewrite.
        if not remaining_plan and past_steps:
            last_step, last_result = past_steps[-1]
            if not str(last_result).startswith('ERROR:'):
                result = {
                    'response': last_result,
                    'plan': [],
                    'replan_count': next_replan_count,
                    'last_error': None,
                }
                set_current_span_outputs(
                    {
                        'response': True,
                        'plan_steps': 0,
                        'fallback': False,
                        'finalized_from_last_step': True,
                        'last_step': last_step,
                    }
                )
                return result

        decision = await self.replanner.ainvoke(
            {
                'objective': state['input'],
                'history': history_text,
                'plan': current_plan_text,
                'past_steps': past_steps_text,
                'last_error': state.get('last_error') or 'None',
            }
        )

        if isinstance(decision.action, DirectResponse):
            result = {
                'response': decision.action.response,
                'plan': [],
                'replan_count': next_replan_count,
                'last_error': None,
            }
            set_current_span_outputs({'response': True, 'plan_steps': 0, 'fallback': False})
            return result

        result = {
            'plan': decision.action.steps,
            'replan_count': next_replan_count,
            'last_error': None,
        }
        set_current_span_outputs({'response': False, 'plan_steps': len(result['plan']), 'fallback': False})
        return result

    def route_after_plan(self, state: AgentState) -> Literal['agent'] | str:
        if state.get('response'):
            return END
        if state.get('plan'):
            return 'agent'
        return END

    def route_after_replan(self, state: AgentState) -> Literal['agent'] | str:
        if state.get('response'):
            return END
        if state.get('plan'):
            return 'agent'
        return END

    def _build_fallback_response(self, state: AgentState) -> str:
        past_steps = state.get('past_steps', [])
        if past_steps:
            last_step, last_result = past_steps[-1]
            return (
                'I could not finish the full plan after several retries. '
                f'Last attempted step: {last_step}.\n'
                f'Last result: {last_result}'
            )
        return 'I could not complete the request after several replanning attempts.'


def build_plan_execute_components(max_replans: int = 3) -> PlanExecuteComponents:
    planner, executor, replanner = build_plan_execute_runnables(
        planner_schema=PlannerDecision,
        replanner_schema=ReplannerDecision,
    )

    return PlanExecuteComponents(
        planner=planner,
        executor=executor,
        replanner=replanner,
        max_replans=max_replans,
    )