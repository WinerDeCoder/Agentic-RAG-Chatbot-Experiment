from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agentic_core.prompts import (
    EXECUTOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REPLANNER_SYSTEM_PROMPT,
)
from config import get_plan_execute_config


UTILS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = UTILS_DIR.parent
load_dotenv(BACKEND_DIR / '.env')


def build_chat_model(*, model_name: str, temperature: float, reasoning_effort: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )


def executor_recursion_limit(max_turns: int) -> int:
    return max(2 * max_turns + 1, 3)


def build_plan_execute_runnables(
    *,
    planner_schema: Any,
    replanner_schema: Any,
) -> tuple[Any, Any, Any, int]:
    config = get_plan_execute_config()
    planner_model = build_chat_model(
        model_name=config.planner_model.name,
        temperature=config.planner_model.temperature,
        reasoning_effort=config.planner_model.reasoning_effort,
    )
    executor_model = build_chat_model(
        model_name=config.executor_model.name,
        temperature=config.executor_model.temperature,
        reasoning_effort=config.executor_model.reasoning_effort,
    )
    replanner_model = build_chat_model(
        model_name=config.replanner_model.name,
        temperature=config.replanner_model.temperature,
        reasoning_effort=config.replanner_model.reasoning_effort,
    )

    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', PLANNER_SYSTEM_PROMPT),
            (
                'human',
                'Objective:\n{objective}\n\nConversation history:\n{history}',
            ),
        ]
    )
    replanner_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', REPLANNER_SYSTEM_PROMPT),
            (
                'human',
                'Objective:\n{objective}\n\n'
                'Goal:\n{goal}\n\n'
                'Conversation history:\n{history}\n\n'
                'Current plan version:\n{plan}\n\n'
                'Latest execution logs:\n{past_steps}\n\n'
                'Last error, if any:\n{last_error}',
            ),
        ]
    )

    planner = planner_prompt | planner_model.with_structured_output(planner_schema)
    replanner = replanner_prompt | replanner_model.with_structured_output(replanner_schema)
    executor_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', EXECUTOR_SYSTEM_PROMPT),
            ('human', '{executor_input}'),
        ]
    )
    executor = executor_prompt | executor_model
    return planner, executor, replanner, executor_recursion_limit(config.flow.executor_max_turns)