from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agentic_core.prompts import (
    EXECUTOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REPLANNER_SYSTEM_PROMPT,
)
from tools import get_query_qdrant_tool, get_search_web_tool


UTILS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = UTILS_DIR.parent
load_dotenv(BACKEND_DIR / '.env')


def get_model_name(env_name: str, default: str) -> str:
    value = os.getenv(env_name, default).strip()
    return value or default


def build_plan_execute_runnables(
    *,
    planner_schema: Any,
    replanner_schema: Any,
) -> tuple[Any, Any, Any]:
    planner_model = ChatOpenAI(
        model=get_model_name('AGENTIC_PLANNER_MODEL', 'gpt-4.1-mini'),
        temperature=0.2,
    )
    executor_model = ChatOpenAI(
        model=get_model_name('AGENTIC_EXECUTOR_MODEL', 'gpt-5-nano'),
        temperature=0.2,
    )
    replanner_model = ChatOpenAI(
        model=get_model_name('AGENTIC_REPLANNER_MODEL', 'gpt-4.1-mini'),
        temperature=0.2,
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
                'Conversation history:\n{history}\n\n'
                'Current remaining plan:\n{plan}\n\n'
                'Completed steps:\n{past_steps}\n\n'
                'Last error, if any:\n{last_error}',
            ),
        ]
    )

    planner = planner_prompt | planner_model.with_structured_output(planner_schema)
    replanner = replanner_prompt | replanner_model.with_structured_output(replanner_schema)
    tools = [get_search_web_tool(), get_query_qdrant_tool()]
    executor = create_react_agent(executor_model, tools, prompt=EXECUTOR_SYSTEM_PROMPT)
    return planner, executor, replanner