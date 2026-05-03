from __future__ import annotations

import os
from typing import Any

import gradio as gr
import requests


API_BASE_URL = os.getenv("SIMPLE_CHAT_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("SIMPLE_CHAT_TIMEOUT_SECONDS", "180"))


def _handle_request_error(exc: requests.exceptions.RequestException) -> None:
    if isinstance(exc, requests.exceptions.ReadTimeout):
        raise gr.Error(
            (
                'The backend is still working, but the UI request timed out before it finished. '
                f'Current timeout: {REQUEST_TIMEOUT_SECONDS:.0f} seconds. '
                'Increase SIMPLE_CHAT_TIMEOUT_SECONDS if you want to allow longer agentic runs.'
            )
        ) from exc
    raise gr.Error(f'API request failed: {exc}') from exc


def api_get(path: str) -> Any:
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        _handle_request_error(exc)


def api_post(path: str, payload: dict[str, Any]) -> Any:
    try:
        response = requests.post(
            f"{API_BASE_URL}{path}",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        _handle_request_error(exc)


def session_choices(sessions: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [
        (f"{item['title']} ({item['message_count']} msgs)", item["session_id"])
        for item in sessions
    ]


def to_chatbot_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": item["role"], "content": item.get("content", "")}
        for item in messages
        if item.get("role") in {"user", "assistant"}
    ]


def refresh_sessions(selected_session_id: str | None):
    sessions = api_get("/sessions")
    choices = session_choices(sessions)
    if choices:
        valid_ids = {value for _, value in choices}
        selected_session_id = selected_session_id if selected_session_id in valid_ids else choices[0][1]
    else:
        selected_session_id = None
    return gr.update(choices=choices, value=selected_session_id), sessions


def load_session(session_id: str | None):
    if not session_id:
        return [], "No session selected. Create one to begin."
    session = api_get(f"/sessions/{session_id}")
    status = f"Loaded session '{session['title']}' with {len(session['messages'])} messages."
    return to_chatbot_messages(session["messages"]), status


def create_session(title: str):
    cleaned = title.strip()
    if not cleaned:
        raise gr.Error("Enter a session title first.")
    session = api_post("/sessions", {"title": cleaned})
    dropdown_update, sessions = refresh_sessions(session["session_id"])
    status = f"Created session '{session['title']}'."
    return dropdown_update, sessions, session["session_id"], [], "", status


def send_message(session_id: str | None, message: str, current_history: list[dict[str, str]]):
    cleaned = message.strip()
    if not session_id:
        raise gr.Error("Create or select a session before sending a message.")
    if not cleaned:
        raise gr.Error("Enter a message before sending.")

    response = api_post(f"/sessions/{session_id}/messages", {"message": cleaned})
    session = response["session"]
    chatbot_messages = to_chatbot_messages(session["messages"])
    status = f"Updated '{session['title']}' at {session['updated_at']}."
    return chatbot_messages, "", status


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Simple Chat UI") as app:
        gr.Markdown("# Simple Chat UI\n\nFastAPI backend + JSON session store + GPT-5-nano.")

        sessions_state = gr.State([])
        selected_session_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                session_dropdown = gr.Dropdown(label="Sessions", choices=[], value=None)
                refresh_button = gr.Button("Refresh Sessions")
                new_session_title = gr.Textbox(label="New Session Title", placeholder="Example: HR policy questions")
                create_button = gr.Button("Create Session", variant="primary")
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat History", height=520)
                message_box = gr.Textbox(label="Message", lines=4, placeholder="Type a message and press Send")
                send_button = gr.Button("Send", variant="primary")
                status_box = gr.Markdown("Ready.")

        app.load(
            refresh_sessions,
            inputs=[selected_session_state],
            outputs=[session_dropdown, sessions_state],
        ).then(
            load_session,
            inputs=[session_dropdown],
            outputs=[chatbot, status_box],
        )

        refresh_button.click(
            refresh_sessions,
            inputs=[session_dropdown],
            outputs=[session_dropdown, sessions_state],
        ).then(
            load_session,
            inputs=[session_dropdown],
            outputs=[chatbot, status_box],
        )

        session_dropdown.change(
            load_session,
            inputs=[session_dropdown],
            outputs=[chatbot, status_box],
        ).then(
            lambda value: value,
            inputs=[session_dropdown],
            outputs=[selected_session_state],
        )

        create_button.click(
            create_session,
            inputs=[new_session_title],
            outputs=[session_dropdown, sessions_state, selected_session_state, chatbot, new_session_title, status_box],
        ).then(
            load_session,
            inputs=[selected_session_state],
            outputs=[chatbot, status_box],
        )

        send_button.click(
            send_message,
            inputs=[session_dropdown, message_box, chatbot],
            outputs=[chatbot, message_box, status_box],
        ).then(
            refresh_sessions,
            inputs=[session_dropdown],
            outputs=[session_dropdown, sessions_state],
        )

    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860)
