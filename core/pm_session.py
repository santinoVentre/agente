"""Runtime PM chat session state.

Keeps a per-user active Project Manager conversation so every new message is
routed to the PM agent until the session is explicitly terminated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from db.models import ProjectRegistry


@dataclass
class PMSessionState:
    user_id: int
    project: ProjectRegistry
    chat_id: int
    history: list[dict] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text[:4000]})
        self.history = self.history[-12:]

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text[:4000]})
        self.history = self.history[-12:]


_active_pm_sessions: dict[int, PMSessionState] = {}


def get_pm_session(user_id: int) -> PMSessionState | None:
    return _active_pm_sessions.get(user_id)


def start_pm_session(user_id: int, project: ProjectRegistry, chat_id: int) -> PMSessionState:
    state = PMSessionState(user_id=user_id, project=project, chat_id=chat_id)
    _active_pm_sessions[user_id] = state
    return state


def end_pm_session(user_id: int) -> None:
    _active_pm_sessions.pop(user_id, None)
