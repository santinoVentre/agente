"""Project selection state machine.

When a WEB_DEV intent is detected and existing projects are present, the orchestrator
starts a ProjectSelectionState for that user. The next message from that user is
intercepted by the Telegram handler and resolved here before routing to either:
  - A new Q&A planning session (if user selects 0 / "nuovo")
  - The ProjectManagerAgent for the selected existing project
"""

from __future__ import annotations

from dataclasses import dataclass

from db.models import ProjectRegistry


@dataclass
class ProjectSelectionState:
    """Represents a pending project-selection step for a user."""

    user_id: int
    projects: list[ProjectRegistry]
    pending_message: str  # the original webdev request that triggered the selector

    def format_menu(self) -> str:
        lines = ["🗂️ <b>Scegli il progetto:</b>\n"]
        for i, p in enumerate(self.projects, 1):
            deploy_part = f"\n    🌐 {p.deploy_url}" if p.deploy_url else ""
            desc_part = f" — {p.description[:60]}" if p.description else ""
            lines.append(f"<b>{i}.</b> <code>{p.name}</code>{desc_part}{deploy_part}")
        lines.append("")
        lines.append("0️⃣  <b>0. Crea un nuovo progetto</b>")
        lines.append("")
        lines.append(
            "Rispondi con il <b>numero</b> o il <b>nome</b> del progetto.\n"
            "Per annullare: /annulla"
        )
        return "\n".join(lines)

    def resolve(self, text: str) -> ProjectRegistry | None | bool:
        """Resolve the user's text input to a selection.

        Returns:
          - ProjectRegistry: an existing project was chosen
          - None: user wants a new project (0 / "nuovo" / "new" / etc.)
          - False: input not recognised — keep selector active
        """
        t = text.strip().lower()

        # Abort / cancel
        if t in ("/annulla", "/abort", "/cancel", "annulla", "abort", "cancel"):
            return None  # treat as "new project" so the caller can handle gracefully

        # Numeric choice
        if t.isdigit():
            idx = int(t)
            if idx == 0:
                return None
            if 1 <= idx <= len(self.projects):
                return self.projects[idx - 1]
            return False  # out of range

        # "New" keywords
        if any(kw in t for kw in ("nuovo", "nuov", "new", "crea", "crea nuovo")):
            return None

        # Name match (partial, case-insensitive)
        for p in self.projects:
            if t in p.name.lower() or p.name.lower() in t:
                return p

        return False  # unrecognised


# ── Global selector registry ─────────────────────────────────────────────────

_active_selectors: dict[int, ProjectSelectionState] = {}


def get_selector(user_id: int) -> ProjectSelectionState | None:
    return _active_selectors.get(user_id)


def start_selector(
    user_id: int,
    projects: list[ProjectRegistry],
    pending_message: str,
) -> ProjectSelectionState:
    state = ProjectSelectionState(
        user_id=user_id,
        projects=projects,
        pending_message=pending_message,
    )
    _active_selectors[user_id] = state
    return state


def end_selector(user_id: int) -> None:
    _active_selectors.pop(user_id, None)
