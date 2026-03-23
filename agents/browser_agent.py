"""Browser Agent — navigates the web, scrapes content, takes screenshots."""

from __future__ import annotations

from core.model_router import TaskType
from agents.base_agent import BaseAgent

BROWSER_SYSTEM_PROMPT = """\
Sei un agente specializzato nella navigazione web. Puoi:
- Navigare su qualsiasi sito web
- Estrarre testo e HTML dalle pagine
- Fare screenshot di pagine
- Scaricare file dal web
- Cercare informazioni online

Usa i tool browser per completare i task. Quando cerchi informazioni,
naviga su più siti se necessario per ottenere risultati completi e affidabili.
Rispondi sempre in italiano."""


class BrowserAgent(BaseAgent):
    name = "browser"
    description = "Browses the web, scrapes content, takes screenshots."
    default_task_type = TaskType.WEB_BROWSING

    async def run(self, user_message, task_id, history=None, system_prompt="", model_override=None):
        combined_prompt = f"{BROWSER_SYSTEM_PROMPT}\n\n{system_prompt}" if system_prompt else BROWSER_SYSTEM_PROMPT
        return await super().run(
            user_message=user_message,
            task_id=task_id,
            history=history,
            system_prompt=combined_prompt,
            model_override=model_override,
        )
