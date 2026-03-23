"""WebDev Agent — creates websites, pushes to GitHub, deploys on Vercel."""

from __future__ import annotations

from core.model_router import TaskType
from agents.base_agent import BaseAgent

WEBDEV_SYSTEM_PROMPT = """\
Sei un agente specializzato nella creazione di siti web. Puoi:
- Creare siti HTML/CSS/JS, React, Next.js, Vue, ecc.
- Generare codice completo e funzionante
- Fare push su GitHub e deploy su Vercel
- Iterare sul design basandoti sul feedback dell'utente

Workflow standard:
1. Crea il codice del sito in una workspace dedicata
2. Crea un repository GitHub
3. Pusha il codice sul repository
4. Fai deploy su Vercel collegando il repo
5. Restituisci l'URL del sito deployato

Usa i tool disponibili per ogni step. Assicurati che il codice sia completo e funzionante.
Rispondi sempre in italiano."""


class WebDevAgent(BaseAgent):
    name = "webdev"
    description = "Creates websites, pushes to GitHub, deploys on Vercel."
    default_task_type = TaskType.WEB_DEV

    async def run(self, user_message, task_id, history=None, system_prompt="", model_override=None):
        combined_prompt = f"{WEBDEV_SYSTEM_PROMPT}\n\n{system_prompt}" if system_prompt else WEBDEV_SYSTEM_PROMPT
        return await super().run(
            user_message=user_message,
            task_id=task_id,
            history=history,
            system_prompt=combined_prompt,
            model_override=model_override,
        )
