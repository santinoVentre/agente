"""System Agent — manages the VPS, runs shell commands, handles self-upgrade."""

from __future__ import annotations

from core.model_router import TaskType
from agents.base_agent import BaseAgent

SYSTEM_SYSTEM_PROMPT = """\
Sei un agente specializzato nella gestione del server VPS. Puoi:
- Eseguire comandi shell sul server
- Gestire file e directory
- Monitorare risorse di sistema (CPU, RAM, disco)
- Installare pacchetti e aggiornare il sistema
- Gestire servizi systemd
- Controllare i container Docker
- Gestire la sicurezza del server
- Leggere e analizzare log di sistema

IMPORTANTE — Sicurezza:
- MAI eseguire comandi distruttivi senza approvazione (rm -rf, mkfs, ecc.)
- MAI modificare configurazioni SSH, firewall o sudoers senza approvazione
- Preferisci comandi di sola lettura quando possibile
- Logga tutte le azioni eseguite

Quando ti viene chiesto lo stato del server, raccogli info su CPU, RAM,
disco, uptime, container running, e servizi attivi.
Rispondi sempre in italiano."""


class SystemAgent(BaseAgent):
    name = "system"
    description = "Manages the VPS: shell, files, monitoring, Docker, security."
    default_task_type = TaskType.SYSTEM

    async def run(self, user_message, task_id, history=None, system_prompt="", model_override=None):
        combined_prompt = f"{SYSTEM_SYSTEM_PROMPT}\n\n{system_prompt}" if system_prompt else SYSTEM_SYSTEM_PROMPT
        return await super().run(
            user_message=user_message,
            task_id=task_id,
            history=history,
            system_prompt=combined_prompt,
            model_override=model_override,
        )
