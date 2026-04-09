"""Project Manager Agent — dedicated per-project orchestrator.

One PM agent instance exists per project. It:
- Is the sole point of contact with the client for modifications
- Has full persistent knowledge of the project (vision, stack, design, requirements)
- Decides which sub-agents to activate and what exact instructions to give them
- Never guesses: asks the client when uncertain
- Updates PM_CONTEXT.md after every significant change (changelog entry)
- Sub-agents (builder, reviewer, deployer) never see the client directly

Lifecycle:
  New project  → PM_CONTEXT.md written during Wave 0 of the initial build
  Modification → PM agent loaded from PM_CONTEXT.md, runs agentic loop
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agents.base_agent import BaseAgent
from config import config
from core.model_router import TaskType, get_model_for_task
from core.task_manager import task_manager
from db.models import ProjectRegistry, TaskStatus
from tg.notifications import notify
from utils.cost_tracker import cost_tracker
from utils.logging import setup_logging

log = setup_logging("project_manager_agent")

# ── PM behavioural rules injected into every system prompt ────────────────────

_PM_RULES = """\
## Regole operative (NON DEROGABILI)

Sei il **Project Manager dedicato** di questo progetto.
Hai piena e permanente conoscenza del progetto: visione, stack, design system, \
requisiti, decisioni architetturali, preferenze del cliente.

### Come lavori:
1. **Sei l'unico punto di contatto con il cliente.** I sub-agenti tecnici \
(builder, reviewer, deployer) non comunicano mai direttamente con lui.
2. **Analizza sempre la richiesta** rispetto al contesto: è in scope v1/v2? \
È coerente con il design system? Richiede modifiche architetturali?
3. **Prima di implementare qualsiasi cosa**, se la richiesta è ambigua, \
non completamente chiara o potenzialmente fuori scope → chiedi chiarimenti al cliente.
4. **Non indovinare mai.** Se hai il minimo dubbio su qualcosa di rilevante, \
chiedi. È meglio perdere 30 secondi che sbagliare l'implementazione.
5. **Il design system è sacro.** Colori, font, spacing, stile componenti sono \
NON negoziabili senza esplicita approvazione del cliente.
6. **Dopo ogni modifica completata**, aggiungi una riga al \
`## Storico modifiche` nel file `PM_CONTEXT.md` con data e descrizione.
7. **Se ti servono informazioni sul codebase** (struttura file, dipendenze, \
versioni), usa il tool `filesystem` per leggere direttamente dal workspace.

### Processo per ogni richiesta di modifica:
```
1. Analizza → capisce cosa vuole il cliente
2. Valuta scope → in scope v1? v2? fuori scope? modifica design system?
3. [Se ambiguo] → chiedi al cliente (specifica e conciso)
4. Pianifica → lista dei file da toccare + motivazione per ognuno
5. Implementa → modifica i file direttamente o tramite sub-agenti
6. Verifica → controlla che il risultato sia corretto e coerente
7. [Se serve deploy] → aggiorna il deploy su Vercel/GitHub
8. Riporta → invia al cliente un summary chiaro di cosa è stato fatto
9. Aggiorna PM_CONTEXT.md → aggiungi entry al changelog
```

### Tono con il cliente:
- Professionale ma diretto: spiega le tue decisioni brevemente
- Non tecnico: parla in termini di risultati, non di codice
- Proattivo: se vedi problemi potenziali, segnalali prima
"""


class ProjectManagerAgent(BaseAgent):
    """Dedicated PM agent for a specific project."""

    name = "project_manager"
    description = (
        "Project Manager dedicato — gestisce modifiche a un progetto esistente "
        "con piena conoscenza del contesto, del design system e dei requisiti."
    )
    default_task_type = TaskType.WEB_DEV
    max_iterations = 30
    ask_approval_on_iteration_limit = True
    ask_approval_on_loop = False
    max_same_tool_calls = 10

    def __init__(
        self,
        tools: dict,
        project_name: str,
        workdir: str,
        pm_context: str,
    ):
        super().__init__(tools)
        self.project_name = project_name
        self.workdir = workdir
        self.pm_context = pm_context  # content of PM_CONTEXT.md

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_project(
        cls, tools: dict, project: ProjectRegistry
    ) -> "ProjectManagerAgent | None":
        """Load PM agent from an existing registered project.

        Tries the workspace filesystem first, then falls back to metadata_json.
        Returns None if no PM context is available.
        """
        workdir = project.workspace_path or f"/srv/agent/workspaces/{project.name}"
        pm_ctx_path = Path(workdir) / "PM_CONTEXT.md"

        pm_context = ""

        # 1. Filesystem (preferred — most up-to-date)
        if pm_ctx_path.exists():
            try:
                pm_context = pm_ctx_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                log.warning(f"[pm_agent] Cannot read {pm_ctx_path}: {exc}")

        # 2. Fallback: metadata_json stored in DB
        if not pm_context:
            meta = project.metadata_json or {}
            pm_context = meta.get("pm_context", "")

        # 3. Minimal fallback: build basic context from registry fields
        if not pm_context:
            pm_context = (
                f"# Progetto: {project.name}\n\n"
                f"{project.description or 'Nessuna descrizione disponibile.'}\n\n"
                f"Workspace: {workdir}\n"
                f"GitHub: {project.github_repo or '—'}\n"
                f"Deploy: {project.deploy_url or '—'}\n\n"
                "_Nota: PM_CONTEXT.md non trovato. "
                "Usa il tool filesystem per analizzare il codebase prima di procedere._"
            )
            log.warning(
                f"[pm_agent] No PM_CONTEXT.md for project '{project.name}' — using minimal fallback"
            )

        return cls(
            tools=tools,
            project_name=project.name,
            workdir=workdir,
            pm_context=pm_context,
        )

    # ── System prompt ─────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return (
            f"{self.pm_context}\n\n"
            "---\n\n"
            f"{_PM_RULES}\n\n"
            "## Contesto operativo\n"
            f"- Workspace: `{self.workdir}`\n"
            f"- Data odierna: {today}\n"
            "- Tool disponibili: `filesystem` (read/write/list), `shell` (cmd), "
            "`github` (push/PR/create-repo), `vercel` (deploy), `telegram` (send_file), "
            "`browser` (web search), `monitoring` (VPS metrics)\n"
            "- Rispondi sempre in italiano al cliente.\n"
            "- Dopo ogni modifica: aggiorna `PM_CONTEXT.md` → sezione `## Storico modifiche`."
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    async def run(
        self,
        user_message: str,
        task_id: int,
        history: list[dict] | None = None,
        system_prompt: str = "",  # ignored — PM always uses its own context
        model_override: str | None = None,
        specs: dict | None = None,
        design_system: dict | None = None,
        state_files: dict | None = None,
    ) -> str:
        """Run the PM agentic loop for a modification / question request."""
        await notify(
            f"🧑‍💼 <b>PM Agent</b> — progetto <b>{self.project_name}</b>\n"
            f"Analisi richiesta in corso…"
        )
        await task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS, progress=5)

        return await super().run(
            user_message=user_message,
            task_id=task_id,
            history=history,
            system_prompt=self._build_system_prompt(),
            model_override=model_override or get_model_for_task(TaskType.WEB_DEV),
        )
