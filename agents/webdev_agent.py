"""WebDev Agent — planner → builder → reviewer pipeline per la creazione di siti web."""

from __future__ import annotations

import json

from config import config
from core.model_router import TaskType, get_model_for_task
from core.openrouter_client import openrouter
from core.task_manager import task_manager
from core.webdev_planner import build_design_system_prompt_section
from db.models import TaskStatus
from tg.notifications import notify
from agents.base_agent import BaseAgent
from utils.cost_tracker import cost_tracker
from utils.logging import setup_logging

log = setup_logging("webdev_agent")

# ── System prompts per ruolo ───────────────────────────────────────────────────

_PLANNER_PROMPT = """\
Sei un Web Architect. Il tuo unico compito è produrre un piano di sviluppo DETTAGLIATO in formato JSON.

Analizza la richiesta dell'utente e restituisci SOLO un oggetto JSON valido con questa struttura:
{
  "project_name": "nome-repo-kebab-case",
  "tech_stack": "es. Next.js + Tailwind CSS",
  "description": "descrizione breve del progetto",
  "files": [
    {"path": "percorso/relativo/file.ext", "description": "cosa contiene e perché"}
  ],
  "deploy_target": "vercel",
  "notes": "eventuali note tecniche importanti"
}

Non scrivere nulla fuori dal JSON. Sii preciso sui file da creare."""

_BUILDER_PROMPT = """\
Sei un Senior Full-Stack Developer. Riceverai un piano architetturale e devi implementarlo completamente.

Regole:
- Genera codice completo e funzionante, non placeholder né TODO
- Per ogni file usa il tool filesystem con action=write sul path indicato nel piano
- Crea prima la struttura di directory, poi i file
- Il codice deve essere production-ready
- Le dipendenze devono essere aggiornate: evita versioni vecchie o deprecated
- Prima di fissare versioni, verifica le versioni correnti via internet/shell
- Se una major e' nuovissima e rischiosa, scegli una minor/patch stabile molto recente
- Rispondi in italiano quando parli all'utente
- RISPETTA SCRUPOLOSAMENTE il design system fornito: colori, font, spacing, stile componenti
- Il sito deve sembrare fatto da un'agenzia top (Awwwards-level), non un template generico
- Implementa animazioni e micro-interazioni come indicato nel design system
- Mobile-first responsive OBBLIGATORIO in ogni file CSS/JSX

Workflow:
1. Crea la workspace directory
2. Verifica versioni librerie/framework in tempo reale:
    - npm: `npm view <package> version`
    - python: `pip index versions <package>` (oppure PyPI via browser)
    - docker images: controlla tag stabili recenti
3. Scrivi ogni file indicato nel piano con versioni aggiornate
4. Applica il design system a tutti i componenti visivi
5. Conferma quando hai finito, includendo le versioni scelte"""

_REVIEWER_PROMPT = """\
Sei un Senior Code Reviewer. Riceverai il codice generato da un developer e devi:

1. Leggere i file chiave del progetto con il tool filesystem action=read
2. Identificare problemi concreti: bug, sicurezza, performance, accessibilità, missing files
3. Verificare che le versioni delle dipendenze non siano obsolete
4. Se trovi versioni vecchie, aggiornale a versioni stabili recenti (con compatibilità ragionevole)
5. Applicare i fix direttamente modificando i file con action=write
6. Restituire un report finale con i fix applicati

Concentrati su problemi reali, non su preferenze stilistiche.
Rispondi in italiano."""

_DEPLOYER_PROMPT = """\
Sei un DevOps Engineer. Il tuo compito è pubblicare il progetto:

1. Crea un repository GitHub (privato di default)
2. Pusha tutti i file della workspace sul repo
3. Crea il deploy su Vercel collegando il repo GitHub
4. Restituisci l'URL pubblico del sito

Usa i tool github e vercel disponibili. Rispondi in italiano."""


class WebDevAgent(BaseAgent):
    name = "webdev"
    description = "Creates websites via planner→builder→reviewer pipeline, then deploys on Vercel."
    default_task_type = TaskType.WEB_DEV
    max_iterations = None
    ask_approval_on_iteration_limit = False
    max_same_tool_calls = 6
    loop_approval_threshold = 10
    ask_approval_on_loop = True

    # ── Pipeline helpers ───────────────────────────────────────────────────────

    async def _specs_to_file_plan(self, specs: dict, task_id: int) -> dict:
        """Convert Q&A specs into a detailed file plan via LLM."""
        _FILE_PLAN_PROMPT = """\
Sei un Web Architect. Dato un documento di specifiche di progetto, genera SOLO un JSON con la lista
completa dei file da creare. Struttura:
{
  "project_name": "nome-kebab-case",
  "tech_stack": "stack",
  "description": "breve descrizione",
  "deploy_target": "vercel",
  "files": [
    {"path": "src/app/page.tsx", "description": "Home page con tutte le sezioni"},
    {"path": "src/components/Hero.tsx", "description": "Hero component full-screen"},
    ...
  ],
  "notes": "note tecniche"
}
Includi TUTTI i file necessari: layout, pagine, componenti, stili, config, package.json, README.
Non omettere nulla. Sii preciso."""

        plan_raw = await self._llm_call(
            system_prompt=_FILE_PLAN_PROMPT,
            user_content=json.dumps(specs, indent=2, ensure_ascii=False),
            task_id=task_id,
            task_type=TaskType.COMPLEX_REASONING,
            temperature=0.15,
            max_tokens=2048,
        )
        try:
            js = plan_raw[plan_raw.find("{"):plan_raw.rfind("}") + 1]
            plan = json.loads(js)
            # Merge specs data into plan
            plan.setdefault("project_name", specs.get("project_name", "progetto-web"))
            plan.setdefault("tech_stack", specs.get("tech_stack", "Next.js 15 + Tailwind CSS 4"))
            return plan
        except (json.JSONDecodeError, ValueError):
            log.warning(f"[webdev] _specs_to_file_plan non JSON: {plan_raw[:200]}")
            return {
                "project_name": specs.get("project_name", "progetto-web"),
                "tech_stack": specs.get("tech_stack", "Next.js 15 + Tailwind CSS 4"),
                "description": specs.get("description", ""),
                "files": [],
            }

    async def _llm_call(
        self,
        system_prompt: str,
        user_content: str,
        task_id: int,
        task_type: TaskType = TaskType.CODE_GENERATION,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Single non-tool LLM call for planner/reviewer steps."""
        model = get_model_for_task(task_type)
        response = await openrouter.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            task_id=task_id,
        )
        return response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    async def _run_phase(
        self,
        phase_name: str,
        system_prompt: str,
        user_content: str,
        task_id: int,
        task_type: TaskType = TaskType.CODE_GENERATION,
        progress: int = 0,
    ) -> str:
        """Run a tool-using phase (builder / deployer) with full BaseAgent loop."""
        await notify(f"🔧 <b>WebDev</b> — fase <b>{phase_name}</b> avviata")
        await task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS, progress=progress)
        return await super().run(
            user_message=user_content,
            task_id=task_id,
            system_prompt=system_prompt,
            model_override=get_model_for_task(task_type),
        )

    # ── Main entry point ───────────────────────────────────────────────────────

    async def run(
        self,
        user_message: str,
        task_id: int,
        history: list[dict] | None = None,
        system_prompt: str = "",
        model_override: str | None = None,
        specs: dict | None = None,
        design_system: dict | None = None,
    ) -> str:

        # ── FASE 1: PLANNER ──────────────────────────────────────────────────
        await task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS, progress=5)

        if specs:
            # Specs already provided by the Q&A planner — skip LLM planner
            await notify("🗺️ <b>WebDev</b> — <b>FASE 1/4</b>: Specifiche già pronte, generazione file list…")
            plan = await self._specs_to_file_plan(specs, task_id)
        else:
            # No specs: fall back to LLM-based planning from raw message
            await notify("🗺️ <b>WebDev</b> — <b>FASE 1/4</b>: Pianificazione architettura…")
            plan_raw = await self._llm_call(
                system_prompt=_PLANNER_PROMPT,
                user_content=user_message,
                task_id=task_id,
                task_type=TaskType.COMPLEX_REASONING,
                temperature=0.2,
                max_tokens=2048,
            )
            try:
                json_start = plan_raw.find("{")
                json_end = plan_raw.rfind("}") + 1
                plan: dict = json.loads(plan_raw[json_start:json_end])
            except (json.JSONDecodeError, ValueError):
                log.warning(f"[webdev] Planner non JSON: {plan_raw[:200]}")
                plan = {"project_name": "progetto-web", "tech_stack": "Next.js 15 + Tailwind CSS 4", "files": []}

        project_name = plan.get("project_name", "progetto-web")
        tech_stack = plan.get("tech_stack", "")
        workdir = f"/srv/agent/workspaces/{project_name}"

        await notify(
            f"✅ <b>Piano pronto</b>\n"
            f"📦 Progetto: <code>{project_name}</code>\n"
            f"🛠️ Stack: {tech_stack}\n"
            f"📄 File pianificati: {len(plan.get('files', []))}"
        )

        # ── FASE 2: BUILDER ──────────────────────────────────────────────────
        await notify("👨‍💻 <b>WebDev</b> — <b>FASE 2/4</b>: Sviluppo codice…")
        await task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS, progress=20)

        # Inject design system if available
        design_system_section = ""
        if design_system:
            design_system_section = build_design_system_prompt_section(design_system)
        elif specs and specs.get("design_system"):
            design_system_section = build_design_system_prompt_section(specs["design_system"])

        # Build rich specs context if available
        specs_context = ""
        if specs:
            specs_context = (
                f"\n\n## Specifiche di progetto\n"
                f"Business: {specs.get('business_name', '')}\n"
                f"Scopo: {specs.get('purpose', '')}\n"
                f"Target: {specs.get('target_audience', '')}\n"
                f"Tone: {specs.get('tone_of_voice', '')}\n"
                f"Lingua: {specs.get('copy_language', 'it')}\n"
                f"SEO keywords: {', '.join(specs.get('seo_keywords', []))}\n"
                f"Content: {json.dumps(specs.get('content_strategy', {}), ensure_ascii=False)}\n"
                f"\nPagine dettagliate:\n"
                + json.dumps(specs.get("pages", []), indent=2, ensure_ascii=False)
            )

        builder_context = (
            f"Richiesta originale: {user_message}\n\n"
            f"Piano architetturale:\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n\n"
            f"Directory di lavoro: {workdir}\n\n"
            f"Data corrente: 2026-04-03\n"
            f"Prima di scrivere package.json/requirements, controlla versioni correnti online "
            f"(npm view, pip index, browser) e usa dipendenze aggiornate o quasi.\n"
            f"{specs_context}"
            f"{design_system_section}\n\n"
            f"Crea tutti i file indicati nel piano rispettando ESATTAMENTE il design system."
        )

        build_result = await self._run_phase(
            phase_name="Build",
            system_prompt=_BUILDER_PROMPT,
            user_content=builder_context,
            task_id=task_id,
            task_type=TaskType.CODE_GENERATION,
            progress=20,
        )

        # ── FASE 3: REVIEWER ────────────────────────────────────────────────
        await notify("🔍 <b>WebDev</b> — <b>FASE 3/4</b>: Code review e fix…")
        await task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS, progress=60)

        reviewer_context = (
            f"Progetto: {project_name} ({tech_stack})\n"
            f"Directory: {workdir}\n"
            f"File attesi: {json.dumps([f['path'] for f in plan.get('files', [])], ensure_ascii=False)}\n\n"
            f"Controlla esplicitamente package.json, lockfile e requirements.* contro versioni correnti. "
            f"Aggiorna dipendenze obsolete a release stabili recenti e poi restituisci un report."
        )

        review_result = await self._run_phase(
            phase_name="Review",
            system_prompt=_REVIEWER_PROMPT,
            user_content=reviewer_context,
            task_id=task_id,
            task_type=TaskType.CODE_REVIEW,
            progress=60,
        )

        # ── FASE 4: DEPLOYER ────────────────────────────────────────────────
        await notify("🚀 <b>WebDev</b> — <b>FASE 4/4</b>: Deploy su GitHub + Vercel…")
        await task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS, progress=80)

        deployer_context = (
            f"Progetto: {project_name}\n"
            f"Descrizione: {plan.get('description', '')}\n"
            f"Directory locale: {workdir}\n"
            f"File da pushare: {json.dumps([f['path'] for f in plan.get('files', [])], ensure_ascii=False)}\n\n"
            f"Crea il repo GitHub '{project_name}', pusha i file e fai il deploy su Vercel."
        )

        deploy_result = await self._run_phase(
            phase_name="Deploy",
            system_prompt=_DEPLOYER_PROMPT,
            user_content=deployer_context,
            task_id=task_id,
            task_type=TaskType.TOOL_EXECUTION,
            progress=80,
        )

        await task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS, progress=100)

        # ── REPORT FINALE ────────────────────────────────────────────────────
        task_cost = cost_tracker.get_task_cost(task_id)
        return (
            f"✅ <b>Progetto completato: {project_name}</b>\n\n"
            f"🛠️ Stack: {tech_stack}\n\n"
            f"<b>Review:</b>\n{review_result[:800]}\n\n"
            f"<b>Deploy:</b>\n{deploy_result[:800]}\n\n"
            f"💰 Costo totale task: ${task_cost:.3f}"
        )
