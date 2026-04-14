"""WebDev Agent — planner → builder → reviewer pipeline per la creazione di siti web."""

from __future__ import annotations

import json

from config import config
from core.model_router import TaskType, get_model_for_task
from core.openrouter_client import openrouter
from core.project_registry import project_registry
from core.task_manager import task_manager
from core.webdev_planner import build_design_system_prompt_section
from db.models import TaskStatus
from tg.notifications import notify
from agents.base_agent import BaseAgent
from utils.cost_tracker import cost_tracker
from utils.logging import setup_logging

log = setup_logging("webdev_agent")

_WEB_STACK_POLICY = """\
Policy versioni web moderne e compatibili:
- Default preferito per nuovi siti: Node.js 22 LTS.
- Se usi Next.js moderno, preferisci stack coerente tipo: Next.js 15.x + React 19.x + React DOM 19.x + TypeScript 5.x.
- Se usi Tailwind CSS 4, il setup del progetto, PostCSS e i file CSS devono essere compatibili con Tailwind 4. Non mischiare config o sintassi di Tailwind 3 se hai scelto Tailwind 4.
- Se scegli una versione precedente di un framework per stabilita', adatta anche il codice scritto a QUELLA versione. Non usare API, convenzioni o sintassi di release piu' recenti.
- Package manager: preferisci npm recente stabile coerente con Node 22; dichiara sempre `engines.node` in package.json.
- Per progetti Node/Next crea SEMPRE anche `.nvmrc` con la major Node scelta.
- Evita stack vecchi senza motivo: niente Node 16/18, niente Next 13/14, niente React 17/18 per nuovi progetti salvo vincoli espliciti.
- Prima di chiudere il lavoro, verifica la coerenza tra: package.json, lockfile, `.nvmrc`, framework scelto, API usate nel codice, config Tailwind/PostCSS/Next/TypeScript.
"""

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

_PLANNER_PROMPT += "\n\n" + _WEB_STACK_POLICY + "\nNel campo tech_stack indica uno stack coerente e recente, non generico."

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
- Inserisci in package.json versioni coerenti con il codice generato, non solo "recenti" in astratto

Workflow:
1. Crea la workspace directory
2. Verifica versioni librerie/framework in tempo reale:
    - npm: `npm view <package> version`
    - python: `pip index versions <package>` (oppure PyPI via browser)
    - docker images: controlla tag stabili recenti
3. Scrivi ogni file indicato nel piano con versioni aggiornate
4. Applica il design system a tutti i componenti visivi
5. Conferma quando hai finito, includendo le versioni scelte"""

_BUILDER_PROMPT += "\n\n" + _WEB_STACK_POLICY + "\nSe scrivi package.json devi includere anche `engines.node` e creare `.nvmrc`."

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

_REVIEWER_PROMPT += "\n\n" + _WEB_STACK_POLICY + "\nSe trovi mismatch tra framework/runtime/codice, correggili direttamente."

_DEPLOYER_PROMPT = """\
Sei un DevOps Engineer. Il tuo compito è pubblicare il progetto:

1. Valida prima le integrazioni con `github(action=validate_auth)` e `vercel(action=validate_auth)`
2. Crea o riusa il repository GitHub con `github(action=create_repo)`
3. Pusha TUTTA la workspace con una sola chiamata `github(action=push_directory, source_dir=...)`
4. Crea o riusa il progetto Vercel con `vercel(action=deploy_from_github)`
5. Verifica che esista una deployment reale e restituisci l'URL effettivo
6. Restituisci l'URL pubblico del sito

Se GitHub push o Vercel deploy falliscono, spiega il motivo reale ricevuto dal tool e NON dichiarare successo.
Usa i tool github e vercel disponibili. Rispondi in italiano."""


class WebDevAgent(BaseAgent):
    name = "webdev"
    description = "Creates websites via planner→builder→reviewer pipeline, then deploys on Vercel."
    default_task_type = TaskType.WEB_DEV
    max_iterations = None
    max_steps_per_task = 40
    max_tokens_per_task = config.max_tokens_per_web_task
    ask_approval_on_iteration_limit = False
    max_same_tool_calls = 6
    loop_approval_threshold = 10
    ask_approval_on_loop = True

    # ── Pipeline helpers ───────────────────────────────────────────────────────

    async def _verify_publish(self, project_name: str, task_id: int) -> dict:
        """Verify GitHub repo and Vercel deployment with real tool calls."""
        github_result = await self.execute_tool(
            "github",
            {"action": "get_repo", "repo_name": project_name},
            task_id,
        )
        vercel_project = await self.execute_tool(
            "vercel",
            {"action": "get_project", "project_name": project_name},
            task_id,
        )
        vercel_deployments = await self.execute_tool(
            "vercel",
            {"action": "list_deployments", "project_name": project_name},
            task_id,
        )

        repo_url = ""
        github_repo_ref = ""
        if github_result.get("success"):
            repo = github_result.get("repo", {})
            repo_url = repo.get("html_url", "")
            full_name = repo.get("full_name", "")
            github_repo_ref = full_name or (f"{config.github_owner}/{project_name}" if repo_url else "")

        project_exists = vercel_project.get("success", False)
        deployments = vercel_deployments.get("deployments", []) if vercel_deployments.get("success") else []
        ready_deployment = None
        latest_deployment = deployments[0] if deployments else None
        for deployment in deployments:
            state = str(deployment.get("state", "")).lower()
            if state == "ready":
                ready_deployment = deployment
                break

        chosen_deployment = ready_deployment or latest_deployment
        deploy_url = ""
        deploy_state = "missing"
        if chosen_deployment and chosen_deployment.get("url"):
            deploy_url = f"https://{chosen_deployment['url']}"
            deploy_state = str(chosen_deployment.get("state", "unknown")).lower()

        verified = bool(repo_url and ready_deployment and project_exists)
        return {
            "verified": verified,
            "github_repo": github_repo_ref,
            "repo_url": repo_url,
            "vercel_project_exists": project_exists,
            "deploy_url": deploy_url,
            "deploy_state": deploy_state,
            "github_error": github_result.get("error", "") if not github_result.get("success") else "",
            "vercel_error": (
                vercel_project.get("error", "") or vercel_deployments.get("error", "")
            ) if not verified else "",
        }

    async def _persist_project_status(
        self,
        project_name: str,
        description: str,
        workdir: str,
        state_files: dict,
        verification: dict,
    ) -> None:
        pm_ctx = state_files.get("pm_context", "")
        await project_registry.upsert_project(
            name=project_name,
            description=description[:500],
            workspace_path=workdir,
            github_repo=verification.get("github_repo") or None,
            deploy_provider="vercel",
            deploy_url=verification.get("deploy_url") or None,
            status="active",
            metadata_json={"pm_context": pm_ctx[:8000]} if pm_ctx else None,
            mark_deployed=bool(verification.get("verified")),
        )

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
    {"path": "package.json", "description": "Dipendenze del progetto", "wave": 1},
    {"path": "src/lib/utils.ts", "description": "Utility condivise", "wave": 2},
    {"path": "src/components/Hero.tsx", "description": "Hero component full-screen", "wave": 3},
    {"path": "src/app/page.tsx", "description": "Home page con tutte le sezioni", "wave": 4}
  ],
  "notes": "note tecniche"
}

Assegna OBBLIGATORIAMENTE il campo "wave" (intero 1-4) a ogni file seguendo questa logica:
- Wave 1 (Configurazione): package.json, tsconfig.json, tailwind.config.*, next.config.*, .env.example, README.md, Dockerfile, .gitignore
- Wave 2 (Core condiviso): lib/, utils/, types/, styles/globals.css, hooks/, store/, context/
- Wave 3 (Componenti UI): components/ di ogni tipo (Header, Footer, Hero, Card, etc.)
- Wave 4 (Pagine e routes): app/, pages/, api/ — tutto ciò che assembla i componenti

Includi TUTTI i file necessari. Non omettere nulla. Sii preciso."""

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
            plan.setdefault("tech_stack", specs.get("tech_stack", "Next.js 15 + React 19 + Tailwind CSS 4 + TypeScript 5 + Node 22 LTS"))
            return plan
        except (json.JSONDecodeError, ValueError):
            log.warning(f"[webdev] _specs_to_file_plan non JSON: {plan_raw[:200]}")
            return {
                "project_name": specs.get("project_name", "progetto-web"),
                "tech_stack": specs.get("tech_stack", "Next.js 15 + React 19 + Tailwind CSS 4 + TypeScript 5 + Node 22 LTS"),
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

    def _files_to_xml_tasks(self, files: list[dict], workdir: str, wave: int) -> str:
        """Convert a list of file dicts into an XML task block for the builder LLM."""
        task_items = []
        for f in files:
            path = f.get("path", "")
            desc = f.get("description", "")
            task_items.append(
                f"  <task>\n"
                f"    <path>{workdir}/{path}</path>\n"
                f"    <action>Crea {path}: {desc}. "
                f"Codice completo e production-ready, zero placeholder o TODO.</action>\n"
                f"    <verify>File esiste, nessun placeholder, sintassi corretta, import validi</verify>\n"
                f"  </task>"
            )
        header = f'<tasks wave="{wave}" count="{len(files)}">'
        return header + "\n" + "\n".join(task_items) + "\n</tasks>"

    async def _build_by_waves(
        self,
        plan: dict,
        design_system_section: str,
        specs_context: str,
        user_message: str,
        workdir: str,
        task_id: int,
        state_files: dict,
    ) -> str:
        """Execute the build phase wave by wave.

        Each wave gets a fresh agent run (fresh context window). Waves are
        sequential (later waves depend on earlier ones). Within a wave, all
        files are tackled in a single agentic loop via XML task blocks.
        """
        # Group files by wave number
        waves: dict[int, list[dict]] = {}
        for f in plan.get("files", []):
            w = int(f.get("wave", 2))
            waves.setdefault(w, []).append(f)

        results: list[str] = []
        total_waves = len(waves)

        # Wave 0: write PROJECT.md / REQUIREMENTS.md / PM_CONTEXT.md to disk
        project_md = state_files.get("project_md", "")
        requirements_md = state_files.get("requirements_md", "")
        pm_context = state_files.get("pm_context", "")
        if project_md or requirements_md or pm_context:
            state_instructions = f"Crea la directory {workdir}/ se non esiste.\n"
            if project_md:
                state_instructions += (
                    f"\nScrivi il file {workdir}/PROJECT.md con esattamente questo contenuto:\n"
                    f"```\n{project_md}\n```\n"
                )
            if requirements_md:
                state_instructions += (
                    f"\nScrivi il file {workdir}/REQUIREMENTS.md con esattamente questo contenuto:\n"
                    f"```\n{requirements_md}\n```\n"
                )
            if pm_context:
                state_instructions += (
                    f"\nScrivi il file {workdir}/PM_CONTEXT.md con esattamente questo contenuto:\n"
                    f"```\n{pm_context}\n```\n"
                    "Questo file è il contesto permanente del Project Manager per questo progetto. "
                    "Non modificarne la struttura."
                )
            await self._run_phase(
                phase_name="Wave 0 — Stato progetto",
                system_prompt=(
                    "Sei un DevOps Engineer. Crea i file richiesti sul filesystem "
                    "esattamente come indicato, senza modifiche al contenuto."
                ),
                user_content=state_instructions,
                task_id=task_id,
                task_type=TaskType.TOOL_EXECUTION,
                progress=15,
            )

        for wave_num in sorted(waves.keys()):
            wave_files = waves[wave_num]
            wave_progress = 20 + int((wave_num - 1) / max(total_waves, 1) * 35)
            xml_tasks = self._files_to_xml_tasks(wave_files, workdir, wave_num)

            wave_context = (
                f"Richiesta originale: {user_message}\n\n"
                f"Stack: {plan.get('tech_stack', '')}\n"
                f"Directory di lavoro: {workdir}\n\n"
                f"── WAVE {wave_num}/{total_waves} ──\n"
                f"Implementa TUTTI i file elencati qui sotto. Ogni <task> è un file da creare:\n\n"
                f"{xml_tasks}\n\n"
                f"Data corrente: 2026-04-08\n"
                f"Prima di scrivere package.json o requirements, verifica le versioni correnti "
                f"via `npm view <pkg> version` o browser.\n"
                f"{_WEB_STACK_POLICY}\n"
                f"{specs_context}"
                f"{design_system_section}"
            )

            result = await self._run_phase(
                phase_name=f"Wave {wave_num}/{total_waves} — Build",
                system_prompt=_BUILDER_PROMPT,
                user_content=wave_context,
                task_id=task_id,
                task_type=TaskType.CODE_GENERATION,
                progress=wave_progress,
            )
            results.append(f"[Wave {wave_num}] {result[:300]}")

        return "\n\n".join(results)

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
        state_files: dict | None = None,
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
                plan = {"project_name": "progetto-web", "tech_stack": "Next.js 15 + React 19 + Tailwind CSS 4 + TypeScript 5 + Node 22 LTS", "files": []}

        project_name = plan.get("project_name", "progetto-web")
        tech_stack = plan.get("tech_stack", "")
        workdir = f"/srv/agent/workspaces/{project_name}"

        await notify(
            f"✅ <b>Piano pronto</b>\n"
            f"📦 Progetto: <code>{project_name}</code>\n"
            f"🛠️ Stack: {tech_stack}\n"
            f"📄 File pianificati: {len(plan.get('files', []))}"
        )

        # ── FASE 2: BUILDER (wave execution) ────────────────────────────────
        await notify("👨‍💻 <b>WebDev</b> — <b>FASE 2/4</b>: Sviluppo codice (wave execution)…")
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

        build_result = await self._build_by_waves(
            plan=plan,
            design_system_section=design_system_section,
            specs_context=specs_context,
            user_message=user_message,
            workdir=workdir,
            task_id=task_id,
            state_files=state_files or {},
        )

        # ── FASE 3: REVIEWER ────────────────────────────────────────────────
        await notify("🔍 <b>WebDev</b> — <b>FASE 3/4</b>: Code review e fix…")
        await task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS, progress=60)

        reviewer_context = (
            f"Progetto: {project_name} ({tech_stack})\n"
            f"Directory: {workdir}\n"
            f"File attesi: {json.dumps([f['path'] for f in plan.get('files', [])], ensure_ascii=False)}\n\n"
            f"Controlla esplicitamente package.json, lockfile e requirements.* contro versioni correnti. "
            f"Aggiorna dipendenze obsolete a release stabili recenti e poi restituisci un report.\n"
            f"{_WEB_STACK_POLICY}"
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

        verification = await self._verify_publish(project_name, task_id)
        await self._persist_project_status(
            project_name=project_name,
            description=plan.get("description", ""),
            workdir=workdir,
            state_files=state_files or {},
            verification=verification,
        )

        await task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS, progress=100)

        # ── REPORT FINALE ────────────────────────────────────────────────────
        task_cost = cost_tracker.get_task_cost(task_id)
        github_status = verification.get("repo_url") or f"NON verificato ({verification.get('github_error') or 'repo non trovato'})"
        deploy_status = verification.get("deploy_url") or (
            f"NON verificato ({verification.get('vercel_error') or 'deployment ready non trovata'})"
        )
        deploy_verification_label = "SI" if verification.get("verified") else "NO"
        return (
            f"✅ <b>Progetto completato: {project_name}</b>\n\n"
            f"🛠️ Stack: {tech_stack}\n\n"
            f"<b>Review:</b>\n{review_result[:800]}\n\n"
            f"<b>Deploy:</b>\n{deploy_result[:800]}\n\n"
            f"<b>Deploy verificato:</b> {deploy_verification_label}\n"
            f"<b>GitHub verificato:</b> {github_status}\n"
            f"<b>Vercel verificato:</b> {deploy_status}\n"
            f"<b>Stato deployment:</b> {verification.get('deploy_state', 'missing')}\n\n"
            f"💰 Costo totale task: ${task_cost:.3f}"
        )
