"""WebDev Planning Q&A — interactive session to gather project specs before building.

Flow:
1. User sends a web_dev request
2. Orchestrator detects WEB_DEV intent → calls start_session()
3. A WebDevPlanningSession is created; orchestrator returns the first question
4. Telegram handler intercepts all subsequent messages while session is active
5. User answers each question (can also send photos/files as inspiration)
6. After the last answer → finalize() generates specs.json + design system
7. Orchestrator.handle_webdev_task() is called with the completed specs
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import config
from core.model_router import TaskType, get_model_for_task
from core.openrouter_client import openrouter
from utils.logging import setup_logging

log = setup_logging("webdev_planner")


def _extract_json_object(raw: str) -> dict | None:
    """Extract a JSON object from LLM output, stripping markdown code fences and tolerating noise."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


# ── Question bank ──────────────────────────────────────────────────────────

PLANNING_QUESTIONS: list[dict] = [
    {
        "id": "purpose",
        "question": (
            "🎯 <b>1. Scopo del sito</b>\n\n"
            "Cos'è questo sito e qual è l'obiettivo principale?\n"
            "<i>Es: portfolio fotografo, e-commerce scarpe, landing page startup, blog viaggi, "
            "sito vetrina per pizzeria…</i>"
        ),
        "key": "purpose",
    },
    {
        "id": "audience",
        "question": (
            "👥 <b>2. Target audience</b>\n\n"
            "A chi si rivolge il sito? Descrivi il visitatore ideale.\n"
            "<i>Es: professionisti 30-50 anni, studenti, appassionati di cucina, "
            "aziende B2B nel settore tech…</i>"
        ),
        "key": "audience",
    },
    {
        "id": "pages",
        "question": (
            "📄 <b>3. Pagine e sezioni</b>\n\n"
            "Quali pagine deve avere il sito? Elencale liberamente.\n"
            "<i>Es: Home, Chi siamo, Servizi, Portfolio, Blog, Contatti, FAQ, Prezzi…</i>"
        ),
        "key": "pages",
    },
    {
        "id": "content",
        "question": (
            "✍️ <b>4. Contenuto disponibile</b>\n\n"
            "Hai già materiale da usare?\n"
            "• Testi: sì / no\n"
            "• Logo: sì / no\n"
            "• Foto/immagini: sì / no\n\n"
            "Se sì, me li puoi mandare (anche qui su Telegram nelle prossime domande).\n"
            "Se no, genero tutto con AI."
        ),
        "key": "content",
    },
    {
        "id": "style",
        "question": (
            "🎨 <b>5. Stile visivo</b>\n\n"
            "Com'è la vibe del sito che vuoi?\n"
            "<i>Es: minimalista e pulito, colorato e giocoso, dark e premium, "
            "vintage e caldo, moderno e tech, lusso ed elegante…</i>\n\n"
            "Dai anche 2-3 <b>aggettivi</b> che descrivono come deve sentirsi il visitatore."
        ),
        "key": "style",
    },
    {
        "id": "colors",
        "question": (
            "🖌️ <b>6. Colori e brand</b>\n\n"
            "Hai già colori del brand? (hex, pantone, o descrizione)\n\n"
            "Se no, indica preferenze:\n"
            "<i>Es: blu navy + gold, bianco e nero + accento rosso, "
            "verde naturale + beige, viola + argento…</i>"
        ),
        "key": "colors",
    },
    {
        "id": "inspiration",
        "question": (
            "💡 <b>7. Ispirazione grafica</b>\n\n"
            "Mandami <b>screenshot o URL</b> di siti che ti piacciono "
            "(anche in settori diversi dal tuo).\n\n"
            "Cosa apprezzi di quei siti? (layout, animazioni, tipografia, foto, colori…)\n\n"
            "Se non hai riferimenti, descrivi il sito dei tuoi sogni in poche righe."
        ),
        "key": "inspiration",
        "accepts_media": True,
    },
    {
        "id": "features",
        "question": (
            "⚙️ <b>8. Funzionalità necessarie</b>\n\n"
            "Hai bisogno di funzionalità specifiche?\n"
            "<i>Form contatto, prenotazioni online, e-commerce, newsletter, "
            "login utenti, multilingua, blog CMS, mappa, chatbot, area riservata…</i>\n\n"
            "Se niente di speciale, rispondi <b>no</b>."
        ),
        "key": "features",
    },
    {
        "id": "extras",
        "question": (
            "✅ <b>9. Ultima domanda</b>\n\n"
            "C'è altro che devo sapere prima di iniziare?\n"
            "(deadline, budget, vincoli tecnici, cose che assolutamente non vuoi, "
            "competitors da non copiare, lingua del sito…)\n\n"
            "Se no, scrivi <b>ok, inizia</b>."
        ),
        "key": "extras",
    },
]

# ── LLM prompts ────────────────────────────────────────────────────────────

_SPECS_PROMPT = """\
Sei un web architect senior. Date le risposte di un cliente durante una sessione di discovery,
genera le specifiche complete del progetto web.

REGOLE FONDAMENTALI:
- Questo è un NUOVO progetto. NON riutilizzare nomi, descrizioni o dati di progetti precedenti.
- Il project_name e business_name DEVONO corrispondere a ciò che il cliente ha descritto nelle risposte.
- "description" è una stringa di testo semplice (2-3 frasi), NON un oggetto JSON annidato.

Regole stack/versioni:
- Per nuovi progetti preferisci uno stack moderno e coerente: Next.js 15.x + React 19.x + Tailwind CSS 4 + TypeScript 5.x + Node.js 22 LTS.
- Non proporre Node o framework vecchi senza un vincolo esplicito del cliente.
- Se proponi Tailwind 4, il progetto deve essere configurato in modo compatibile con Tailwind 4.
- Il campo `tech_stack` deve essere concreto e compatibile, non generico.

Restituisci SOLO un oggetto JSON valido con questa struttura esatta:
{
  "project_name": "nome-kebab-case",
  "business_name": "nome commerciale del progetto",
  "description": "cosa fa il sito in 2-3 frasi (stringa semplice, non JSON annidato)",
  "purpose": "obiettivo principale del sito",
  "target_audience": "chi sono i visitatori",
  "tech_stack": "es. Next.js 15 + React 19 + Tailwind CSS 4 + TypeScript 5 + Node 22 LTS + Framer Motion",
  "pages": [
    {
      "name": "Home",
      "slug": "/",
      "sections": ["hero", "features", "testimonials", "cta"],
      "description": "cosa mostra questa pagina"
    }
  ],
  "features": ["lista di funzionalità richieste"],
  "content_strategy": {
    "has_user_texts": false,
    "needs_ai_copy": true,
    "has_user_images": false,
    "needs_ai_images": true,
    "has_user_logo": false
  },
  "seo_keywords": ["keyword1", "keyword2", "keyword3"],
  "tone_of_voice": "professionale | amichevole | autorevole | giocoso | minimale | lussuoso",
  "copy_language": "it | en | bilingual",
  "deploy_target": "vercel",
  "notes": "note tecniche o vincoli importanti"
}"""

_DESIGN_SYSTEM_PROMPT = """\
Sei un senior UI/UX designer con esperienza in progetti award-winning (Awwwards, FWA).
Data una descrizione di progetto web, genera un design system completo, coerente e distintivo.

Ispirazione da siti top: Linear, Stripe, Vercel, Loom, Raycast, Apple, Notion, Framer.
Evita design generici e "corporate boring". Sii opinionato.

Restituisci SOLO un oggetto JSON valido:
{
  "color_palette": {
    "primary": "#1a1a2e",
    "secondary": "#16213e",
    "accent": "#e94560",
    "background": "#0f0f0f",
    "surface": "#1a1a1a",
    "text_primary": "#ffffff",
    "text_secondary": "#a0a0a0",
    "border": "#2a2a2a",
    "gradient_from": "#1a1a2e",
    "gradient_to": "#e94560"
  },
  "typography": {
    "heading_font": "Inter",
    "body_font": "Inter",
    "mono_font": "JetBrains Mono",
    "heading_weight": "800",
    "body_weight": "400",
    "base_size": "16px",
    "scale_ratio": "1.333",
    "line_height": "1.6"
  },
  "spacing": {
    "base_unit": "8px",
    "section_padding": "120px 0",
    "container_max_width": "1200px",
    "card_padding": "32px"
  },
  "style_tokens": {
    "border_radius_sm": "8px",
    "border_radius_md": "16px",
    "border_radius_lg": "24px",
    "box_shadow": "0 20px 60px rgba(0,0,0,0.3)",
    "transition_fast": "all 0.15s ease",
    "transition_normal": "all 0.3s cubic-bezier(0.4,0,0.2,1)",
    "blur_background": "blur(20px)",
    "gradient_text": "linear-gradient(135deg, #primary 0%, #accent 100%)"
  },
  "design_direction": "Descrizione della direzione stilistica in 2-3 frasi evocative",
  "layout_pattern": "hero-fullscreen | hero-split | centered | grid | editorial",
  "animation_style": "none | subtle | moderate | expressive | cinematic",
  "hero_style": "full-bleed-video | animated-gradient | glassmorphism | minimal-text | bold-typography",
  "component_style": "glassmorphism | neomorphism | flat | outlined | filled | brutalist",
  "inspiration_sources": ["sito1.com", "sito2.com"]
}

Adatta ogni scelta al settore, pubblico e stile desiderato. Non usare i valori di esempio, crea quelli giusti."""

_IMAGE_ANALYSIS_PROMPT = (
    "Analizza questo screenshot/design come un UI designer senior. "
    "Estrai in JSON: "
    "{'colors': ['#hex1', '#hex2'], 'style': 'descrizione stile', "
    "'layout': 'layout type', 'typography_feel': 'serif/sans/display', "
    "'mood': 'mood', 'distinctive': 'elemento più distintivo'}. "
    "Solo JSON, nessun testo fuori."
)


# ── Session class ──────────────────────────────────────────────────────────

@dataclass
class WebDevPlanningSession:
    """State machine that drives the Q&A dialogue for a web project."""

    user_id: int
    initial_message: str
    answers: dict[str, Any] = field(default_factory=dict)
    media_files: list[str] = field(default_factory=list)
    image_insights: list[str] = field(default_factory=list)
    current_question_idx: int = 0
    completed: bool = False
    specs: dict | None = None
    design_system: dict | None = None
    project_md: str = ""
    requirements_md: str = ""
    pm_context: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def current_question(self) -> dict | None:
        if self.current_question_idx < len(PLANNING_QUESTIONS):
            return PLANNING_QUESTIONS[self.current_question_idx]
        return None

    def accepts_media_now(self) -> bool:
        q = self.current_question
        return q is not None and q.get("accepts_media", False)

    def progress_bar(self) -> str:
        total = len(PLANNING_QUESTIONS)
        done = min(self.current_question_idx, total)
        filled = "█" * done + "░" * (total - done)
        return f"<code>[{filled}]</code> {done}/{total}"

    def record_answer(self, answer: str) -> None:
        q = self.current_question
        if q:
            self.answers[q["key"]] = answer
            self.current_question_idx += 1
            _schedule_redis_save(self.user_id, self)

    def add_media(self, file_path: str) -> None:
        self.media_files.append(file_path)

    async def add_media_with_analysis(self, file_path: str) -> str:
        """Store media and analyze it for design insights. Returns the insight text."""
        self.media_files.append(file_path)
        insight = await analyze_inspiration_image(file_path)
        if insight:
            self.image_insights.append(insight)
        return insight

    async def finalize(self) -> dict:
        """Generate specs + design system from the collected Q&A answers."""
        # Merge image insights into the inspiration answer
        if self.image_insights:
            existing = self.answers.get("inspiration", "")
            self.answers["inspiration"] = (
                existing + "\n\nInsights estratti dalle immagini:\n"
                + "\n".join(self.image_insights)
            ).strip()

        self.specs = await _generate_specs(self.initial_message, self.answers)
        self.design_system = await _generate_design_system(self.specs, self.answers)
        self.project_md, self.requirements_md = await _generate_state_files(self.specs, self.answers)
        self.pm_context = await _generate_pm_context(
            self.specs, self.design_system, self.answers, self.project_md, self.requirements_md
        )
        self.completed = True
        log.info(f"[webdev_planner] Session finalized for user {self.user_id}: {self.specs.get('project_name')}")
        return {
            "specs": self.specs,
            "design_system": self.design_system,
            "media": self.media_files,
            "project_md": self.project_md,
            "requirements_md": self.requirements_md,
            "pm_context": self.pm_context,
        }

    def format_specs_summary(self) -> str:
        """Human-readable summary of the generated specs for Telegram."""
        if not self.specs or not self.design_system:
            return "Specifiche non ancora generate."

        s = self.specs
        ds = self.design_system
        cp = ds.get("color_palette", {})
        ty = ds.get("typography", {})

        pages_list = ", ".join(p["name"] for p in s.get("pages", []))
        features_list = ", ".join(s.get("features", [])[:5]) or "—"

        return (
            f"✅ <b>Specifiche generate!</b>\n\n"
            f"📦 <b>Progetto:</b> {s.get('business_name', s.get('project_name'))}\n"
            f"📝 {s.get('description', '')}\n\n"
            f"📄 <b>Pagine:</b> {pages_list}\n"
            f"⚙️ <b>Features:</b> {features_list}\n"
            f"🛠️ <b>Stack:</b> {s.get('tech_stack', '')}\n\n"
            f"🎨 <b>Design System:</b>\n"
            f"  • Direzione: {ds.get('design_direction', '')[:120]}\n"
            f"  • Layout: {ds.get('layout_pattern', '')} | Animazioni: {ds.get('animation_style', '')}\n"
            f"  • Colori: <code>{cp.get('primary', '')} / {cp.get('accent', '')}</code>\n"
            f"  • Font: {ty.get('heading_font', '')} / {ty.get('body_font', '')}\n\n"
            f"🚀 Avvio pipeline di sviluppo…"
        )


# ── Global session registry ────────────────────────────────────────────────

_active_sessions: dict[int, WebDevPlanningSession] = {}

_REDIS_PREFIX = "webdev_session:"
_SESSION_TTL = 86400  # 24h


def get_session(user_id: int) -> WebDevPlanningSession | None:
    return _active_sessions.get(user_id)


def start_session(user_id: int, initial_message: str) -> WebDevPlanningSession:
    session = WebDevPlanningSession(user_id=user_id, initial_message=initial_message)
    _active_sessions[user_id] = session
    _schedule_redis_save(user_id, session)
    log.info(f"[webdev_planner] Started session for user {user_id}")
    return session


def end_session(user_id: int) -> None:
    _active_sessions.pop(user_id, None)
    _schedule_redis_delete(user_id)
    log.info(f"[webdev_planner] Ended session for user {user_id}")


def abort_session(user_id: int) -> bool:
    """Abort an in-progress session. Returns True if one was active."""
    if user_id in _active_sessions:
        end_session(user_id)
        return True
    return False


# ── Redis persistence helpers ──────────────────────────────────────────────

async def save_session_to_redis(user_id: int, session: WebDevPlanningSession) -> None:
    """Persist Q&A session state to Redis."""
    import redis.asyncio as aioredis
    try:
        r = aioredis.from_url(config.redis_url, decode_responses=True)
        key = f"{_REDIS_PREFIX}{user_id}"
        data = {
            "user_id": str(user_id),
            "initial_message": session.initial_message,
            "answers": json.dumps(session.answers, ensure_ascii=False),
            "media_files": json.dumps(session.media_files),
            "image_insights": json.dumps(session.image_insights, ensure_ascii=False),
            "current_question_idx": str(session.current_question_idx),
        }
        await r.hset(key, mapping=data)
        await r.expire(key, _SESSION_TTL)
        await r.aclose()
    except Exception as e:
        log.warning(f"[webdev_planner] Failed to save session to Redis: {e}")


async def restore_sessions_from_redis() -> int:
    """Restore Q&A planning sessions from Redis at startup. Returns count restored."""
    import redis.asyncio as aioredis
    try:
        r = aioredis.from_url(config.redis_url, decode_responses=True)
        keys = []
        async for key in r.scan_iter(match=f"{_REDIS_PREFIX}*"):
            keys.append(key)

        restored = 0
        for key in keys:
            try:
                data = await r.hgetall(key)
                if not data:
                    continue
                uid = int(data["user_id"])
                session = WebDevPlanningSession(
                    user_id=uid,
                    initial_message=data["initial_message"],
                )
                session.answers = json.loads(data.get("answers", "{}"))
                session.media_files = json.loads(data.get("media_files", "[]"))
                session.image_insights = json.loads(data.get("image_insights", "[]"))
                session.current_question_idx = int(data.get("current_question_idx", "0"))
                _active_sessions[uid] = session
                restored += 1
            except Exception as e:
                log.warning(f"[webdev_planner] Failed to restore session from {key}: {e}")
                await r.delete(key)

        await r.aclose()
        if restored:
            log.info(f"[webdev_planner] Restored {restored} Q&A sessions from Redis")
        return restored
    except Exception as e:
        log.warning(f"[webdev_planner] Failed to restore sessions from Redis: {e}")
        return 0


def _schedule_redis_save(user_id: int, session: WebDevPlanningSession) -> None:
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(save_session_to_redis(user_id, session))
    except RuntimeError:
        pass


def _schedule_redis_delete(user_id: int) -> None:
    import asyncio

    async def _delete():
        import redis.asyncio as aioredis
        try:
            r = aioredis.from_url(config.redis_url, decode_responses=True)
            await r.delete(f"{_REDIS_PREFIX}{user_id}")
            await r.aclose()
        except Exception:
            pass

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_delete())
    except RuntimeError:
        pass


# ── LLM helpers ────────────────────────────────────────────────────────────

async def _generate_specs(initial_message: str, answers: dict) -> dict:
    """Synthesize Q&A answers into a structured project specs dict."""
    model = get_model_for_task(TaskType.COMPLEX_REASONING)

    qa_text = f"Richiesta iniziale: {initial_message}\n\n"
    for q in PLANNING_QUESTIONS:
        answer = answers.get(q["key"], "Non specificato")
        clean_q = re.sub(r"<[^>]+>", "", q["question"])[:120]
        qa_text += f"Q: {clean_q}\nR: {answer}\n\n"

    response = await openrouter.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SPECS_PROMPT},
            {"role": "user", "content": qa_text},
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    raw = (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    raw = raw.strip()

    parsed = _extract_json_object(raw)
    if parsed is not None:
        # Validate description is a plain string, not a nested JSON structure
        desc = parsed.get("description", "")
        if isinstance(desc, dict):
            parsed["description"] = desc.get("description", str(desc))[:500]
        elif isinstance(desc, str) and desc.strip().startswith("{"):
            try:
                nested = json.loads(desc)
                parsed["description"] = nested.get("description", str(nested))[:500]
            except (json.JSONDecodeError, ValueError):
                pass
        return parsed
    log.warning(f"[webdev_planner] Specs non-JSON: {raw[:200]}")
    return {
        "project_name": "progetto-web",
            "description": initial_message,
            "tech_stack": "Next.js 15 + React 19 + Tailwind CSS 4 + TypeScript 5 + Node 22 LTS",
            "pages": [{"name": "Home", "slug": "/", "sections": ["hero", "cta"]}],
        }


_STATE_FILES_PROMPT = """\
Dato un documento di specifiche di progetto web, genera due file markdown di stato da salvare nel repo.

Restituisci SOLO un oggetto JSON valido:
{
  "project_md": "# NomeProgetto\\n\\n## Visione\\n...\\n\\n## Stack\\n...\\n\\n## Target Audience\\n...",
  "requirements_md": "# Requirements\\n\\n## v1 — Must Have\\n- ...\\n\\n## v2 — Nice to Have\\n- ...\\n\\n## Out of scope\\n- ..."
}

PROJECT.md: visione del progetto, stack tecnico, audience, tone of voice, URL deploy previsto (max 250 parole).
REQUIREMENTS.md: divide i requisiti in v1 (funzionalità che DEVONO esserci al lancio), v2 (future), out-of-scope."""


_PM_CONTEXT_PROMPT = """\
Sei un esperto di project management software. Ricevi i dati completi di un progetto web \
appena pianificato e devi generare il documento `PM_CONTEXT.md`.

Questo documento sarà il "cervello permanente" del Project Manager AI dedicato a questo progetto.
Ogni volta che il cliente chiederà una modifica futura, l'agente PM carica questo file e \
risponde/delega con piena conoscenza del contesto.

Il documento DEVE permettere all'agente di:
1. Rispondere a qualsiasi domanda sul progetto senza cercare altrove
2. Valutare se una richiesta è in scope v1, v2 o fuori scope
3. Prendere decisioni tecniche coerenti con l'architettura e il design system scelti
4. Ricordare le preferenze, il tono e le aspettative del cliente
5. Sapere con certezza dove sono i file e come è strutturato il codice

Genera il documento Markdown COMPLETO e STRUTTURATO. Nessun codeblock wrapper, solo il markdown grezzo."""

_PM_CONTEXT_USER_TEMPLATE = """\
## Dati del progetto

### Specs
{specs_json}

### Design System
{design_system_json}

### Risposte Q&A del cliente
{qa_json}

### PROJECT.md già generato
{project_md}

### REQUIREMENTS.md già generato
{requirements_md}
"""


async def _generate_pm_context(
    specs: dict,
    design_system: dict,
    answers: dict,
    project_md: str,
    requirements_md: str,
) -> str:
    """Generate the full PM_CONTEXT.md content from project data."""
    model = get_model_for_task(TaskType.COMPLEX_REASONING)

    user_content = _PM_CONTEXT_USER_TEMPLATE.format(
        specs_json=json.dumps(specs, indent=2, ensure_ascii=False)[:1500],
        design_system_json=json.dumps(design_system, indent=2, ensure_ascii=False)[:800],
        qa_json=json.dumps(answers, indent=2, ensure_ascii=False)[:800],
        project_md=project_md[:500],
        requirements_md=requirements_md[:500],
    )

    response = await openrouter.chat(
        model=model,
        messages=[
            {"role": "system", "content": _PM_CONTEXT_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=2000,
    )
    raw = (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    raw = raw.strip()
    if not raw:
        # Minimal fallback
        cp = design_system.get("color_palette", {})
        ty = design_system.get("typography", {})
        return (
            f"# PM Context: {specs.get('business_name', specs.get('project_name'))}\n\n"
            f"## Visione\n{specs.get('description', '')}\n\n"
            f"## Stack\n{specs.get('tech_stack', '')}\n\n"
            f"## Target\n{specs.get('target_audience', '')}\n\n"
            f"## Design System\n"
            f"- Primary: {cp.get('primary', '')} | Accent: {cp.get('accent', '')}\n"
            f"- Font: {ty.get('heading_font', '')} / {ty.get('body_font', '')}\n"
            f"- Stile: {design_system.get('component_style', '')}\n\n"
            f"## Requisiti v1\n{requirements_md[:600]}\n\n"
            "## Storico modifiche\n"
        )
    # Ensure there's a changelog section
    if "## Storico modifiche" not in raw:
        raw += "\n\n## Storico modifiche\n"
    return raw


async def _generate_state_files(specs: dict, answers: dict) -> tuple[str, str]:
    """Generate PROJECT.md and REQUIREMENTS.md content strings from project specs."""
    model = get_model_for_task(TaskType.COMPLEX_REASONING)
    context = json.dumps({"specs": specs, "q_and_a": answers}, indent=2, ensure_ascii=False)[:3000]
    response = await openrouter.chat(
        model=model,
        messages=[
            {"role": "system", "content": _STATE_FILES_PROMPT},
            {"role": "user", "content": context},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    raw = (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    raw = raw.strip()
    parsed = _extract_json_object(raw)
    if parsed is not None:
        return parsed.get("project_md") or "", parsed.get("requirements_md") or ""
    log.warning(f"[webdev_planner] State files non-JSON: {raw[:200]}")
    project_md = (
        f"# {specs.get('business_name', specs.get('project_name', 'Progetto'))}\n\n"
        f"{specs.get('description', '')}\n\n"
        f"**Stack:** {specs.get('tech_stack', '')}\n"
        f"**Target:** {specs.get('target_audience', '')}\n"
        f"**Tone:** {specs.get('tone_of_voice', '')}"
    )
    requirements_md = (
        "# Requirements\n\n## v1 — Must Have\n"
        + "\n".join(f"- {f}" for f in specs.get("features", []))
    )
    return project_md, requirements_md


async def _generate_design_system(specs: dict, answers: dict) -> dict:
    """Generate a complete design system based on project specs and user preferences."""
    model = get_model_for_task(TaskType.COMPLEX_REASONING)

    context = (
        f"Progetto: {specs.get('description', '')}\n"
        f"Business: {specs.get('business_name', '')}\n"
        f"Settore/Purpose: {specs.get('purpose', '')}\n"
        f"Target: {specs.get('target_audience', '')}\n"
        f"Tone of voice: {specs.get('tone_of_voice', '')}\n\n"
        f"Stile desiderato: {answers.get('style', '')}\n"
        f"Colori indicati: {answers.get('colors', '')}\n"
        f"Ispirazione: {answers.get('inspiration', '')}"
    )

    response = await openrouter.chat(
        model=model,
        messages=[
            {"role": "system", "content": _DESIGN_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        temperature=0.4,
        max_tokens=1200,
    )
    raw = (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    raw = raw.strip()

    parsed = _extract_json_object(raw)
    if parsed is not None:
        return parsed
    log.warning(f"[webdev_planner] Design system non-JSON: {raw[:200]}")
    return {}


async def analyze_inspiration_image(image_path: str) -> str:
    """Use vision LLM to extract design insights from an inspirational image.

    Returns a JSON string describing colors, style, layout, mood.
    Returns empty string on failure.
    """
    try:
        img_data = Path(image_path).read_bytes()
        b64 = base64.b64encode(img_data).decode()
        ext = Path(image_path).suffix.lower().lstrip(".")
        mime = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp",
        }.get(ext, "image/jpeg")

        # Use the most capable model available for vision
        model = config.model_expensive

        response = await openrouter.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _IMAGE_ANALYSIS_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=400,
        )
        return (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    except Exception as exc:
        log.warning(f"[webdev_planner] Image analysis failed for {image_path}: {exc}")
        return ""


def build_design_system_prompt_section(design_system: dict) -> str:
    """Convert a design system dict into a rich builder prompt section."""
    if not design_system:
        return ""

    cp = design_system.get("color_palette", {})
    ty = design_system.get("typography", {})
    sp = design_system.get("spacing", {})
    st = design_system.get("style_tokens", {})

    css_vars = []
    for k, v in cp.items():
        css_vars.append(f"  --color-{k.replace('_', '-')}: {v};")

    return f"""
## Design System (obbligatorio rispettare)

**Direzione stilistica:** {design_system.get('design_direction', '')}
**Layout pattern:** {design_system.get('layout_pattern', 'hero-fullscreen')}
**Stile componenti:** {design_system.get('component_style', 'flat')}
**Animazioni:** {design_system.get('animation_style', 'subtle')}
**Hero style:** {design_system.get('hero_style', 'animated-gradient')}

**Palette colori:**
- Primary: {cp.get('primary', '')}
- Secondary: {cp.get('secondary', '')}
- Accent: {cp.get('accent', '')}
- Background: {cp.get('background', '')}
- Surface: {cp.get('surface', '')}
- Text primary: {cp.get('text_primary', '')}
- Text secondary: {cp.get('text_secondary', '')}
- Gradient: da {cp.get('gradient_from', cp.get('primary', ''))} a {cp.get('gradient_to', cp.get('accent', ''))}

**Tipografia:**
- Heading: {ty.get('heading_font', 'Inter')} weight {ty.get('heading_weight', '700')}
- Body: {ty.get('body_font', 'Inter')} weight {ty.get('body_weight', '400')}
- Base size: {ty.get('base_size', '16px')} | Scale: {ty.get('scale_ratio', '1.25')}
- Line height: {ty.get('line_height', '1.6')}

**Spacing:**
- Unit: {sp.get('base_unit', '8px')}
- Section padding: {sp.get('section_padding', '80px 0')}
- Container max-width: {sp.get('container_max_width', '1200px')}

**Style tokens:**
- Border radius: {st.get('border_radius_md', '12px')}
- Box shadow: {st.get('box_shadow', '0 4px 24px rgba(0,0,0,0.1)')}
- Transition: {st.get('transition_normal', 'all 0.3s ease')}
- Backdrop: {st.get('blur_background', 'blur(10px)')}

**CSS variables da usare:**
```css
:root {{
{chr(10).join(css_vars)}
}}
```

**Regole di stile:**
- Implementa esattamente questi colori, font e token — NON inventare altri
- Usa le Google Fonts indicate (importa dal link)
- Le animazioni devono essere smooth, non fastidiose
- Il sito deve sembrare fatto da un'agenzia top, non un template WordPress
- Responsive mobile-first OBBLIGATORIO
- Aggiungi micro-interazioni: hover effects, scroll animations (intersection observer)
"""
