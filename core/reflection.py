"""Self-improvement engine — reflect on completed tasks and evolve agent behavior.

After every task completion the engine:
1. Asks a cheap LLM to critique the execution
2. Stores the reflection in memory (category="self_improvement")
3. Every SYNTHESIS_THRESHOLD reflections, synthesizes guidelines per task_type
4. Guidelines are injected into future system prompts via get_improvement_context()
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from config import config
from core.memory import memory
from core.model_router import TaskType, get_model_for_task
from core.openrouter_client import openrouter
from utils.logging import setup_logging

log = setup_logging("reflection")

# ── Prompts ────────────────────────────────────────────────────────────────

_REFLECTION_PROMPT = """\
Sei un meta-analista di agenti AI. Il tuo compito è analizzare l'esecuzione di un task e identificare
miglioramenti concreti e attuabili.

Analizza il task fornito e restituisci SOLO un oggetto JSON valido (nessun testo fuori dal JSON):
{
  "success": true/false,
  "quality_score": 7,
  "what_worked": "cosa ha funzionato bene (max 80 parole)",
  "what_failed": "cosa è andato storto o poteva essere migliore (max 80 parole)",
  "root_cause": "causa principale di eventuali problemi (max 40 parole)",
  "improvement_action": "azione concreta da fare diversamente la prossima volta (max 60 parole)",
  "prompt_hint": "frase breve da aggiungere al system prompt per migliorare questo task_type (max 50 parole)"
}

Sii critico, preciso e conciso. quality_score va da 1 (pessimo) a 10 (eccellente)."""

_SYNTHESIS_PROMPT = """\
Sei un ingegnere AI specializzato nel miglioramento di sistemi agentici.
Ti vengono fornite riflessioni raccolte dopo vari task dello stesso tipo.

Sintetizza le riflessioni in linee guida strategiche aggiornate.
Restituisci SOLO un oggetto JSON valido:
{
  "task_type": "tipo",
  "guidelines": [
    "linea guida concreta 1",
    "linea guida concreta 2",
    "linea guida concreta 3"
  ],
  "anti_patterns": [
    "cosa evitare assolutamente 1",
    "cosa evitare assolutamente 2"
  ],
  "focus_areas": "le 2 aree più critiche su cui concentrarsi in futuro"
}

Massimo 5 guidelines e 3 anti_patterns. Sii conciso e direttamente applicabile."""


# ── Engine ─────────────────────────────────────────────────────────────────

class ReflectionEngine:
    """Analyzes completed tasks and evolves agent behavior over time."""

    SYNTHESIS_THRESHOLD = 8   # Synthesize after N cumulative reflections per task_type
    MEMORY_CATEGORY = "self_improvement"
    GUIDELINES_CATEGORY = "improvement_guidelines"

    # ── Public API ─────────────────────────────────────────────────────────

    async def analyze_task(
        self,
        user_id: int,
        task_id: int,
        task_type: str,
        user_message: str,
        outcome: str,
        success: bool,
        cost: float,
        duration_seconds: float,
    ) -> None:
        """Analyze a completed task and store a structured reflection.

        Intentionally fire-and-forget safe: all errors are swallowed so a
        reflection failure never breaks the main agent flow.
        """
        try:
            await self._do_analyze(
                user_id, task_id, task_type, user_message,
                outcome, success, cost, duration_seconds,
            )
        except Exception as exc:
            log.warning(f"[reflection] analyze_task silently failed: {exc}")

    async def get_improvement_context(self, user_id: int, task_type: str) -> str:
        """Return current improvement guidelines as a formatted string for prompt injection.

        Returns an empty string when no guidelines exist yet.
        """
        try:
            raw = await memory.recall(user_id, f"guidelines_{task_type}")
            if not raw:
                return ""
            guidelines: dict = json.loads(raw)
            lines = [f"\n## Auto-learned guidelines for {task_type}:"]
            for g in guidelines.get("guidelines", []):
                lines.append(f"  ✅ {g}")
            for a in guidelines.get("anti_patterns", []):
                lines.append(f"  ❌ Avoid: {a}")
            focus = guidelines.get("focus_areas", "")
            if focus:
                lines.append(f"  🎯 Focus: {focus}")
            return "\n".join(lines)
        except Exception:
            return ""

    async def get_all_guidelines_summary(self, user_id: int) -> str:
        """Return a summary of all existing guidelines (for /status or debug)."""
        all_mem = await memory.recall_by_category(user_id, self.GUIDELINES_CATEGORY)
        if not all_mem:
            return "Nessuna linea guida auto-appresa ancora."
        lines = ["📚 <b>Linee guida auto-apprese:</b>"]
        for m in all_mem:
            try:
                g = json.loads(m["value"])
                tt = g.get("task_type", m["key"])
                count = len(g.get("guidelines", []))
                lines.append(f"• <b>{tt}</b>: {count} guidelines")
            except Exception:
                pass
        return "\n".join(lines)

    # ── Internal ────────────────────────────────────────────────────────────

    async def _do_analyze(
        self,
        user_id: int,
        task_id: int,
        task_type: str,
        user_message: str,
        outcome: str,
        success: bool,
        cost: float,
        duration_seconds: float,
    ) -> None:
        model = get_model_for_task(TaskType.SIMPLE_CHAT)  # cheapest model

        context = (
            f"Task type: {task_type}\n"
            f"Richiesta: {user_message[:400]}\n"
            f"Esito: {'✅ Successo' if success else '❌ Fallimento'}\n"
            f"Output agente (troncato): {outcome[:800]}\n"
            f"Costo: ${cost:.4f}  |  Durata: {duration_seconds:.0f}s"
        )

        response = await openrouter.chat(
            model=model,
            messages=[
                {"role": "system", "content": _REFLECTION_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        raw = ((response.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()

        try:
            js = raw[raw.find("{"):raw.rfind("}") + 1]
            reflection: dict = json.loads(js)
        except (json.JSONDecodeError, ValueError):
            log.warning(f"[reflection] Non-JSON response for task #{task_id}: {raw[:200]}")
            return

        score = reflection.get("quality_score", "?")
        log.info(f"[reflection] Task #{task_id} ({task_type}) — quality {score}/10")

        # Persist reflection
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        await memory.remember(
            user_id,
            key=f"reflection_{task_type}_{ts}",
            value=json.dumps(reflection, ensure_ascii=False),
            category=self.MEMORY_CATEGORY,
        )

        # Bump counter and maybe synthesize
        count_key = f"reflection_count_{task_type}"
        count = int((await memory.recall(user_id, count_key)) or "0") + 1
        await memory.remember(user_id, count_key, str(count), category=self.MEMORY_CATEGORY)

        if count % self.SYNTHESIS_THRESHOLD == 0:
            log.info(f"[reflection] Threshold reached ({count}) — synthesizing for {task_type}")
            await self._synthesize(user_id, task_type)

    async def _synthesize(self, user_id: int, task_type: str) -> None:
        """Aggregate recent reflections into strategic guidelines."""
        all_mem = await memory.recall_by_category(user_id, self.MEMORY_CATEGORY)
        reflections = [
            m for m in all_mem
            if m["key"].startswith(f"reflection_{task_type}_")
        ]
        if len(reflections) < 3:
            return

        recent = reflections[-15:]
        combined = "\n\n".join(
            f"Riflessione {i + 1}: {r['value']}"
            for i, r in enumerate(recent)
        )

        model = get_model_for_task(TaskType.SIMPLE_CHAT)
        response = await openrouter.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYNTHESIS_PROMPT},
                {
                    "role": "user",
                    "content": f"Task type: {task_type}\n\nRiflessioni:\n{combined[:4000]}",
                },
            ],
            temperature=0.1,
            max_tokens=512,
        )
        raw = ((response.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()

        try:
            js = raw[raw.find("{"):raw.rfind("}") + 1]
            guidelines: dict = json.loads(js)
        except (json.JSONDecodeError, ValueError):
            log.warning(f"[reflection] Synthesis produced non-JSON for {task_type}: {raw[:200]}")
            return

        await memory.remember(
            user_id,
            key=f"guidelines_{task_type}",
            value=json.dumps(guidelines, ensure_ascii=False),
            category=self.GUIDELINES_CATEGORY,
        )
        log.info(f"[reflection] Guidelines updated for task_type={task_type}")


# Singleton
reflection_engine = ReflectionEngine()
