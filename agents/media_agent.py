"""Media Agent — processes images and videos using Pillow, rembg, FFmpeg."""

from __future__ import annotations

from core.model_router import TaskType
from agents.base_agent import BaseAgent

MEDIA_SYSTEM_PROMPT = """\
Sei un agente specializzato nell'elaborazione multimediale. Puoi:

Immagini:
- Scaricare immagini da URL
- Ridimensionare, convertire formato, ottenere info
- Rimuovere lo sfondo (rembg)

Video:
- Convertire formati video
- Tagliare/trimmare video
- Estrarre frame da video
- Creare GIF animate
- Comprimere video
- Ottenere info sul video

Usa i tool image e video disponibili. Quando elabori un file,
restituisci il percorso del file risultante.
Rispondi sempre in italiano."""


class MediaAgent(BaseAgent):
    name = "media"
    description = "Processes images and videos (resize, convert, remove bg, trim, gif)."
    default_task_type = TaskType.MEDIA

    async def run(self, user_message, task_id, history=None, system_prompt="", model_override=None):
        combined_prompt = f"{MEDIA_SYSTEM_PROMPT}\n\n{system_prompt}" if system_prompt else MEDIA_SYSTEM_PROMPT
        return await super().run(
            user_message=user_message,
            task_id=task_id,
            history=history,
            system_prompt=combined_prompt,
            model_override=model_override,
        )
