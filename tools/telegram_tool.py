"""Telegram Tool — send files, photos, and videos to the owner via Telegram API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from db.models import RiskLevel
from tools.base_tool import BaseTool
from tg.notifications import send_file, send_photo, send_video, notify


class TelegramTool(BaseTool):
    name = "telegram"
    description = (
        "Invia file, foto, video o messaggi a Santino tramite Telegram. "
        "Usa action='send_file' per documenti, 'send_photo' per immagini, "
        "'send_video' per video, 'send_message' per messaggi di testo."
    )
    risk_level = RiskLevel.LOW

    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["send_file", "send_photo", "send_video", "send_message"],
                    "description": "Tipo di invio: send_file, send_photo, send_video, send_message",
                },
                "path": {
                    "type": "string",
                    "description": "Percorso assoluto del file da inviare (per send_file/photo/video)",
                },
                "caption": {
                    "type": "string",
                    "description": "Didascalia opzionale per file/foto/video",
                },
                "text": {
                    "type": "string",
                    "description": "Testo del messaggio (solo per send_message)",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        action = kwargs.get("action", "")
        path = kwargs.get("path", "")
        caption = kwargs.get("caption", "")
        text = kwargs.get("text", "")

        if action == "send_message":
            if not text:
                return {"success": False, "error": "Parametro 'text' richiesto per send_message"}
            await notify(text)
            return {"success": True, "message": "Messaggio inviato"}

        if action in ("send_file", "send_photo", "send_video"):
            if not path:
                return {"success": False, "error": "Parametro 'path' richiesto"}
            p = Path(path)
            if not p.exists():
                return {"success": False, "error": f"File non trovato: {path}"}

            if action == "send_file":
                await send_file(str(p), caption=caption)
            elif action == "send_photo":
                await send_photo(str(p), caption=caption)
            elif action == "send_video":
                await send_video(str(p), caption=caption)

            return {"success": True, "message": f"{action} completato: {p.name}"}

        return {"success": False, "error": f"Azione non supportata: {action}"}
