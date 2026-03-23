"""Image processing tool — download, resize, remove background, convert."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from config import config
from db.models import RiskLevel
from tools.base_tool import BaseTool
from utils.logging import setup_logging

log = setup_logging("tool_image")


class ImageTool(BaseTool):
    name = "image"
    description = "Download images, resize, remove background, convert formats."
    risk_level = RiskLevel.LOW

    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["download", "resize", "remove_bg", "convert", "info"],
                    "description": "Image operation to perform.",
                },
                "input_path": {"type": "string", "description": "Input image path."},
                "output_path": {"type": "string", "description": "Output image path."},
                "url": {"type": "string", "description": "URL to download image from."},
                "width": {"type": "integer", "description": "Target width for resize."},
                "height": {"type": "integer", "description": "Target height for resize."},
                "format": {"type": "string", "description": "Target format (png, jpg, webp)."},
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        action = kwargs["action"]
        try:
            if action == "download":
                url = kwargs["url"]
                ext = Path(url).suffix or ".jpg"
                output = Path(kwargs.get("output_path", str(config.media_dir / f"img_{hash(url)}{ext}")))
                output.parent.mkdir(parents=True, exist_ok=True)
                async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as c:
                    resp = await c.get(url)
                    resp.raise_for_status()
                    output.write_bytes(resp.content)
                return {"success": True, "path": str(output), "size": len(resp.content)}

            elif action == "resize":
                from PIL import Image
                img = Image.open(kwargs["input_path"])
                w = kwargs.get("width", img.width)
                h = kwargs.get("height", img.height)
                img = img.resize((w, h), Image.LANCZOS)
                out = kwargs.get("output_path", kwargs["input_path"])
                img.save(out)
                return {"success": True, "path": out, "size": f"{w}x{h}"}

            elif action == "remove_bg":
                from rembg import remove
                from PIL import Image
                import io
                img = Image.open(kwargs["input_path"])
                result = remove(img)
                out = kwargs.get("output_path", kwargs["input_path"].replace(".", "_nobg."))
                if not out.endswith(".png"):
                    out = out.rsplit(".", 1)[0] + ".png"
                result.save(out)
                return {"success": True, "path": out}

            elif action == "convert":
                from PIL import Image
                img = Image.open(kwargs["input_path"])
                fmt = kwargs.get("format", "png")
                out = kwargs.get("output_path", kwargs["input_path"].rsplit(".", 1)[0] + f".{fmt}")
                if fmt.lower() in ("jpg", "jpeg") and img.mode == "RGBA":
                    img = img.convert("RGB")
                img.save(out, format=fmt.upper())
                return {"success": True, "path": out}

            elif action == "info":
                from PIL import Image
                img = Image.open(kwargs["input_path"])
                return {
                    "success": True,
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                }

            else:
                return {"success": False, "error": f"Unknown action: {action}"}

        except Exception as e:
            return {"success": False, "error": str(e)}
