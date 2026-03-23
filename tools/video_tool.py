"""Video processing tool — FFmpeg wrapper for video operations."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from config import config
from db.models import RiskLevel
from tools.base_tool import BaseTool
from utils.logging import setup_logging

log = setup_logging("tool_video")


class VideoTool(BaseTool):
    name = "video"
    description = "Process videos using FFmpeg: convert, trim, extract frames, create GIFs, get info."
    risk_level = RiskLevel.LOW

    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["convert", "trim", "extract_frames", "create_gif", "info", "compress"],
                    "description": "Video operation to perform.",
                },
                "input_path": {"type": "string", "description": "Input video path."},
                "output_path": {"type": "string", "description": "Output path."},
                "format": {"type": "string", "description": "Target format (mp4, webm, avi)."},
                "start_time": {"type": "string", "description": "Start time for trim (HH:MM:SS)."},
                "end_time": {"type": "string", "description": "End time for trim (HH:MM:SS)."},
                "fps": {"type": "integer", "description": "Frames per second for extraction.", "default": 1},
            },
            "required": ["action", "input_path"],
        }

    async def _run_ffmpeg(self, args: list[str], timeout: int = 300) -> dict[str, Any]:
        cmd = ["ffmpeg", "-y"] + args
        log.info(f"FFmpeg: {' '.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode != 0:
            return {"success": False, "error": stderr.decode("utf-8", errors="replace")[:2000]}
        return {"success": True}

    async def execute(self, **kwargs) -> dict[str, Any]:
        action = kwargs["action"]
        input_path = kwargs["input_path"]

        try:
            if action == "convert":
                out = kwargs.get("output_path", input_path.rsplit(".", 1)[0] + f".{kwargs.get('format', 'mp4')}")
                result = await self._run_ffmpeg(["-i", input_path, out])
                return {**result, "path": out} if result["success"] else result

            elif action == "trim":
                out = kwargs.get("output_path", input_path.rsplit(".", 1)[0] + "_trimmed.mp4")
                args = ["-i", input_path]
                if kwargs.get("start_time"):
                    args.extend(["-ss", kwargs["start_time"]])
                if kwargs.get("end_time"):
                    args.extend(["-to", kwargs["end_time"]])
                args.extend(["-c", "copy", out])
                result = await self._run_ffmpeg(args)
                return {**result, "path": out} if result["success"] else result

            elif action == "extract_frames":
                out_dir = kwargs.get("output_path", str(config.media_dir / "frames"))
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                fps = kwargs.get("fps", 1)
                result = await self._run_ffmpeg([
                    "-i", input_path, "-vf", f"fps={fps}",
                    f"{out_dir}/frame_%04d.png"
                ])
                return {**result, "output_dir": out_dir} if result["success"] else result

            elif action == "create_gif":
                out = kwargs.get("output_path", input_path.rsplit(".", 1)[0] + ".gif")
                result = await self._run_ffmpeg([
                    "-i", input_path, "-vf",
                    "fps=10,scale=480:-1:flags=lanczos",
                    "-loop", "0", out
                ])
                return {**result, "path": out} if result["success"] else result

            elif action == "info":
                proc = await asyncio.create_subprocess_exec(
                    "ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_format", "-show_streams", input_path,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
                import json
                return {"success": True, "info": json.loads(stdout.decode())}

            elif action == "compress":
                out = kwargs.get("output_path", input_path.rsplit(".", 1)[0] + "_compressed.mp4")
                result = await self._run_ffmpeg([
                    "-i", input_path, "-vcodec", "libx264",
                    "-crf", "28", "-preset", "fast", out
                ])
                return {**result, "path": out} if result["success"] else result

            else:
                return {"success": False, "error": f"Unknown action: {action}"}

        except asyncio.TimeoutError:
            return {"success": False, "error": "FFmpeg timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
