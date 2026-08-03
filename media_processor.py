from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv()


class MediaProcessor:
    """Best-effort media enrichment with graceful fallback for bad files/API issues."""

    def __init__(self, dataset_dir: str | Path = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.api_key = os.getenv("GROQ_API_KEY", "") if os.getenv("USE_GROQ", "0") == "1" else ""

    def enrich(self, message: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(message)
        media_type = str(message.get("media_type", "")).strip().lower()
        rel_path = context.get("voice_path") if media_type == "voice" else context.get("image_path")
        media_path = self.dataset_dir / str(rel_path)
        try:
            if not media_type or not rel_path or not media_path.exists() or media_path.stat().st_size == 0:
                enriched["media_text"] = ""
            elif media_type == "voice":
                enriched["media_text"] = self._transcribe_voice(media_path)
            elif media_type == "image":
                enriched["media_text"] = self._describe_image(media_path)
            else:
                enriched["media_text"] = ""
        except Exception as exc:
            enriched["media_error"] = f"{type(exc).__name__}: {exc}"
            enriched["media_text"] = ""
        return enriched

    def _transcribe_voice(self, media_path: Path) -> str:
        if not self.api_key:
            return ""
        try:
            from groq import Groq
        except ImportError:
            return ""
        client = Groq(api_key=self.api_key)
        with media_path.open("rb") as audio:
            result = client.audio.transcriptions.create(
                file=(media_path.name, audio.read()),
                model="whisper-large-v3",
                response_format="text",
            )
        return str(result).strip()

    def _describe_image(self, media_path: Path) -> str:
        if not self.api_key:
            return ""
        try:
            from groq import Groq
        except ImportError:
            return ""
        mime = "image/jpeg" if media_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        encoded = base64.b64encode(media_path.read_bytes()).decode("ascii")
        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract concise notification-relevant text and visual clues."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ],
                }
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
