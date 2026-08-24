"""
christman_ocr_shared.py
The Christman AI Project — Luma Cognify AI
Shared OCR Engine for Brockston, AlphaVox, and Seraphenia
"""

from __future__ import annotations

import asyncio
import json
import hashlib
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np
import websockets

try:
    from christman_voice_sdk.audio.config import get_config
    from christman_voice_sdk.utils.logger import get_logger
except ImportError:
    from audio.config import get_config
    from utils.logger import get_logger

logger = get_logger(__name__)

# ── Config from SDK ─────────────────────────────────────────────────────────
config = get_config()
DEREK_WS_URI = config.get("derek.ws_uri", "ws://localhost:8000/ws/derek")
CONFIDENCE_THRESHOLD = 0.50
SCREEN_POLL_INTERVAL = 2.5
PDF_DPI = 300


# Lazy imports for heavy deps
def _import_pil():
    from PIL import ImageGrab, Image
    return ImageGrab, Image


def _import_paddleocr():
    from paddleocr import PaddleOCR
    return PaddleOCR


def _import_fitz():
    import fitz  # PyMuPDF
    return fitz


# ═════════════════════════════════════════════════════════════════════════════
# CORE OCR ENGINE
# ═════════════════════════════════════════════════════════════════════════════
class ChristmanOCREngine:
    """Singleton PaddleOCR engine shared across beings."""

    _instance: Optional["ChristmanOCREngine"] = None

    def __init__(self):
        logger.info("Initialising PaddleOCR engine...")
        PaddleOCR = _import_paddleocr()
        try:
            self._ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except (TypeError, ValueError):
            self._ocr = PaddleOCR(lang="en")
        logger.info("PaddleOCR engine ready")

    @classmethod
    def get(cls) -> "ChristmanOCREngine":
        if cls._instance is None:
            cls._instance = ChristmanOCREngine()
        return cls._instance

    def extract(self, image) -> Dict:
        """Extract text from PIL Image."""
        try:
            img_array = np.array(image)
            result = self._ocr.ocr(img_array, cls=True)

            if not result or not result[0]:
                return self._empty_result()

            lines = []
            confidences = []
            words = []

            for detection in result[0]:
                _, (text, confidence) = detection
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                lines.append({"text": text, "confidence": float(confidence)})
                confidences.append(confidence)
                words.append(text)

            if not lines:
                return self._empty_result()

            return {
                "text": " ".join(words),
                "lines": lines,
                "confidence": float(np.mean(confidences)),
                "line_count": len(lines),
            }
        except Exception as e:
            logger.warning(f"OCR extraction error: {e}")
            return self._empty_result()

    @staticmethod
    def _empty_result() -> Dict:
        return {"text": "", "lines": [], "confidence": 0.0, "line_count": 0}


# ═════════════════════════════════════════════════════════════════════════════
# SCREEN CAPTURE
# ═════════════════════════════════════════════════════════════════════════════
class ScreenCapture:
    """Screen capture with change detection."""

    def __init__(self):
        self._last_hash: Optional[str] = None

    def grab(self):
        """Return (PIL.Image, changed: bool)"""
        ImageGrab, Image = _import_pil()
        try:
            screenshot = ImageGrab.grab()
            screen_hash = hashlib.md5(screenshot.tobytes()).hexdigest()
            changed = screen_hash != self._last_hash
            self._last_hash = screen_hash
            return screenshot, changed
        except Exception as e:
            if sys.platform.startswith("linux"):
                tmp = "/tmp/christman_screen.png"
                os.system(f"scrot {tmp} 2>/dev/null")
                if Path(tmp).exists():
                    screenshot = Image.open(tmp)
                    screen_hash = hashlib.md5(screenshot.tobytes()).hexdigest()
                    changed = screen_hash != self._last_hash
                    self._last_hash = screen_hash
                    return screenshot, changed
            logger.warning(f"Screen capture failed: {e}")
            return None, False


# ═════════════════════════════════════════════════════════════════════════════
# DOCUMENT LOADER
# ═════════════════════════════════════════════════════════════════════════════
class DocumentLoader:
    """Load images and PDFs into PIL Images."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    def load(self, file_path: str) -> List:
        """Load document and return list of PIL Images (one per page)."""
        _, Image = _import_pil()
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        if path.suffix.lower() == ".pdf":
            return self._load_pdf(file_path)

        if path.suffix.lower() in self.IMAGE_EXTENSIONS:
            try:
                img = Image.open(file_path).convert("RGB")
                logger.info(f"Loaded image: {path.name}")
                return [img]
            except Exception as e:
                logger.error(f"Failed to load image {file_path}: {e}")
                return []

        logger.warning(f"Unsupported file type: {path.suffix}")
        return []

    def _load_pdf(self, pdf_path: str) -> List:
        """Convert PDF pages to PIL Images."""
        try:
            fitz = _import_fitz()
            _, Image = _import_pil()
            doc = fitz.open(pdf_path)
            images = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                matrix = fitz.Matrix(PDF_DPI / 72, PDF_DPI / 72)
                pix = page.get_pixmap(matrix=matrix)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)

            doc.close()
            logger.info(f"Loaded PDF: {len(images)} pages from {Path(pdf_path).name}")
            return images
        except Exception as e:
            logger.error(f"PDF load failed: {e}")
            return []


# ═════════════════════════════════════════════════════════════════════════════
# DEREK AUDIO RELAY
# ═════════════════════════════════════════════════════════════════════════════
class DerekAudioRelay:
    """Routes text to Derek for synthesis with fallback."""

    def __init__(self, derek_uri: str = DEREK_WS_URI):
        self.derek_uri = derek_uri

    async def speak(self, text: str, voice_profile: str = "default", being: str = "Brockston") -> bool:
        """Send text to Derek. Returns True if successful."""
        if not text.strip():
            return False

        try:
            async with websockets.connect(self.derek_uri, open_timeout=3) as ws:
                payload = {
                    "command": "tts",
                    "payload": {
                        "text": text,
                        "voice_profile": voice_profile,
                        "source": "christman_ocr",
                        "being": being,
                    }
                }
                await ws.send(json.dumps(payload))
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                parsed = json.loads(response)
                if parsed.get("status") == "ok":
                    logger.info(f"{being} → Derek spoke {len(text)} chars")
                    return True
        except Exception:
            pass

        logger.info("Derek offline — using local TTS fallback")
        self._local_speak(text)
        return False

    @staticmethod
    def _local_speak(text: str):
        """System TTS fallback."""
        safe_text = text.replace('"', '\\"').replace("'", "\\'")
        if sys.platform == "darwin":
            os.system(f'say "{safe_text}"')
        elif sys.platform.startswith("linux"):
            os.system(f'espeak "{safe_text}" -s 150 2>/dev/null')
        else:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except ImportError:
                logger.warning(f"Text to speak: {text}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN BEING INTERFACE
# ═════════════════════════════════════════════════════════════════════════════
class ChristmanOCR:
    """Unified OCR interface for each being."""

    def __init__(
        self,
        being_name: str = "Brockston",
        derek_uri: str = DEREK_WS_URI,
        voice_profile: str = "default",
    ):
        self.being_name = being_name
        self.voice_profile = voice_profile
        self._engine = ChristmanOCREngine.get()
        self._screen = ScreenCapture()
        self._loader = DocumentLoader()
        self._relay = DerekAudioRelay(derek_uri)
        logger.info(f"{being_name} OCR ready | profile: {voice_profile}")

    async def read_screen(self) -> Dict:
        """Read current screen."""
        image, changed = self._screen.grab()
        if image is None:
            logger.warning(f"{self.being_name}: screen capture unavailable")
            return ChristmanOCREngine._empty_result()

        if not changed:
            logger.info(f"{self.being_name}: screen unchanged, skipping")
            return ChristmanOCREngine._empty_result()

        result = await asyncio.to_thread(self._engine.extract, image)

        if result["text"]:
            logger.info(f"Found {result['line_count']} lines (confidence {result['confidence']:.2f})")
            await self._relay.speak(result["text"], self.voice_profile, self.being_name)
        else:
            logger.info(f"{self.being_name}: no readable text on screen")

        return result

    async def read_document(self, file_path: str) -> Dict:
        """Read document (image or PDF)."""
        logger.info(f"{self.being_name}: reading document → {file_path}")
        images = await asyncio.to_thread(self._loader.load, file_path)

        if not images:
            logger.error(f"{self.being_name}: could not load {file_path}")
            return {"text": "", "pages": 0, "confidence": 0.0, "source": file_path}

        all_text = []
        confidences = []

        for page_num, image in enumerate(images, 1):
            result = await asyncio.to_thread(self._engine.extract, image)
            if result["text"]:
                prefix = f"Page {page_num}. " if len(images) > 1 else ""
                all_text.append(prefix + result["text"])
                confidences.append(result["confidence"])

        full_text = "\n\n".join(all_text)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        if full_text:
            logger.info(f"{self.being_name}: {len(full_text)} chars extracted, speaking...")
            await self._relay.speak(full_text, self.voice_profile, self.being_name)
        else:
            logger.info(f"{self.being_name}: no readable text found")

        return {
            "text": full_text,
            "pages": len(images),
            "confidence": avg_conf,
            "source": file_path,
        }

    async def watch_screen(self, interval: float = SCREEN_POLL_INTERVAL):
        """Continuous screen monitoring."""
        logger.info(f"{self.being_name}: starting screen watch (every {interval}s)")
        try:
            while True:
                await self.read_screen()
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            logger.info(f"{self.being_name}: screen watch stopped")
        except Exception as e:
            logger.error(f"{self.being_name}: watch error — {e}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
async def _cli_main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Christman OCR — shared engine for Brockston, AlphaVox, Seraphenia"
    )
    parser.add_argument("--screen", action="store_true", help="Read current screen")
    parser.add_argument("--scan", metavar="FILE", help="Scan image or PDF")
    parser.add_argument("--watch", action="store_true", help="Continuous screen watch")
    parser.add_argument("--interval", type=float, default=SCREEN_POLL_INTERVAL)
    parser.add_argument("--being", default="Brockston", choices=["Brockston", "AlphaVox", "Seraphenia"])
    parser.add_argument("--profile", default="default")

    args = parser.parse_args()

    ocr = ChristmanOCR(being_name=args.being, voice_profile=args.profile)

    if args.screen:
        await ocr.read_screen()
    elif args.scan:
        await ocr.read_document(args.scan)
    elif args.watch:
        await ocr.watch_screen(interval=args.interval)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(_cli_main())
