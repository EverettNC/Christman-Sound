"""
christman_ocr_shared.py
The Christman AI Project — Luma Cognify AI
OCR that reads printed text (a screen, an image, a PDF) so it can be spoken
aloud in the caller's own voice.

WHAT CHANGED AND WHY (2026-10-29) — BEING-NEUTRAL, SPOKEN BY THE PACKAGE
------------------------------------------------------------------------
This module used to name the beings it served, default to one of them, and
send the text it read over a websocket to another being's relay to be
spoken. The package holds no being's identity, so all of that is gone:

  * no default being and no list of beings: `ChristmanOCR(voice=...)` takes
    the caller's voice and nothing else about the caller;
  * no relay and no relay address: the text is spoken by this package's own
    mouth, `christman_sound.synth.speak`, in the voice the caller passed;
  * `voice=` is required. With no voice there is nothing to speak in, so
    `ReadAloud` refuses at construction (ValueError) instead of choosing one.

A read-aloud that cannot be voiced raises `SynthesisUnavailable` with the
real reason (no voice could be loaded, no lexicon, no C library, nothing
played), chained to the error that caused it. There is no fallback voice:
the outside engines (`say`, espeak, pyttsx3) were cut on 2026-10-17 and stay
cut. `read_screen` and `read_document` let that typed failure through;
`watch_screen` logs any error and stops. `subprocess` and `sys` stay: the
scrot screen grab and the platform checks there use them.
"""

from __future__ import annotations

import asyncio
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from christman_voice_sdk.engines.synthesis_result import SynthesisUnavailable
    from christman_voice_sdk.utils.logger import get_logger
except ImportError:
    from engines.synthesis_result import SynthesisUnavailable
    from utils.logger import get_logger

logger = get_logger(__name__)

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
    """Singleton PaddleOCR engine, shared by every caller in the process."""

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
                "error": None,
            }
        except Exception as e:
            # Rule 6: an engine failure is not "no text on the page". The
            # result carries the error so a caller can tell them apart.
            logger.error(f"OCR extraction failed: {e}", exc_info=True)
            return self._empty_result(error=f"{type(e).__name__}: {e}")

    @staticmethod
    def _empty_result(error: Optional[str] = None) -> Dict:
        return {"text": "", "lines": [], "confidence": None, "line_count": 0,
                "error": error}


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
                try:
                    subprocess.run(["scrot", tmp], check=False,
                                   stderr=subprocess.DEVNULL, timeout=10)
                except (OSError, subprocess.TimeoutExpired) as scrot_exc:
                    logger.warning(f"scrot fallback failed: {scrot_exc}")
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
# READ-ALOUD, IN THE CALLER'S VOICE
# ═════════════════════════════════════════════════════════════════════════════
class ReadAloud:
    """Speaks text through christman_sound.synth in the voice the caller gives."""

    def __init__(self, voice: Any, lexicon_path: Optional[Union[str, Path]] = None):
        """
        voice         a christman_sound VoiceProfile or FrameVoice, or a path:
                      a frame voice folder, or a voice profile JSON file.
        lexicon_path  the Vosk alignment lexicon; None leaves it to
                      christman_sound (which reads $CHRISTMAN_LEXICON_PATH).
        """
        if voice is None:
            raise ValueError("ReadAloud needs the caller's voice (voice=); "
                             "it does not choose one")
        self.voice = voice
        self.lexicon_path = lexicon_path

    def _loaded_voice(self):
        if not isinstance(self.voice, (str, Path)):
            return self.voice
        path = Path(self.voice).expanduser()
        if path.is_dir():
            from christman_sound.stitch import load_frame_voice
            return load_frame_voice(path)
        from christman_sound.voice_profile import load_voice_profile
        return load_voice_profile(path)

    def _speak_now(self, text: str) -> None:
        from christman_sound.synth import speak
        kwargs = {"voice": self._loaded_voice()}
        if self.lexicon_path is not None:
            kwargs["lexicon_path"] = self.lexicon_path
        speak(text, **kwargs)

    async def speak(self, text: str) -> bool:
        """
        Speak `text`. Returns True once it played, False for empty text.
        Raises `SynthesisUnavailable`, with the reason, when it was not voiced.
        """
        if not text.strip():
            return False
        try:
            await asyncio.to_thread(self._speak_now, text)
        except Exception as exc:
            err = SynthesisUnavailable(
                f"read-aloud was not voiced ({type(exc).__name__}: {exc})",
                where="OCR read-aloud",
            )
            logger.error(f"Text was NOT voiced: {err}")
            raise err from exc
        logger.info(f"Read aloud {len(text)} chars")
        return True


# ═════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ═════════════════════════════════════════════════════════════════════════════
class ChristmanOCR:
    """Reads a screen or a document and speaks what it read in the caller's voice."""

    def __init__(self, *, voice: Any, lexicon_path: Optional[Union[str, Path]] = None):
        self._voice = ReadAloud(voice, lexicon_path)
        self._engine = ChristmanOCREngine.get()
        self._screen = ScreenCapture()
        self._loader = DocumentLoader()
        logger.info("OCR ready")

    async def read_screen(self) -> Dict:
        """Read current screen."""
        image, changed = self._screen.grab()
        if image is None:
            logger.warning("screen capture unavailable")
            return ChristmanOCREngine._empty_result()

        if not changed:
            logger.info("screen unchanged, skipping")
            return ChristmanOCREngine._empty_result()

        result = await asyncio.to_thread(self._engine.extract, image)

        if result["text"]:
            logger.info(f"Found {result['line_count']} lines (confidence {result['confidence']:.2f})")
            await self._voice.speak(result["text"])
        else:
            logger.info("no readable text on screen")

        return result

    async def read_document(self, file_path: str) -> Dict:
        """Read document (image or PDF)."""
        logger.info(f"reading document → {file_path}")
        images = await asyncio.to_thread(self._loader.load, file_path)

        if not images:
            logger.error(f"could not load {file_path}")
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
            logger.info(f"{len(full_text)} chars extracted, speaking...")
            await self._voice.speak(full_text)
        else:
            logger.info("no readable text found")

        return {
            "text": full_text,
            "pages": len(images),
            "confidence": avg_conf,
            "source": file_path,
        }

    async def watch_screen(self, interval: float = SCREEN_POLL_INTERVAL):
        """Continuous screen monitoring."""
        logger.info(f"starting screen watch (every {interval}s)")
        try:
            while True:
                await self.read_screen()
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            logger.info("screen watch stopped")
        except Exception as e:
            logger.error(f"screen watch error — {e}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
async def _cli_main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Christman OCR — read printed text aloud in the caller's voice"
    )
    parser.add_argument("--screen", action="store_true", help="Read current screen")
    parser.add_argument("--scan", metavar="FILE", help="Scan image or PDF")
    parser.add_argument("--watch", action="store_true", help="Continuous screen watch")
    parser.add_argument("--interval", type=float, default=SCREEN_POLL_INTERVAL)
    parser.add_argument("--voice", required=True,
                        help="frame voice folder, or voice profile JSON, to speak in")
    parser.add_argument("--lexicon", default=None, help="Vosk alignment lexicon")

    args = parser.parse_args()

    ocr = ChristmanOCR(voice=args.voice, lexicon_path=args.lexicon)

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
