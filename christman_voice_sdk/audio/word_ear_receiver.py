# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
Porch / Filament → Sound live word wire. The receiving end.

Corti is the cochlea. That wire is corti_receiver.py on :5486.
This file is the word-ear. Different nerve. Same house.

What this is:
  - A localhost-only HTTP receiver. It binds 127.0.0.1 and nothing else.
  - One POST route. The body is one audio clip (WAV PCM, or JSON audioBase64).
  - The recognizer is SpeechRecognitionEngine.recognize_from_bytes.
  - The answer is a RecognitionResult. Failure text never lands in `text`.

What this is not:
  - Not Corti. No pitch card. No kind label.
  - Not a cloud ear. No xAI. No Whisper. No fallback that leaves the machine.

Failure behaviour:
  - No model / no vosk → UNAVAILABLE, text empty.
  - Silence → NO_SPEECH, text empty.
  - Unreadable audio → ERROR in metadata, text empty.
  - Nothing returns 200 with invented words.

Run: python3 -m christman_voice_sdk.audio.word_ear_receiver
"""

from __future__ import annotations

import io
import json
import logging
import os
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs

from .recognition_result import RecognitionResult, RecognitionStatus
from .speech_recognition_engine import SpeechRecognitionEngine, VOSK_SAMPLE_RATE

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)

DEFAULT_PORT = 5487  # Corti cards live on 5486. Words live next door.
BIND_HOST = "127.0.0.1"
MAX_BODY_BYTES = 8 * 1024 * 1024
EVENT_PATH = "/api/sound/word-ear"
HEALTH_PATH = "/health"


def allowed_origins() -> Tuple[str, ...]:
    raw = os.environ.get("CORTI_ALLOWED_ORIGINS", "")
    extra = tuple(o.strip() for o in raw.split(",") if o.strip())
    return DEFAULT_ALLOWED_ORIGINS + extra


def _pcm_from_wav(blob: bytes) -> Tuple[bytes, int, int]:
    with wave.open(io.BytesIO(blob), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"word-ear wants 16-bit PCM, got sample width {width}")
    if channels == 1:
        pcm = frames
    elif channels == 2:
        pcm = b"".join(frames[i : i + 2] for i in range(0, len(frames), 4))
    else:
        raise ValueError(f"word-ear wants mono or stereo, got {channels} channels")
    return pcm, rate, width


def _first_file_from_multipart(raw: bytes, content_type: str) -> Optional[bytes]:
    if "boundary=" not in content_type:
        return None
    boundary = content_type.split("boundary=", 1)[1].strip().encode("utf-8")
    marker = b"--" + boundary
    parts = raw.split(marker)
    for part in parts:
        if b"Content-Disposition" not in part:
            continue
        header, _, body = part.partition(b"\r\n\r\n")
        if not body:
            continue
        if body.endswith(b"\r\n"):
            body = body[:-2]
        if b"filename=" in header or b'name="file"' in header:
            return body
    return None


class WordEar:
    def __init__(self) -> None:
        self.engine = SpeechRecognitionEngine()

    def hear_bytes(self, blob: bytes) -> RecognitionResult:
        if not blob:
            return RecognitionResult.no_speech(source="word-ear", reason="empty_body")
        try:
            pcm, rate, width = _pcm_from_wav(blob)
        except Exception as exc:
            return RecognitionResult.error(
                f"Audio is not a readable WAV: {exc}",
                source="word-ear",
                kind="bad_wav",
            )
        return self.engine.recognize_from_bytes(
            pcm, sample_rate=rate or VOSK_SAMPLE_RATE, sample_width=width
        )

    def health(self) -> Dict[str, Any]:
        return {
            "status": "alive",
            "seat": "SpeechRecognitionEngine.recognize_from_bytes",
            "available": self.engine.available,
            "unavailable_reason": self.engine.unavailable_reason,
            "bind": f"{BIND_HOST}:{DEFAULT_PORT}{EVENT_PATH}",
            "cloud": False,
        }


def _payload(result: RecognitionResult) -> Dict[str, Any]:
    body = result.to_dict()
    body["ok"] = result.status is RecognitionStatus.OK
    if result.status is RecognitionStatus.OK:
        body["error"] = None
    else:
        body["error"] = result.metadata.get("error") or result.status.value
    return body


def _make_handler(ear: WordEar) -> type:
    origins = allowed_origins()

    class Handler(BaseHTTPRequestHandler):
        server_version = "ChristmanSoundWordEar/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            origin = self.headers.get("Origin")
            if origin in origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.end_headers()
            self.wfile.write(raw)

        def do_OPTIONS(self) -> None:
            origin = self.headers.get("Origin")
            if origin is not None and origin not in origins:
                self._send_json(403, {"ok": False, "error": f"origin not allowed: {origin}"})
                return
            self.send_response(204)
            if origin in origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:
            if self.path.split("?", 1)[0] != HEALTH_PATH:
                self._send_json(404, {"ok": False, "error": f"no such route: {self.path}"})
                return
            self._send_json(200, ear.health())

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != EVENT_PATH:
                self._send_json(404, {"ok": False, "error": f"no such route: {self.path}"})
                return
            origin = self.headers.get("Origin")
            if origin is not None and origin not in origins:
                self._send_json(403, {"ok": False, "error": f"origin not allowed: {origin}"})
                return
            length_header = self.headers.get("Content-Length")
            try:
                length = int(length_header or "")
            except ValueError:
                self._send_json(411, {"ok": False, "error": "Content-Length required"})
                return
            if length > MAX_BODY_BYTES:
                self._send_json(413, {"ok": False, "error": "clip exceeds 8 MB"})
                return
            raw = self.rfile.read(length)
            ctype = (self.headers.get("Content-Type") or "").lower()
            blob: Optional[bytes] = None
            if "multipart/form-data" in ctype:
                blob = _first_file_from_multipart(raw, ctype)
            elif "application/json" in ctype:
                try:
                    card = json.loads(raw)
                except json.JSONDecodeError as exc:
                    self._send_json(400, {"ok": False, "error": f"body is not JSON: {exc}"})
                    return
                if not isinstance(card, dict):
                    self._send_json(400, {"ok": False, "error": "body must be a JSON object"})
                    return
                b64 = card.get("audioBase64") or card.get("audio_base64")
                if isinstance(b64, str) and b64.strip():
                    import base64

                    try:
                        blob = base64.b64decode(b64)
                    except Exception:
                        self._send_json(400, {"ok": False, "error": "audioBase64 is not readable"})
                        return
            elif raw:
                blob = raw
            if not blob:
                result = RecognitionResult.no_speech(source="word-ear", reason="no_audio")
            else:
                result = ear.hear_bytes(blob)
            code = 200 if result.status is RecognitionStatus.OK else 503
            if result.status is RecognitionStatus.NO_SPEECH:
                code = 200
            self._send_json(code, _payload(result))

    return Handler


def create_receiver(port: Optional[int] = None) -> Tuple[ThreadingHTTPServer, WordEar]:
    if port is None:
        raw = os.environ.get("CHRISTMAN_WORD_EAR_PORT", str(DEFAULT_PORT))
        try:
            port = int(raw)
        except ValueError:
            raise ValueError(f"CHRISTMAN_WORD_EAR_PORT={raw!r} is not a port number") from None
    ear = WordEar()
    server = ThreadingHTTPServer((BIND_HOST, port), _make_handler(ear))
    return server, ear


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server, ear = create_receiver()
    host, port = server.server_address[0], server.server_address[1]
    logger.info("Word ear listening on http://%s:%d%s", host, port, EVENT_PATH)
    logger.info("health: http://%s:%d%s", host, port, HEALTH_PATH)
    logger.info("engine available: %s", ear.engine.available)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
        server.shutdown()
    return 0


__all__ = ["WordEar", "create_receiver", "main", "EVENT_PATH", "HEALTH_PATH", "DEFAULT_PORT"]


if __name__ == "__main__":
    raise SystemExit(main())
