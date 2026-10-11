# ===============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ===============================================================================

"""
Word-ear → Sound live wire. The receiving end for words.

Corti is pitch. This file is words.
Filament and Porch post one catch of 16-bit mono WAV here.
This seats it: SpeechRecognitionEngine.recognize_from_bytes(...).

What this is:
  - A localhost-only HTTP receiver. It binds 127.0.0.1 and nothing else.
    A vulnerable user's voice does not leave the device; a bind address
    is how that promise is kept, not a comment.
  - One POST route. The body is WAV bytes, a multipart file, or JSON
    {audioBase64}. Anything else is a 400 with the reason in the body.

What this is not:
  - Not Corti. No VocalEvent card is read here.
  - Not a cloud mill. There is no xAI path and no Whisper path.
  - Not a decoder for webm/mp3. The mic already writes WAV. If the
    body is not WAV, the turn says so instead of pretending.

Failure behaviour (Rule 6):
  - Malformed body is a 400 with the parse error named.
  - A body that is not WAV is a 400.
  - An oversized body is a 413, read no further.
  - A browser origin not on the allowlist is a 403, named in the body.
  - Engine UNAVAILABLE / ERROR is returned as JSON with empty text.
    Diagnostics live in metadata.error. Never in the text slot.

Nothing is constructed at import. Build a receiver with `create_receiver`
or run `python3 -m christman_voice_sdk.audio.word_ear_receiver`.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
import wave
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from urllib.parse import parse_qs

from .recognition_result import RecognitionResult, RecognitionStatus
from .speech_recognition_engine import SpeechRecognitionEngine

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
)

DEFAULT_PORT = 5487  # Corti cards live on 5486. Words live next door.
BIND_HOST = "127.0.0.1"  # Not configurable. Offline-first is a bind address.
MAX_BODY_BYTES = 8 * 1024 * 1024
EVENT_PATH = "/api/sound/word-ear"
HEALTH_PATH = "/health"
RING_SIZE = 64


def allowed_origins() -> Tuple[str, ...]:
    raw = os.environ.get("WORD_EAR_ALLOWED_ORIGINS", "")
    extra = tuple(o.strip() for o in raw.split(",") if o.strip())
    return DEFAULT_ALLOWED_ORIGINS + extra


def pcm_from_wav(blob: bytes) -> Tuple[bytes, int, int]:
    """Strip a RIFF WAV to 16-bit mono PCM. Raise ValueError if it is not that."""
    try:
        with wave.open(io.BytesIO(blob), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except wave.Error as exc:
        raise ValueError(f"body is not a WAV: {exc}") from exc
    if channels != 1:
        raise ValueError(f"word-ear wants mono WAV, got {channels} channels")
    if width != 2:
        raise ValueError(f"word-ear wants 16-bit PCM, got {width * 8}-bit")
    if not frames:
        raise ValueError("WAV contained no frames")
    return frames, rate, width


def _boundary(content_type: str) -> Optional[bytes]:
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            value = part.split("=", 1)[1].strip().strip('"')
            return value.encode("ascii", errors="replace")
    return None


def extract_audio_blob(raw: bytes, content_type: str) -> bytes:
    """
    Pull WAV bytes out of raw body, multipart file=, or JSON audioBase64.
    Does not decode codecs. WAV only after this returns.
    """
    ctype = (content_type or "").split(";", 1)[0].strip().lower()

    if ctype in {"audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"}:
        return raw

    if ctype == "application/json":
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"JSON body is not readable: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        b64 = payload.get("audioBase64") or payload.get("audio_base64")
        if not isinstance(b64, str) or not b64.strip():
            raise ValueError("JSON body needs audioBase64")
        try:
            return base64.b64decode(b64, validate=False)
        except Exception as exc:
            raise ValueError(f"audioBase64 is not base64: {exc}") from exc

    if ctype.startswith("multipart/"):
        mark = _boundary(content_type)
        if not mark:
            raise ValueError("multipart body has no boundary")
        token = b"--" + mark
        chunks = raw.split(token)
        for chunk in chunks:
            if b"filename=" not in chunk and b'name="file"' not in chunk:
                continue
            split_at = chunk.find(b"\r\n\r\n")
            if split_at < 0:
                continue
            blob = chunk[split_at + 4 :]
            if blob.endswith(b"\r\n"):
                blob = blob[:-2]
            if blob.endswith(b"--"):
                blob = blob[:-2]
            if blob:
                return blob
        raise ValueError("multipart body had no file part")

    raise ValueError(f"unsupported Content-Type: {content_type or 'missing'}")


@dataclass
class ReceiverState:
    received: int = 0
    rejected: int = 0
    last_status: Optional[str] = None
    ring: Deque[RecognitionResult] = field(default_factory=lambda: deque(maxlen=RING_SIZE))
    lock: threading.Lock = field(default_factory=threading.Lock)


class WordEarReceiver:
    """Holds the seat. One engine, one ring of results."""

    def __init__(self, engine: Optional[SpeechRecognitionEngine] = None) -> None:
        self.engine = engine or SpeechRecognitionEngine()
        self.state = ReceiverState()

    def take_wav(self, wav_bytes: bytes) -> RecognitionResult:
        try:
            pcm, rate, width = pcm_from_wav(wav_bytes)
        except ValueError as exc:
            return RecognitionResult.error(str(exc), source="word-ear", kind="not_wav")
        result = self.engine.recognize_from_bytes(
            pcm, sample_rate=rate, sample_width=width
        )
        with self.state.lock:
            self.state.received += 1
            self.state.last_status = result.status.value
            self.state.ring.appendleft(result)
        return result

    def health(self) -> Dict[str, Any]:
        with self.state.lock:
            return {
                "status": "alive",
                "seat": "SpeechRecognitionEngine.recognize_from_bytes",
                "bind": f"{BIND_HOST}",
                "path": EVENT_PATH,
                "engine_available": self.engine.available,
                "unavailable_reason": self.engine.unavailable_reason,
                "received": self.state.received,
                "rejected": self.state.rejected,
                "last_status": self.state.last_status,
                "ring_size": len(self.state.ring),
            }


def result_payload(result: RecognitionResult) -> Dict[str, Any]:
    """Transport JSON. text is empty on every non-OK status."""
    payload = result.to_dict()
    payload["ok"] = result.is_user_speech
    if result.is_user_speech:
        payload["error"] = None
    else:
        payload["error"] = result.metadata.get("error") or result.status.value
        payload["text"] = ""
    payload["duration"] = result.metadata.get("duration")
    payload["language"] = result.metadata.get("language")
    return payload


def _make_handler(receiver: WordEarReceiver) -> type:
    origins = allowed_origins()

    class Handler(BaseHTTPRequestHandler):
        server_version = "ChristmanSoundWordEar/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors()
            self.end_headers()
            self.wfile.write(body)

        def _origin(self) -> Optional[str]:
            return self.headers.get("Origin")

        def _origin_allowed(self) -> bool:
            origin = self._origin()
            return origin is None or origin in origins

        def _send_cors(self) -> None:
            origin = self._origin()
            if origin is not None and origin in origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def _reject(self, code: int, reason: str) -> None:
            with receiver.state.lock:
                receiver.state.rejected += 1
            logger.warning("rejected (%d): %s", code, reason)
            self._send_json(code, {"ok": False, "text": "", "error": reason})

        def do_OPTIONS(self) -> None:
            if not self._origin_allowed():
                self._reject(403, f"origin not allowed: {self._origin()}")
                return
            self.send_response(204)
            self._send_cors()
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_GET(self) -> None:
            if self.path.split("?", 1)[0] != HEALTH_PATH:
                self._reject(404, f"no such route: {self.path}")
                return
            self._send_json(200, receiver.health())

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path != EVENT_PATH:
                self._reject(404, f"no such route: {self.path}")
                return
            if not self._origin_allowed():
                self._reject(403, f"origin not allowed: {self._origin()}")
                return

            length_header = self.headers.get("Content-Length")
            try:
                length = int(length_header or "")
            except ValueError:
                self._reject(411, "Content-Length required")
                return
            if length > MAX_BODY_BYTES:
                self._reject(413, f"body {length} bytes exceeds cap {MAX_BODY_BYTES}")
                return

            raw = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", "")
            try:
                blob = extract_audio_blob(raw, content_type)
            except ValueError as exc:
                self._reject(400, str(exc))
                return

            result = receiver.take_wav(blob)
            code = 200
            if result.status is RecognitionStatus.UNAVAILABLE:
                code = 503
            elif result.status is RecognitionStatus.ERROR:
                code = 500
            self._send_json(code, result_payload(result))

    return Handler


def create_receiver(
    port: Optional[int] = None,
    engine: Optional[SpeechRecognitionEngine] = None,
) -> Tuple[ThreadingHTTPServer, WordEarReceiver]:
    if port is None:
        raw = os.environ.get("CHRISTMAN_WORD_EAR_PORT", str(DEFAULT_PORT))
        try:
            port = int(raw)
        except ValueError:
            raise ValueError(
                f"CHRISTMAN_WORD_EAR_PORT={raw!r} is not a port number"
            ) from None
    receiver = WordEarReceiver(engine=engine)
    server = ThreadingHTTPServer((BIND_HOST, port), _make_handler(receiver))
    return server, receiver


def start_in_thread(
    port: Optional[int] = None,
    engine: Optional[SpeechRecognitionEngine] = None,
) -> Tuple[ThreadingHTTPServer, WordEarReceiver, threading.Thread]:
    server, receiver = create_receiver(port=port, engine=engine)
    thread = threading.Thread(
        target=server.serve_forever, name="word-ear-receiver", daemon=True
    )
    thread.start()
    return server, receiver, thread


def main(argv: Optional[List[str]] = None) -> int:
    del argv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server, receiver = create_receiver()
    host, port = server.server_address[0], server.server_address[1]
    logger.info("Word-ear listening on http://%s:%d%s", host, port, EVENT_PATH)
    logger.info("health: http://%s:%d%s", host, port, HEALTH_PATH)
    logger.info("engine available: %s", receiver.engine.available)
    if receiver.engine.unavailable_reason:
        logger.error("engine unavailable: %s", receiver.engine.unavailable_reason)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down — received %d catches", receiver.state.received)
        server.shutdown()
    return 0


__all__ = [
    "WordEarReceiver",
    "ReceiverState",
    "create_receiver",
    "start_in_thread",
    "main",
    "EVENT_PATH",
    "HEALTH_PATH",
    "DEFAULT_PORT",
    "pcm_from_wav",
]


if __name__ == "__main__":
    raise SystemExit(main())
