# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
Corti → Sound live wire. The receiving end.

Corti is the cochlea. This file is the last inch on the Sound side.
The Corti client (`sound-live.ts`) posts every closed VocalEvent card
here. This seats it: `Hearing.hear(corti_event=card, text=None)`.
corti_ingest strips the client `kind` at that boundary.

What this is:
  - A localhost-only HTTP receiver. It binds 127.0.0.1 and nothing else.
    Audio of a vulnerable user does not leave the device; a bind address
    is how that promise is kept, not a comment.
  - One POST route. The body is one closed Corti VocalEvent card
    (durationMs, peakRms, f0, tape, …).

What this is not:
  - Not a SoundDetectorBackend. The ear measures; nothing here classifies.
  - Not a speech pipeline. No text arrives on this wire, so
    structural_affect and harm_frame do not run — and the turn says so
    in its notes instead of pretending they did.

Failure behaviour (Rule 6):
  - Malformed JSON is a 400 with the parse error in the body.
  - A body that is not a JSON object is a 400.
  - An oversized body is a 413, read no further.
  - A browser origin not on the allowlist is a 403, named in the body.
  - A consumer callback that raises is logged with traceback and counted;
    the count is visible in /health. Nothing returns 200 with a lie in it.

Nothing is constructed at import. Build a receiver with `create_receiver`
or run `python3 -m christman_voice_sdk.audio.corti_receiver`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from .hearing import Hearing, HearingTurn

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# The Corti client dev server (vite --port 8080). Override or extend with
# CORTI_ALLOWED_ORIGINS, comma-separated.
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:8080",
    "http://127.0.0.1:8080",
)

DEFAULT_PORT = 5486  # AlphaVox acoustic keys live on 5485; the card lives next door.
BIND_HOST = "127.0.0.1"  # Not configurable. Offline-first is a bind address.
MAX_BODY_BYTES = 2 * 1024 * 1024  # A 360-frame tape is ~20 KB. 2 MB is already generous.
EVENT_PATH = "/api/sound/corti-event"
HEALTH_PATH = "/health"
RING_SIZE = 64


def allowed_origins() -> Tuple[str, ...]:
    raw = os.environ.get("CORTI_ALLOWED_ORIGINS", "")
    extra = tuple(o.strip() for o in raw.split(",") if o.strip())
    return DEFAULT_ALLOWED_ORIGINS + extra


@dataclass
class ReceiverState:
    """Counters the /health route reports. Every number here was counted, not styled."""

    received: int = 0
    rejected: int = 0
    callback_failures: int = 0
    last_event_at: Optional[float] = None
    ring: Deque[HearingTurn] = field(default_factory=lambda: deque(maxlen=RING_SIZE))
    lock: threading.Lock = field(default_factory=threading.Lock)


class CortiReceiver:
    """
    Holds the seat. One Hearing, one ring of turns, one optional consumer.

    `on_turn` is called with each HearingTurn after it lands in the ring.
    If it raises, the exception is logged with traceback and counted in
    /health as callback_failures — the wire stays up, the failure stays
    visible.
    """

    def __init__(
        self,
        hearing: Optional[Hearing] = None,
        on_turn: Optional[Callable[[HearingTurn], None]] = None,
    ) -> None:
        self.hearing = hearing or Hearing()
        self.on_turn = on_turn
        self.state = ReceiverState()

    def take_card(self, card: Dict[str, Any]) -> HearingTurn:
        """One card in, one turn out. No text on this wire, and the turn says so."""
        turn = self.hearing.hear(corti_event=card, text=None)
        with self.state.lock:
            self.state.received += 1
            self.state.last_event_at = turn.ingest.event.timestamp if turn.ingest else None
            self.state.ring.appendleft(turn)
        if self.on_turn is not None:
            try:
                self.on_turn(turn)
            except Exception:
                logger.exception("corti_receiver consumer callback raised")
                with self.state.lock:
                    self.state.callback_failures += 1
        return turn

    def health(self) -> Dict[str, Any]:
        with self.state.lock:
            return {
                "status": "alive",
                "seat": "Hearing.hear(corti_event, text=None)",
                "received": self.state.received,
                "rejected": self.state.rejected,
                "callback_failures": self.state.callback_failures,
                "ring_size": len(self.state.ring),
                "last_event_at": self.state.last_event_at,
            }


def _make_handler(receiver: CortiReceiver) -> type:
    origins = allowed_origins()

    class Handler(BaseHTTPRequestHandler):
        server_version = "ChristmanSoundCortiWire/1.0"

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
            self._send_json(code, {"error": reason})

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
            if self.path != HEALTH_PATH:
                self._reject(404, f"no such route: {self.path}")
                return
            self._send_json(200, receiver.health())

        def do_POST(self) -> None:
            if self.path != EVENT_PATH:
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
            try:
                card = json.loads(raw)
            except json.JSONDecodeError as err:
                self._reject(400, f"body is not JSON: {err}")
                return
            if not isinstance(card, dict):
                self._reject(400, f"card must be a JSON object, got {type(card).__name__}")
                return

            turn = receiver.take_card(card)
            self._send_json(200, turn.to_dict())

    return Handler


def create_receiver(
    port: Optional[int] = None,
    hearing: Optional[Hearing] = None,
    on_turn: Optional[Callable[[HearingTurn], None]] = None,
) -> Tuple[ThreadingHTTPServer, CortiReceiver]:
    """
    Build the server without starting it. Port 0 asks the OS for a free one
    (tests). Default port is CHRISTMAN_SOUND_PORT or 5486.
    """
    if port is None:
        raw = os.environ.get("CHRISTMAN_SOUND_PORT", str(DEFAULT_PORT))
        try:
            port = int(raw)
        except ValueError:
            raise ValueError(
                f"CHRISTMAN_SOUND_PORT={raw!r} is not a port number"
            ) from None
    receiver = CortiReceiver(hearing=hearing, on_turn=on_turn)
    server = ThreadingHTTPServer((BIND_HOST, port), _make_handler(receiver))
    return server, receiver


def start_in_thread(
    port: Optional[int] = None,
    hearing: Optional[Hearing] = None,
    on_turn: Optional[Callable[[HearingTurn], None]] = None,
) -> Tuple[ThreadingHTTPServer, CortiReceiver, threading.Thread]:
    """For embedding in a being or a test. Caller owns server.shutdown()."""
    server, receiver = create_receiver(port=port, hearing=hearing, on_turn=on_turn)
    thread = threading.Thread(
        target=server.serve_forever, name="corti-receiver", daemon=True
    )
    thread.start()
    return server, receiver, thread


def main(argv: Optional[List[str]] = None) -> int:
    del argv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server, receiver = create_receiver()
    host, port = server.server_address[0], server.server_address[1]
    logger.info("Corti wire listening on http://%s:%d%s", host, port, EVENT_PATH)
    logger.info("health: http://%s:%d%s", host, port, HEALTH_PATH)
    logger.info("allowed browser origins: %s", ", ".join(allowed_origins()))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down — received %d cards", receiver.state.received)
        server.shutdown()
    return 0


__all__ = [
    "CortiReceiver",
    "ReceiverState",
    "create_receiver",
    "start_in_thread",
    "main",
    "EVENT_PATH",
    "HEALTH_PATH",
    "DEFAULT_PORT",
]

if __name__ == "__main__":
    raise SystemExit(main())
