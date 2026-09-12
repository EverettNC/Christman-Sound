"""The mill: HTTP seam between a show and the Christman Voice SDK.

The show (PODCASTLIVE) posts {text, being, reference_audio_path?, emotion_params?}
to /generate and gets back a WAV plus phoneme-timed viseme frames. Everything
runs here, on this machine, through the SDK. No hosted TTS. No key.

Run from the repo root:  python3 -m christman_voice_sdk.mill
Env: CHRISTMAN_MILL_PORT (default 5000), CHRISTMAN_OUTPUT_DIR (takes land under it).

Degrades loudly: /status lists exactly which modules and voices are missing, and
/generate refuses with the same list instead of returning silence.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REQUIRED = ("torch", "TTS", "soundfile")
BEINGS = ("everett", "patty", "brandon")
OUT = Path(os.environ.get("CHRISTMAN_OUTPUT_DIR", "/tmp/christman_sdk")) / "mill"


def missing_modules() -> list[str]:
    return [m for m in REQUIRED if importlib.util.find_spec(m) is None]


def reference_for(being: str) -> dict:
    """Where this being's voice comes from, or why it cannot be found."""
    try:
        from CHRISTMAN_EAR_CANAL.VOICES import resolve_being_reference
    except Exception as exc:  # the resolver itself is broken: say so, do not guess a path
        return {"path": None, "error": f"{type(exc).__name__}: {exc}"}
    path = resolve_being_reference(being)
    return {"path": str(path) if path else None, "error": None if path else "no reference WAV found"}


def status() -> dict:
    missing = missing_modules()
    return {
        "ok": not missing,
        "engine": "xtts_v2",
        "degraded": bool(missing),
        "missing": missing,
        "mfa": shutil.which("mfa") is not None,
        "beings": {b: reference_for(b) for b in BEINGS},
    }


def generate(body: dict) -> tuple[int, dict]:
    text = str(body.get("text") or "").strip()
    being = str(body.get("being") or "brandon").strip().lower()
    if not text:
        return 400, {"error": "empty text", "stage": "tts"}
    missing = missing_modules()
    if missing:
        return 503, {"error": "degraded", "stage": "tts", "missing": missing}
    ref = body.get("reference_audio_path") or reference_for(being)["path"]
    if not ref:
        return 503, {"error": "no-reference", "stage": "tts", "being": being}

    from christman_voice_sdk.synthesis.voice_synthesis import synthesize_speech

    wav = synthesize_speech(text, body.get("emotion_params") or {}, str(ref))
    if not wav or not Path(wav).is_file():
        return 503, {"error": "synthesis returned no audio", "stage": "tts", "being": being}

    OUT.mkdir(parents=True, exist_ok=True)
    take = uuid.uuid4().hex
    dst = OUT / f"{take}.wav"
    shutil.move(str(wav), dst)
    with wave.open(str(dst), "rb") as w:
        duration = w.getnframes() / w.getframerate()

    from christman_voice_sdk.synthesis.phoneme_labeler import PhonemeLabeler

    labeler = PhonemeLabeler()
    phonemes = labeler.label_audio(dst, text)
    frames = labeler.phonemes_to_visemes(phonemes, fps=30)
    (OUT / f"{take}.lipsync.json").write_text(json.dumps(frames))
    return 200, {
        "audio_url": f"/take/{take}.wav",
        "lipsync_url": f"/take/{take}.lipsync.json",
        "duration": duration,
        "being": being,
        "reference": str(ref),
        "phoneme_source": "mfa" if labeler.mfa_available else "energy",
    }


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/status":
            self._json(200, status())
            return
        if self.path.startswith("/take/"):
            name = Path(self.path[len("/take/"):]).name
            f = OUT / name
            if not f.is_file():
                self._json(404, {"error": "no such take", "take": name})
                return
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav" if name.endswith(".wav") else "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._json(404, {"error": "not found", "routes": ["/status", "/generate", "/take/<name>"]})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/generate", "/speak"):
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": f"bad json: {exc}"})
            return
        try:
            code, obj = generate(body)
        except Exception as exc:  # loud: the caller sees the real failure
            code, obj = 500, {"error": f"{type(exc).__name__}: {exc}", "stage": "tts"}
        self._json(code, obj)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[mill] {fmt % args}")


def main() -> None:
    port = int(os.environ.get("CHRISTMAN_MILL_PORT", "5000"))
    print(f"[mill] status: {json.dumps(status())}")
    print(f"[mill] listening on http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
