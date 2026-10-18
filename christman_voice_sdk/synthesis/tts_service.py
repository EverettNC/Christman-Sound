"""
TTS Service - Local API

WHAT CHANGED AND WHY (2026-10-18) — THE SOVEREIGN MOUTH IS WIRED
----------------------------------------------------------------
voice_synthesis now renders through christman_synth (engine "christman").
Two things here follow from that:

  - The content-hash cache key now includes the voice that made the file
    (engine name, f0, and the lexicon file's identity). Before, the key was
    text, reference, language and emotion only, so a WAV left in
    static/audio by the old XTTS or `say` renderer for the same text would
    have been served as this voice's output with status "ok".
  - The lexicon's and the synthesizer's typed failures reach /generate
    unchanged and are answered with `available: false` and the reason:
    503 when the lexicon is missing or unreadable (LexiconUnavailable,
    LexiconFormatError), 422 when the text itself cannot be voiced
    (WordNotInLexicon with the missing `words`, PhoneNotMapped,
    NothingToSay). Before, those would have been a bare 500.

WHAT CHANGED AND WHY (2026-10-17) — OUTSIDE VOICE CUT
-----------------------------------------------------
Everett's ruling: every outside speech engine is cut. This service rendered
through voice_synthesis, whose only renderer was Coqui XTTS (now in
quarantine/2026-10_outside_voices/). Removed (quoted from main 99a6cce):

    (docstring) Generate speech WAV from text using local XTTS zero-shot cloning.
    ...
    if not result_path:
        raise RuntimeError("Local synthesis failed and returned None.")
    ...
        "engine": "xtts_v2_local",

`/status` reported "xtts_v2_local" as a fixed string whether or not XTTS had
loaded. It now reports `synthesis_available` and the real reason from
`voice_synthesis.synthesis_readiness()`. `/generate` answers 503 with that
reason (`available: false`) when no voice can be made, instead of a 500.
`local_text_to_speech` lets the typed `SynthesisUnavailable` through to the
caller. The content-hash cache (text, reference, language and emotion in the
key) is kept whole.

EARLIER HISTORY OF THIS FILE
----------------------------
1. CLOUD EGRESS TERMINATED.
   The predecessor used `gTTS` inside Flask endpoints, shipping user text 
   to Google Cloud to generate MP3s. This violated Cardinal Rule 13 and patient data 
   sovereignty. It has been replaced with the local `SpeechSynthesisEngine` wrapper.

2. INVENTED VOICE PROFILES REMOVED.
   The old service hardcoded a list of fake voices mapped to `gTTS` domain suffixes 
   (e.g., "us_male", "uk_female") which had no corresponding acoustic models.
   The service now accepts a speaker reference path to perform true zero-shot 
   cloning, entirely locally.
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

from flask import Flask, jsonify, send_from_directory, send_file, request
from werkzeug.utils import secure_filename
from .voice_synthesis import (
    ENGINE_NAME,
    EVERETT_F0_MEDIAN_HZ,
    LexiconError,
    NothingToSay,
    get_speech_synthesis_engine,
    synthesis_readiness,
)
from ..engines.synthesis_result import SynthesisUnavailable

try:
    from christman_sound.lexicon import LEXICON_ENV, LexiconFormatError, LexiconUnavailable
except Exception:  # christman_sound not importable: voice_synthesis already reports why
    LEXICON_ENV = "CHRISTMAN_LEXICON_PATH"
    class LexiconUnavailable(LexiconError): ...  # type: ignore[no-redef]
    class LexiconFormatError(LexiconError): ...  # type: ignore[no-redef]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: 2026-10-11: the service bound 0.0.0.0 — a text-in, file-out endpoint
#: open to the whole network of an AAC device. Loopback by default; set
#: CHRISTMAN_TTS_HOST to widen it on purpose.
DEFAULT_HOST = os.getenv("CHRISTMAN_TTS_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("CHRISTMAN_TTS_PORT", "5000"))

def _static_dir(*parts: str) -> str:
    return os.path.join(os.getcwd(), "static", *parts)

def _hash_content(*parts: str) -> str:
    payload = "|".join(parts)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

def _voice_tag() -> str:
    """
    Which voice made a cached file: engine, f0, and the lexicon file's
    identity (path, size, mtime). Part of the cache key so a file made by any
    other voice, or from another lexicon, is never served as this one.
    """
    lexicon = os.environ.get(LEXICON_ENV, "")
    ident = "no-lexicon"
    if lexicon:
        try:
            st = os.stat(lexicon)
            ident = f"{os.path.realpath(lexicon)}:{st.st_size}:{st.st_mtime_ns}"
        except OSError as exc:
            ident = f"unreadable:{lexicon}:{type(exc).__name__}"
    return f"{ENGINE_NAME}|f0={EVERETT_F0_MEDIAN_HZ}|{ident}"


def local_text_to_speech(
    text: str,
    reference_audio_path: Optional[str] = None,
    output_path: Optional[str] = None,
    emotion_params: Optional[Dict] = None,
    lang: str = "en",
) -> str:
    """
    Generate a speech WAV from text through Christman-Sound's own renderer.
    Raises `SynthesisUnavailable` with the reason when no voice can be made.
    """
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    if not output_path:
        audio_dir = _static_dir("audio")
        os.makedirs(audio_dir, mode=0o700, exist_ok=True)
        # emotion_params is part of the key. 2026-10-11: it was not, so a
        # cached neutral rendering was served for an emotional request.
        emotion_key = json.dumps(emotion_params or {}, sort_keys=True)
        content_hash = _hash_content(text, str(reference_audio_path), lang, emotion_key, _voice_tag())
        filename = f"{content_hash}.wav"
        output_path = os.path.join(audio_dir, filename)

        if os.path.exists(output_path):
            return output_path

    engine = get_speech_synthesis_engine(reference_audio=reference_audio_path)
    result_path = engine.generate_speech_audio(
        text=text,
        emotion_params=emotion_params,
        language=lang,
        play_audio=False,
        output_path=output_path
    )
    if not result_path:
        raise SynthesisUnavailable("the renderer returned no file", where="local_text_to_speech")

    return result_path

app = Flask(__name__)

@app.route("/")
def index():
    return send_file("static/index.html")

@app.route("/generate", methods=["POST"])
def generate_speech():
    """Generates speech strictly locally."""
    data = request.json or {}
    text = data.get("text", "")
    reference_audio = data.get("reference_audio_path")
    emotion_params = data.get("emotion_params")
    lang = data.get("lang", "en")

    if not text:
        return jsonify({"error": "text is required"}), 400

    try:
        audio_path = local_text_to_speech(
            text=text,
            reference_audio_path=reference_audio,
            emotion_params=emotion_params,
            lang=lang
        )
        return jsonify({"audio_url": f"/audio/{os.path.basename(audio_path)}", "status": "ok"})
    except SynthesisUnavailable as e:
        logger.error(f"No voice: {e}")
        return jsonify({"error": str(e), "available": False, "reason": e.reason,
                        "status": "unavailable"}), 503
    except (LexiconUnavailable, LexiconFormatError) as e:
        logger.error(f"No lexicon: {e}")
        return jsonify({"error": str(e), "available": False, "reason": str(e),
                        "status": "unavailable"}), 503
    except (LexiconError, NothingToSay) as e:
        # The text cannot be voiced as typed (a word not in the lexicon, a
        # non-speech phone, no word at all). Named, never spelled or skipped.
        logger.error(f"Text not voiced: {e}")
        body = {"error": str(e), "available": False, "reason": str(e), "status": "not_voiced",
                "kind": type(e).__name__}
        if hasattr(e, "words"):
            body["words"] = list(e.words)
        return jsonify(body), 422
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route("/audio/<filename>")
def serve_audio(filename: str):
    # Rule 12: only a bare filename inside the audio directory is served.
    safe = secure_filename(filename)
    if not safe or safe != filename or not safe.endswith(".wav"):
        return jsonify({"error": "invalid filename"}), 400
    return send_from_directory(_static_dir("audio"), safe, mimetype="audio/wav")

@app.route("/status", methods=["GET"])
def status():
    """Health check: the service is up; whether it can make a voice is reported, not assumed."""
    ready, reason = synthesis_readiness()
    return jsonify({
        "status": "online",
        "cloud_dependencies": False,
        "synthesis_available": bool(ready),
        "synthesis_reason": None if ready else reason,
        "last_update": datetime.now().isoformat(),
    })

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT)
