"""
TTS Service - Local API

The service speaks in the voice it is started with: the caller's voice
profile file or frame voice folder, passed as an argument
(`python -m christman_voice_sdk.synthesis.tts_service --voice <path>`, or
`configure_voice(<path or loaded voice>)` before serving, or `voice=` to
local_text_to_speech). There is no default voice. With none, /generate
answers 503 with the reason and /status reports synthesis unavailable.

Rendering is voice_synthesis (christman_sound.synth, Christman-Sound's own
renderer); no outside speech engine and no cloud service (gTTS was removed:
it shipped the text of the person speaking to Google).

The content-hash cache key carries the voice that made the file: the engine
name, the fingerprint of the caller's voice (the SHA-256 of the passed
profile file; christman_sound.synth.voice_fingerprint), and the lexicon
file's identity. So audio made in another voice, in an earlier measurement
of this one, from another lexicon, or by any earlier renderer is never served
as this voice. The emotion parameters are part of the key too (2026-10-11:
they were not, so a cached neutral rendering was served for an emotional
request).

The lexicon's and the synthesizer's typed failures reach /generate unchanged
and are answered with `available: false` and the reason: 503 when there is
no voice or no lexicon (SynthesisUnavailable, LexiconUnavailable,
LexiconFormatError), 422 when the text itself cannot be voiced
(WordNotInLexicon with the missing `words`, PhoneNotMapped, NothingToSay).
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
    LexiconError,
    NothingToSay,
    get_speech_synthesis_engine,
    resolve_voice,
    synthesis_readiness,
    voice_fingerprint,
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

#: The voice this service speaks in, set by configure_voice (or --voice).
_service_voice = None


def configure_voice(voice) -> None:
    """Set the voice the service speaks in: a voice profile file, a frame voice folder, or a loaded voice."""
    global _service_voice
    _service_voice = voice


def _voice_tag(voice) -> str:
    """
    Which voice makes a file: the engine, the fingerprint of the caller's
    voice (the passed profile's hash), and the lexicon file's identity (path,
    size, mtime). Part of the cache key so a file made by any other voice, by
    an earlier measurement of this one, or from another lexicon, is never
    served as this one. `voice` is the loaded voice (resolve_voice).
    """
    lexicon = os.environ.get(LEXICON_ENV, "")
    ident = "no-lexicon"
    if lexicon:
        try:
            st = os.stat(lexicon)
            ident = f"{os.path.realpath(lexicon)}:{st.st_size}:{st.st_mtime_ns}"
        except OSError as exc:
            ident = f"unreadable:{lexicon}:{type(exc).__name__}"
    return f"{ENGINE_NAME}|voice={voice_fingerprint(voice)}|{ident}"


def local_text_to_speech(
    text: str,
    reference_audio_path: Optional[str] = None,
    output_path: Optional[str] = None,
    emotion_params: Optional[Dict] = None,
    lang: str = "en",
    *,
    voice,
) -> str:
    """
    Generate a speech WAV from text through Christman-Sound's own renderer,
    in the caller's `voice` (required: a voice profile file, a frame voice
    folder, or a loaded voice). Raises `SynthesisUnavailable` with the reason
    when there is no voice or none can be made.
    """
    if not text or not text.strip():
        raise ValueError("text must not be empty")
    loaded = resolve_voice(voice)

    if not output_path:
        audio_dir = _static_dir("audio")
        os.makedirs(audio_dir, mode=0o700, exist_ok=True)
        # emotion_params is part of the key. 2026-10-11: it was not, so a
        # cached neutral rendering was served for an emotional request.
        emotion_key = json.dumps(emotion_params or {}, sort_keys=True)
        content_hash = _hash_content(text, str(reference_audio_path), lang, emotion_key, _voice_tag(loaded))
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
        output_path=output_path,
        voice=loaded,
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
            lang=lang,
            voice=_service_voice,
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
    # Readiness is checked for the voice this service was started with.
    ready, reason = synthesis_readiness(None, _service_voice)
    return jsonify({
        "status": "online",
        "cloud_dependencies": False,
        "synthesis_available": bool(ready),
        "synthesis_reason": None if ready else reason,
        "last_update": datetime.now().isoformat(),
    })

def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Christman-Sound local speech service")
    ap.add_argument("--voice", required=True,
                    help="the voice to speak in: a voice profile file or a frame voice folder")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    configure_voice(args.voice)
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT)


if __name__ == "__main__":
    main()
