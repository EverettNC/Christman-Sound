"""
TTS Service - Local API

WHAT CHANGED AND WHY
--------------------
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
from .voice_synthesis import get_speech_synthesis_engine

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

def local_text_to_speech(
    text: str,
    reference_audio_path: Optional[str] = None,
    output_path: Optional[str] = None,
    emotion_params: Optional[Dict] = None,
    lang: str = "en",
) -> str:
    """Generate speech WAV from text using local XTTS zero-shot cloning."""
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    if not output_path:
        audio_dir = _static_dir("audio")
        os.makedirs(audio_dir, mode=0o700, exist_ok=True)
        # emotion_params is part of the key. 2026-10-11: it was not, so a
        # cached neutral rendering was served for an emotional request.
        emotion_key = json.dumps(emotion_params or {}, sort_keys=True)
        content_hash = _hash_content(text, str(reference_audio_path), lang, emotion_key)
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
        raise RuntimeError("Local synthesis failed and returned None.")

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
    """Health check: Local TTS service availability."""
    return jsonify({
        "status": "online",
        "cloud_dependencies": False,
        "engine": "xtts_v2_local",
        "last_update": datetime.now().isoformat(),
    })

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT)
