"""
voice_capture_client.py
The Christman AI Project — Luma Cognify AI
Voice Frequency Capture + Stealth TTS Relay
"""

from __future__ import annotations

import asyncio
import json
import time
import threading
import sys
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import scipy.signal as signal

import argparse
import websockets  # pyright: ignore[reportMissingImports]

from ..audio.config import get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)

# ── Config from SDK ───────────────────────────────────────────────────────────
config = get_config()
DEREK_WS_URI = config.get("derek.ws_uri", "ws://localhost:8000/ws/derek")
PROFILE_DIR = Path.home() / ".christman_ai" / "voice_profiles"
SAMPLE_RATE = 44100
CAPTURE_DURATION = 8
DEFAULT_PROFILE = "default"

PROFILE_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — VOICE FREQUENCY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def capture_audio(duration: int = CAPTURE_DURATION) -> np.ndarray:
    """Record microphone input."""
    logger.info(f"Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    logger.info("Capture complete.")
    return audio.flatten()


def extract_frequency_signature(audio: np.ndarray) -> Dict:
    """Extract real voice frequency signature."""
    audio = audio - np.mean(audio)  # Remove DC offset

    # Fundamental frequency (F0)
    frame_len = int(0.030 * SAMPLE_RATE)
    hop_len = int(0.010 * SAMPLE_RATE)
    f0_values = []

    for start in range(0, len(audio) - frame_len, hop_len):
        frame = audio[start:start + frame_len]
        corr = np.correlate(frame, frame, mode="full")[len(frame):]
        min_lag = int(SAMPLE_RATE / 500)
        max_lag = int(SAMPLE_RATE / 80)
        corr_search = corr[min_lag:max_lag]

        if len(corr_search) > 0 and corr_search.max() > 0.1:
            peak_lag = np.argmax(corr_search) + min_lag
            f0 = SAMPLE_RATE / peak_lag
            f0_values.append(f0)

    if f0_values:
        f0_values = np.array(f0_values)
        fundamental = float(np.median(f0_values))
        pitch_min = float(np.percentile(f0_values, 10))
        pitch_max = float(np.percentile(f0_values, 90))
    else:
        fundamental = 150.0
        pitch_min = 100.0
        pitch_max = 250.0

    # Spectral analysis
    freqs = np.fft.rfftfreq(len(audio), 1.0 / SAMPLE_RATE)
    fft_mag = np.abs(np.fft.rfft(audio))

    spectral_centroid = float(np.sum(freqs * fft_mag) / fft_mag.sum()) if fft_mag.sum() > 0 else 1500.0

    # Simple formant estimation
    def find_formant(low_hz: int, high_hz: int) -> float:
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if mask.sum() == 0:
            return (low_hz + high_hz) / 2
        return float(freqs[mask][np.argmax(fft_mag[mask])])

    formant_f1 = find_formant(200, 1000)
    formant_f2 = find_formant(700, 3000)

    # Zero Crossing Rate
    zcr = float(np.mean(np.diff(np.signbit(audio).astype(int)) != 0))

    # Speaking rate approximation
    energy = audio ** 2
    kernel = np.ones(int(0.020 * SAMPLE_RATE)) / int(0.020 * SAMPLE_RATE)
    env = np.convolve(energy, kernel, mode="same")
    peaks, _ = signal.find_peaks(env, height=np.mean(env) * 0.5, distance=int(0.05 * SAMPLE_RATE))
    speaking_rate_norm = float(len(peaks) / (len(audio) / SAMPLE_RATE))

    return {
        "fundamental_hz": fundamental,
        "pitch_range_hz": [pitch_min, pitch_max],
        "formant_f1_hz": formant_f1,
        "formant_f2_hz": formant_f2,
        "speaking_rate_norm": speaking_rate_norm,
        "spectral_centroid": spectral_centroid,
        "zcr_mean": zcr,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PROFILE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def save_profile(name: str, signature: Dict) -> Path:
    """Save voice profile."""
    profile_path = PROFILE_DIR / f"{name}.json"
    data = {
        "name": name,
        "captured": time.strftime("%Y-%m-%d %H:%M:%S"),
        "signature": signature,
    }
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Profile saved: {profile_path}")
    return profile_path


def load_profile(name: str) -> Dict:
    """Load voice profile."""
    profile_path = PROFILE_DIR / f"{name}.json"
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile '{name}' not found. Run --capture first.")
    with open(profile_path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded profile: {data['name']}")
    return data["signature"]


def list_profiles() -> list[str]:
    """List saved profiles."""
    return [p.stem for p in PROFILE_DIR.glob("*.json")]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SPEAKING (Derek relay + fallback)
# ══════════════════════════════════════════════════════════════════════════════

async def speak_via_derek(text: str, signature: Optional[Dict] = None) -> str:
    """Send TTS request to Derek."""
    try:
        async with websockets.connect(DEREK_WS_URI, open_timeout=5) as ws:
            payload = {"text": text, "mode": "stealth"}
            if signature:
                payload["voice_profile"] = {
                    "pitch_hz": signature.get("fundamental_hz", 150),
                    "pitch_range": signature.get("pitch_range_hz", [100, 250]),
                    "formant_f1": signature.get("formant_f1_hz"),
                    "formant_f2": signature.get("formant_f2_hz"),
                }
            await ws.send(json.dumps({"command": "tts", "payload": payload}))
            return await ws.recv()
    except Exception as e:
        logger.warning(f"Derek unreachable: {e}")
        return '{"status": "offline"}'


def local_speak_fallback(text: str, signature: Optional[Dict] = None):
    """Local TTS fallback."""
    pitch = signature.get("fundamental_hz", 150) if signature else 150
    if sys.platform == "darwin":
        os.system(f'say -v "Samantha" "{text}"')
    else:
        logger.warning("Local TTS fallback not fully implemented on this platform.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def run_capture(name: str):
    """Capture and save a voice profile."""
    logger.info(f"Capturing profile: {name}")
    audio = capture_audio()
    signature = extract_frequency_signature(audio)
    save_profile(name, signature)


async def run_speak(text: str, profile_name: str):
    """Speak using profile."""
    try:
        signature = load_profile(profile_name)
    except FileNotFoundError:
        signature = None
    response = await speak_via_derek(text, signature)
    if "offline" in response:
        local_speak_fallback(text, signature)


def main():
    parser = argparse.ArgumentParser(description="Christman Voice Capture + Stealth TTS")
    parser.add_argument("--capture", metavar="NAME", help="Capture new profile")
    parser.add_argument("--speak", metavar="TEXT", help="Speak text")
    parser.add_argument("--profile", default="default", help="Profile name")
    parser.add_argument("--list", action="store_true", help="List profiles")

    args = parser.parse_args()

    if args.list:
        profiles = list_profiles()
        print("Saved profiles:", profiles if profiles else "None yet")
    elif args.capture:
        run_capture(args.capture)
    elif args.speak:
        asyncio.run(run_speak(args.speak, args.profile))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
