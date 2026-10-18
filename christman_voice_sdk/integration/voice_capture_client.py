"""
voice_capture_client.py
The Christman AI Project — Luma Cognify AI
Voice Frequency Capture + Stealth TTS Relay

WHAT CHANGED AND WHY (2026-10-17) — OUTSIDE VOICES CUT
------------------------------------------------------
Everett's ruling: the AlphaVox rebuild cuts every outside speech engine. The
local fallback spoke in the operating system's voice. Removed from
`local_speak_fallback` (quoted from main 99a6cce):

    if sys.platform == "darwin":
        cmd = ["say", "-v", "Samantha"]
    elif sys.platform.startswith("linux"):
        cmd = ["espeak", "--stdin", "-s", "150"]
    ...
            proc = subprocess.run(cmd, input=data, check=False,
                                  stderr=subprocess.DEVNULL, timeout=120)
    ...
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

`local_speak_fallback` keeps its name and raises
`SynthesisUnavailable("sovereign synthesizer not wired yet")`. When Derek is
offline, `run_speak` now ends in that typed failure instead of a stranger's
voice. The Derek relay (`speak_via_derek`) — a Christman being, not an
outside engine — and voice-profile capture are unchanged. `import
subprocess` and `import sys` went with the fallback; nothing else used them.

EARLIER HISTORY (2026-10-11, every line measured before it was changed)
----------------------------------------------------------------------

1. THE SAFETY NET HAD A NameError IN IT.

       os.system(f'say -v "Samantha" "{text}"')

   `os` was never imported. The local fallback runs at exactly one moment —
   when Derek is already unreachable — and at that moment it raised
   NameError. It was also a shell string with the text interpolated into it.
   The text now goes to `say` on stdin; nothing is parsed as a command.

2. IT MEASURED NOTHING AND REPORTED 150 Hz.

       else:
           fundamental = 150.0
           pitch_min = 100.0
           pitch_max = 250.0
       ...
       spectral_centroid = ... if fft_mag.sum() > 0 else 1500.0
       ...
       if mask.sum() == 0:
           return (low_hz + high_hz) / 2

   Silence, or a recording with no voiced frames, came back as a complete
   voice profile: F0 150 Hz, range 100–250, centroid 1500, formants at the
   band midpoints. Saved under the person's name. Sent to Derek as their
   voice. A value that was not measured is None (HANDOFF rule), and the
   profile now says how many frames were voiced so the reader knows what the
   number stands on.

3. `speak_via_derek` filled the same gaps again on the way out:

       "pitch_hz": signature.get("fundamental_hz", 150)

   None now travels as null. Derek decides what to do without a reading;
   this file does not invent one for it.

4. `PROFILE_DIR.mkdir(...)` ran at import. Importing a module must not write
   to the home directory. It runs when a profile is saved.

5. The Derek socket default is plaintext `ws://` to a config-overridable
   host. Rule 12: accepted only to loopback; anything else must be `wss://`.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import scipy.signal as signal

import argparse
import websockets  # pyright: ignore[reportMissingImports]

# sounddevice needs the PortAudio system library. Only capture_audio() needs
# sounddevice. Analysis, profiles and speaking must not die at import on a
# machine with no microphone stack, so it is imported where it is used and
# its absence is reported there, loudly.
try:
    import sounddevice as sd  # pyright: ignore[reportMissingImports]
    _SOUNDDEVICE_ERROR: Optional[str] = None
except (ImportError, OSError) as _exc:   # OSError: "PortAudio library not found"
    sd = None  # type: ignore[assignment]
    _SOUNDDEVICE_ERROR = f"{type(_exc).__name__}: {_exc}"

from ..audio.config import get_config, require_local_or_tls
from ..engines.synthesis_result import SOVEREIGN_NOT_WIRED, SynthesisUnavailable
from ..utils.logger import get_logger

logger = get_logger(__name__)

# ── Config from SDK ───────────────────────────────────────────────────────────
config = get_config()
DEREK_WS_URI = require_local_or_tls(
    config.get("derek.ws_uri", "ws://localhost:8000/ws/derek"), what="derek.ws_uri"
)
PROFILE_DIR = Path.home() / ".christman_ai" / "voice_profiles"
SAMPLE_RATE = 44100
CAPTURE_DURATION = 8
DEFAULT_PROFILE = "default"

#: Autocorrelation peak below this is treated as unvoiced. A threshold, not a
#: measurement — named so it can be argued with.
VOICING_THRESHOLD = 0.1


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — VOICE FREQUENCY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def capture_audio(duration: int = CAPTURE_DURATION) -> np.ndarray:
    """Record microphone input. Raises RuntimeError if there is no capture stack."""
    if sd is None:
        raise RuntimeError(
            "capture_audio() needs sounddevice and the PortAudio library, which "
            f"did not load here ({_SOUNDDEVICE_ERROR}). Nothing was recorded."
        )
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


def extract_frequency_signature(audio: np.ndarray) -> Dict[str, Any]:
    """
    Extract a voice frequency signature.

    Every value is either measured from `audio` or None. `voiced_frames` and
    `total_frames` say what the pitch numbers stand on; `measured` lists which
    fields hold a reading.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return _unmeasured_signature(0, 0)

    audio = audio - np.mean(audio)  # Remove DC offset

    # Fundamental frequency (F0) — autocorrelation per 30 ms frame
    frame_len = int(0.030 * SAMPLE_RATE)
    hop_len = int(0.010 * SAMPLE_RATE)
    f0_values = []
    total_frames = 0

    for start in range(0, len(audio) - frame_len, hop_len):
        total_frames += 1
        frame = audio[start:start + frame_len]
        corr = np.correlate(frame, frame, mode="full")[len(frame):]
        min_lag = int(SAMPLE_RATE / 500)
        max_lag = int(SAMPLE_RATE / 80)
        corr_search = corr[min_lag:max_lag]

        if len(corr_search) > 0 and corr_search.max() > VOICING_THRESHOLD:
            peak_lag = np.argmax(corr_search) + min_lag
            f0_values.append(SAMPLE_RATE / peak_lag)

    if f0_values:
        f0_arr = np.array(f0_values)
        fundamental: Optional[float] = float(np.median(f0_arr))
        pitch_range: Optional[list] = [float(np.percentile(f0_arr, 10)),
                                       float(np.percentile(f0_arr, 90))]
    else:
        fundamental = None
        pitch_range = None

    # Spectral analysis
    freqs = np.fft.rfftfreq(len(audio), 1.0 / SAMPLE_RATE)
    fft_mag = np.abs(np.fft.rfft(audio))
    total_mag = float(fft_mag.sum())
    silent = total_mag <= 0.0

    spectral_centroid = None if silent else float(np.sum(freqs * fft_mag) / total_mag)

    def find_formant(low_hz: int, high_hz: int) -> Optional[float]:
        if silent:
            return None
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if mask.sum() == 0:
            return None
        return float(freqs[mask][np.argmax(fft_mag[mask])])

    formant_f1 = find_formant(200, 1000)
    formant_f2 = find_formant(700, 3000)

    # Zero Crossing Rate
    zcr = float(np.mean(np.diff(np.signbit(audio).astype(int)) != 0)) if audio.size > 1 else None

    # Speaking rate approximation (energy envelope peaks per second)
    energy = audio ** 2
    kernel_len = max(1, int(0.020 * SAMPLE_RATE))
    kernel = np.ones(kernel_len) / kernel_len
    env = np.convolve(energy, kernel, mode="same")
    if silent or float(np.mean(env)) <= 0.0:
        speaking_rate_norm = None
    else:
        peaks, _ = signal.find_peaks(env, height=np.mean(env) * 0.5,
                                     distance=int(0.05 * SAMPLE_RATE))
        speaking_rate_norm = float(len(peaks) / (len(audio) / SAMPLE_RATE))

    sig: Dict[str, Any] = {
        "fundamental_hz": fundamental,
        "pitch_range_hz": pitch_range,
        "formant_f1_hz": formant_f1,
        "formant_f2_hz": formant_f2,
        "speaking_rate_norm": speaking_rate_norm,
        "spectral_centroid": spectral_centroid,
        "zcr_mean": zcr,
        "voiced_frames": len(f0_values),
        "total_frames": total_frames,
        "duration_seconds": float(len(audio) / SAMPLE_RATE),
    }
    sig["measured"] = [k for k, v in sig.items() if v is not None and k != "measured"]
    if fundamental is None:
        logger.warning(
            "No voiced frames in %d frames — fundamental_hz and pitch_range_hz "
            "are None. This is not a 150 Hz voice; it is no reading.",
            total_frames,
        )
    return sig


def _unmeasured_signature(voiced: int, total: int) -> Dict[str, Any]:
    return {
        "fundamental_hz": None, "pitch_range_hz": None,
        "formant_f1_hz": None, "formant_f2_hz": None,
        "speaking_rate_norm": None, "spectral_centroid": None, "zcr_mean": None,
        "voiced_frames": voiced, "total_frames": total, "duration_seconds": 0.0,
        "measured": [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PROFILE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def save_profile(name: str, signature: Dict) -> Path:
    """Save voice profile. Creates the profile directory on first use (0700)."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
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
    if not PROFILE_DIR.exists():
        return []
    return [p.stem for p in PROFILE_DIR.glob("*.json")]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SPEAKING (Derek relay + fallback)
# ══════════════════════════════════════════════════════════════════════════════

async def speak_via_derek(text: str, signature: Optional[Dict] = None) -> str:
    """Send TTS request to Derek. Unmeasured profile fields travel as null."""
    try:
        async with websockets.connect(DEREK_WS_URI, open_timeout=5) as ws:
            payload: Dict[str, Any] = {"text": text, "mode": "stealth"}
            if signature:
                payload["voice_profile"] = {
                    "pitch_hz": signature.get("fundamental_hz"),
                    "pitch_range": signature.get("pitch_range_hz"),
                    "formant_f1": signature.get("formant_f1_hz"),
                    "formant_f2": signature.get("formant_f2_hz"),
                    "voiced_frames": signature.get("voiced_frames"),
                }
            await ws.send(json.dumps({"command": "tts", "payload": payload}))
            return await ws.recv()
    except Exception as e:
        logger.warning(f"Derek unreachable: {e}")
        return '{"status": "offline"}'


def local_speak_fallback(text: str, signature: Optional[Dict] = None) -> bool:
    """
    There is no local voice until the sovereign synthesizer is wired. This
    raises `SynthesisUnavailable` with the reason; it never speaks in an
    outside voice and never returns as if it had spoken.
    """
    del signature
    err = SynthesisUnavailable(SOVEREIGN_NOT_WIRED, where="voice_capture_client.local_speak_fallback")
    logger.error(f"Text was NOT voiced ({(text or '')[:40]!r}): {err}")
    raise err


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
