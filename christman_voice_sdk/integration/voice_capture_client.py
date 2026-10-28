"""
voice_capture_client.py
The Christman AI Project — Luma Cognify AI
Voice Frequency Capture + Stealth TTS Relay

WHAT CHANGED AND WHY (2026-10-28) — MEASURED AT THE RECORDING'S OWN RATE
------------------------------------------------------------------------
Lane 2 of ADR-001. sdk-review A, finding 7, measured on AlphaVox's phrases:

1. `SAMPLE_RATE = 44100` was used for every input. AlphaVox's phrases are
   24 kHz, so F0 read 1.3 to 1.9 times too high (44100 / 24000 = 1.84; "i am
   angry" read 207.0 Hz, christman_yin reads it near 136). Now
   `extract_frequency_signature(audio, sample_rate)` takes the rate, with no
   default, and `extract_frequency_signature_from_file(path)` reads it from
   the WAV header. 44100 is only the rate `capture_audio` records at
   (CAPTURE_SAMPLE_RATE), and `run_capture` passes it.

2. The pitch was an autocorrelation peak whose lag was one sample short
   (`argmax + min_lag` over a search that began at lag 1). That tracker is
   gone: F0 is christman_yin through christman_sound.dsp.get_dsp(), the one
   loader, via tone/voice_measure.py (Everett: "We have our own DSP for that
   exact reason"). No library, no pitch: PitchUnavailable is raised.

3. The "formants" were the strongest FFT bin in 200-1000 Hz and in
   700-3000 Hz. The bands overlap, and F1 equalled F2 on 5 runs. They are
   harmonics, not resonances. Now F1, F2 and F3 are LPC formants from
   christman_sound.formants.measure_formants (christman_lpc per 25 ms frame,
   roots on the real axis dropped, so no 0 Hz formant; bandwidth and range
   rules; voiced frames only; medians), the same measurement that wrote
   AlphaVox's alphavox_voice.json. A recording shorter than one frame, or
   with no voiced frame, gives None and `formant_reason` says why.

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

# websockets is needed only to reach Derek (speak_via_derek). It is imported
# there, so measuring a voice does not depend on it (2026-10-28: the module
# failed to import without it, and two tests with it).

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

from christman_sound.formants import AudioTooShortError, measure_formants

from ..audio.config import get_config, require_local_or_tls
from ..engines.synthesis_result import SOVEREIGN_NOT_WIRED, SynthesisUnavailable
from ..tone.voice_measure import pitch_contour, read_wav
from ..utils.logger import get_logger

logger = get_logger(__name__)

# ── Config from SDK ───────────────────────────────────────────────────────────
config = get_config()
DEREK_WS_URI = require_local_or_tls(
    config.get("derek.ws_uri", "ws://localhost:8000/ws/derek"), what="derek.ws_uri"
)
PROFILE_DIR = Path.home() / ".christman_ai" / "voice_profiles"
#: The rate capture_audio records at. Never assumed for any other audio.
CAPTURE_SAMPLE_RATE = 44100
CAPTURE_DURATION = 8
DEFAULT_PROFILE = "default"

#: LPC formant ceiling. A chosen value: christman_sound.formants' default,
#: the one AlphaVox's alphavox_voice.json was measured with.
FORMANT_CEILING_HZ = 5000.0


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
        int(duration * CAPTURE_SAMPLE_RATE),
        samplerate=CAPTURE_SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    logger.info("Capture complete.")
    return audio.flatten()


def extract_frequency_signature(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Extract a voice frequency signature from mono samples at `sample_rate`.

    The rate is required: it is the rate the samples were recorded at, never
    assumed. Every value is either measured from `audio` or None.
    `voiced_frames` and `total_frames` say what the pitch numbers stand on
    (christman_yin, 10 ms frames); `formant_frames` what the formants stand
    on; `measured` lists which fields hold a reading. Raises PitchUnavailable
    when christman_dsp is not built (nothing is measured without it).
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size:
        audio = audio - np.mean(audio)  # Remove DC offset
    track = pitch_contour(audio, sample_rate)   # validates the rate; raises if no library
    sr = track.sample_rate
    if audio.size == 0:
        return _unmeasured_signature(sr)

    fundamental = track.median_hz
    p10, p90 = track.percentile_hz(10), track.percentile_hz(90)
    pitch_range: Optional[list] = [p10, p90] if fundamental is not None else None

    # Spectral analysis
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    fft_mag = np.abs(np.fft.rfft(audio))
    total_mag = float(fft_mag.sum())
    silent = total_mag <= 0.0
    spectral_centroid = None if silent else float(np.sum(freqs * fft_mag) / total_mag)

    # Formants: LPC, voiced frames only (christman_sound.formants).
    f1 = f2 = f3 = None
    formant_frames: Dict[str, int] = {"f1": 0, "f2": 0, "f3": 0}
    formant_reason: Optional[str] = None
    try:
        summary = measure_formants(audio.tolist(), sr, max_formant_hz=FORMANT_CEILING_HZ).summary
    except AudioTooShortError as exc:
        summary = None
        formant_reason = f"too short for one LPC frame: {exc}"
    if summary is not None:
        f1, f2, f3 = summary.f1_median_hz, summary.f2_median_hz, summary.f3_median_hz
        formant_frames = {"f1": summary.n_f1, "f2": summary.n_f2, "f3": summary.n_f3}
        if f1 is None:
            formant_reason = (
                f"no voiced frame gave formants ({summary.n_voiced} voiced of {summary.n_frames})"
            )

    # Zero Crossing Rate
    zcr = float(np.mean(np.diff(np.signbit(audio).astype(int)) != 0)) if audio.size > 1 else None

    # Speaking rate approximation (energy envelope peaks per second)
    energy = audio ** 2
    kernel_len = max(1, int(0.020 * sr))
    kernel = np.ones(kernel_len) / kernel_len
    env = np.convolve(energy, kernel, mode="same")
    if silent or float(np.mean(env)) <= 0.0:
        speaking_rate_norm = None
    else:
        peaks, _ = signal.find_peaks(env, height=np.mean(env) * 0.5,
                                     distance=max(1, int(0.05 * sr)))
        speaking_rate_norm = float(len(peaks) / (len(audio) / sr))

    sig: Dict[str, Any] = {
        "fundamental_hz": fundamental,
        "pitch_range_hz": pitch_range,
        "formant_f1_hz": f1,
        "formant_f2_hz": f2,
        "formant_f3_hz": f3,
        "speaking_rate_norm": speaking_rate_norm,
        "spectral_centroid": spectral_centroid,
        "zcr_mean": zcr,
        "voiced_frames": track.voiced_frames,
        "total_frames": track.total_frames,
        "formant_frames": formant_frames,
        "formant_reason": formant_reason,
        "duration_seconds": float(len(audio) / sr),
        "sample_rate": sr,
        "pitch_tracker": "christman_yin (christman_sound.dsp)",
        "formant_method": f"LPC, christman_sound.formants, ceiling {FORMANT_CEILING_HZ:.0f} Hz",
    }
    _not_readings = {"measured", "voiced_frames", "total_frames", "formant_frames", "formant_reason",
                     "sample_rate", "pitch_tracker", "formant_method", "duration_seconds"}
    sig["measured"] = [k for k, v in sig.items() if v is not None and k not in _not_readings]
    if fundamental is None:
        logger.warning(
            "No voiced frames in %d frames — fundamental_hz and pitch_range_hz "
            "are None. This is not a 150 Hz voice; it is no reading.",
            track.total_frames,
        )
    return sig


def extract_frequency_signature_from_file(path: str) -> Dict[str, Any]:
    """The signature of a WAV file, measured at the rate its header gives."""
    audio, sample_rate = read_wav(path)
    return extract_frequency_signature(audio, sample_rate)


def _unmeasured_signature(sample_rate: int) -> Dict[str, Any]:
    return {
        "fundamental_hz": None, "pitch_range_hz": None,
        "formant_f1_hz": None, "formant_f2_hz": None, "formant_f3_hz": None,
        "speaking_rate_norm": None, "spectral_centroid": None, "zcr_mean": None,
        "voiced_frames": 0, "total_frames": 0,
        "formant_frames": {"f1": 0, "f2": 0, "f3": 0},
        "formant_reason": "no samples",
        "duration_seconds": 0.0, "sample_rate": sample_rate,
        "pitch_tracker": "christman_yin (christman_sound.dsp)",
        "formant_method": f"LPC, christman_sound.formants, ceiling {FORMANT_CEILING_HZ:.0f} Hz",
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
        import websockets  # pyright: ignore[reportMissingImports]

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
    signature = extract_frequency_signature(audio, CAPTURE_SAMPLE_RATE)
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
