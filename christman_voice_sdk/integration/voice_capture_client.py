"""
voice_capture_client.py
The Christman AI Project — Luma Cognify AI
Voice frequency capture, and speaking through the package's own mouth.

MEASURED AT THE RECORDING'S OWN RATE
------------------------------------
`extract_frequency_signature(audio, sample_rate)` takes the rate, with no
default: it is the rate the samples were recorded at, never assumed (a 24 kHz
recording read as 44.1 kHz comes out 1.84 times too high).
`extract_frequency_signature_from_file(path)` reads the rate from the WAV
header. 44100 is only the rate `capture_audio` records at
(CAPTURE_SAMPLE_RATE), and `run_capture` passes it.

  - F0 is christman_yin through christman_sound.dsp.get_dsp(), the one
    loader, via tone/voice_measure.py (Everett: "We have our own DSP for that
    exact reason"). No library, no pitch: PitchUnavailable is raised.
  - F1, F2 and F3 are LPC formants from christman_sound.formants.
    measure_formants (christman_lpc per 25 ms frame, roots on the real axis
    dropped, so no 0 Hz formant; bandwidth and range rules; voiced frames
    only; medians) — the same measurement scripts/measure_voice.py writes a
    voice profile with. A recording shorter than one frame, or with no voiced
    frame, gives None and `formant_reason` says why.
  - A value that was not measured is None, never a stand-in (silence once
    came back as a complete "150 Hz" profile). The signature says how many
    frames were voiced, so the reader knows what each number stands on.
  - The profile directory is created when a profile is saved, never at
    import.

SPEAKING
--------
`speak(text, voice)` speaks through the package's own mouth
(christman_voice_sdk.synthesis.voice_synthesis, rendering through
christman_sound.synth) in the caller's voice: a voice profile file, a frame
voice folder, or a loaded VoiceProfile or FrameVoice. There is no default
voice. It plays, or raises the typed failure with the reason (NothingVoiced,
PlaybackFailed, or the lexicon's and synthesizer's own refusals). No outside
speech engine (no `say`, espeak, pyttsx3, browser speech or cloud engine)
and no relay to any other process.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import scipy.signal as signal

import argparse

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

from ..tone.voice_measure import pitch_contour, read_wav
from ..utils.logger import get_logger

logger = get_logger(__name__)

PROFILE_DIR = Path.home() / ".christman_ai" / "voice_profiles"
#: The rate capture_audio records at. Never assumed for any other audio.
CAPTURE_SAMPLE_RATE = 44100
CAPTURE_DURATION = 8

#: LPC formant ceiling. A chosen value: christman_sound.formants' default,
#: the one scripts/measure_voice.py measures a voice profile with.
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
# SECTION 3 — SPEAKING (the package's own mouth, the caller's voice)
# ══════════════════════════════════════════════════════════════════════════════

def speak(text: str, voice: Any, lexicon_path: Optional[str] = None) -> str:
    """
    Speak `text` in the caller's `voice` through the package's own mouth and
    play it. `voice` is required: a voice profile file, a frame voice folder,
    or a loaded VoiceProfile or FrameVoice. Returns the WAV path once it has
    played. Raises NothingVoiced (no voice, or nothing could be made),
    PlaybackFailed, or the lexicon's and synthesizer's own typed failures;
    never speaks in another voice and never returns as if it had spoken.
    """
    from ..synthesis.voice_synthesis import NothingVoiced, SpeechSynthesisEngine

    if voice is None:
        raise NothingVoiced("no voice given: pass the caller's voice (a voice profile file or frame voice folder)",
                            where="voice_capture_client.speak")
    mouth = SpeechSynthesisEngine(lexicon_path=lexicon_path, voice=voice)
    path = mouth.generate_speech_audio(text, play_audio=True)
    if path is None:
        raise NothingVoiced("no text to speak", where="voice_capture_client.speak")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def run_capture(name: str):
    """Capture and save a voice profile."""
    logger.info(f"Capturing profile: {name}")
    audio = capture_audio()
    signature = extract_frequency_signature(audio, CAPTURE_SAMPLE_RATE)
    save_profile(name, signature)


def main():
    parser = argparse.ArgumentParser(description="Christman voice capture, and speaking in the caller's voice")
    parser.add_argument("--capture", metavar="NAME", help="Capture new profile")
    parser.add_argument("--speak", metavar="TEXT", help="Speak text (needs --voice)")
    parser.add_argument("--voice", metavar="PATH", help="the voice to speak in: a voice profile file or frame voice folder")
    parser.add_argument("--lexicon", metavar="PATH", default=None, help="align_lexicon.txt (else $CHRISTMAN_LEXICON_PATH)")
    parser.add_argument("--list", action="store_true", help="List profiles")

    args = parser.parse_args()

    if args.list:
        profiles = list_profiles()
        print("Saved profiles:", profiles if profiles else "None yet")
    elif args.capture:
        run_capture(args.capture)
    elif args.speak:
        if not args.voice:
            parser.error("--speak needs --voice: there is no default voice")
        print(speak(args.speak, args.voice, args.lexicon))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
