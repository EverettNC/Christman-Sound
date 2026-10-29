# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Speaker baseline: built only from measurements, 30 events, one speaker.
The mouth: the being's own voice in; a local checkpoint or an honest error.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from christman_voice_sdk.audio import mouth as M  # noqa: E402
from christman_voice_sdk.audio.audio_encoder import (  # noqa: E402
    AudioEncoder,
    CheckpointLoadError,
)
from christman_voice_sdk.audio.prosody import (  # noqa: E402
    ProsodyAnalyzer,
    SpeakerBaseline,
    VocalEvent,
    build_speaker_baseline,
)


def _events(n, f0_every=1):
    rng = np.random.default_rng(7)
    out = []
    for i in range(n):
        out.append(VocalEvent(
            duration=0.4 + 0.01 * (i % 5),
            attack=0.02 + 0.001 * (i % 3),
            peak_rms=float(0.1 + 0.01 * rng.standard_normal()),
            median_f0=(float(120 + 5 * rng.standard_normal()) if i % f0_every == 0 else None),
            mean_zcr=0.08,
        ))
    return out


def test_baseline_needs_thirty_events_and_a_speaker():
    with pytest.raises(ValueError, match="needs 30"):
        build_speaker_baseline(_events(29), "speaker-a")
    with pytest.raises(ValueError, match="speaker_id"):
        build_speaker_baseline(_events(40), "  ")


def test_baseline_turns_on_speaker_relative_reading():
    b = build_speaker_baseline(_events(30), "speaker-a")
    assert b.is_populated and b.sample_count == 30
    assert 110 < b.f0_median < 130
    reading = ProsodyAnalyzer(baseline=b).analyze(_events(1)[0])
    assert reading.reference == "speaker_baseline"
    assert reading.deviations and all(d.reference == "speaker_baseline"
                                      for d in reading.deviations)


def test_a_missing_f0_lock_is_not_filled_in():
    # F0 locked on 15 of 30 events: not enough for its own median
    b = build_speaker_baseline(_events(30, f0_every=2), "speaker-a")
    assert b.f0_median is None
    assert b.is_populated is False
    assert b.rms_rest is not None


# -- the mouth -----------------------------------------------------------------

def _sine(freq, sr, seconds=1.0):
    t = np.arange(int(sr * seconds)) / sr
    return 0.5 * np.sin(2 * np.pi * freq * t)


def test_mel_front_end_matches_the_encoders_geometry():
    mel = M.melspectrogram(_sine(440.0, 16000))
    assert mel.shape == (80, 81)                 # 16000 / 200 + 1 centred frames
    assert mel.min() >= -4.0 and mel.max() <= 4.0
    windows = M.mel_windows(mel)
    assert windows.shape[1:] == (80, 16)


def test_a_440_hz_tone_lands_in_the_440_hz_band():
    mel = M.melspectrogram(_sine(440.0, 16000))
    peak = int(np.argmax(mel[:, mel.shape[1] // 2]))
    edges = M._mel_to_hz(np.linspace(M._hz_to_mel(M.FMIN), M._hz_to_mel(M.FMAX), 82))
    assert edges[peak] < 440.0 < edges[peak + 2]


def test_other_rates_are_resampled_and_said_so():
    a = M.to_16k(_sine(440.0, 22050), 22050)
    assert len(a) == 16000
    with pytest.raises(ValueError, match="mono"):
        M.to_16k(np.zeros((2, 100)), 16000)
    with pytest.raises(ValueError, match="float"):
        M.to_16k(np.zeros(100, dtype=np.int16), 16000)


def test_too_short_is_an_error_not_a_padded_guess():
    with pytest.raises(ValueError):
        M.mel_windows(M.melspectrogram(_sine(440.0, 16000, seconds=0.05)))


def test_no_checkpoint_no_mouth(monkeypatch):
    monkeypatch.delenv(M.CHECKPOINT_ENV, raising=False)
    with pytest.raises(CheckpointLoadError, match="no mouth checkpoint"):
        M.Mouth.from_env()
    with pytest.raises(CheckpointLoadError, match="not found"):
        M.Mouth(checkpoint="/nonexistent/mouth.pth")


def test_random_weights_never_move_the_mouth():
    with pytest.raises(ValueError, match="random weights"):
        M.Mouth(encoder=AudioEncoder(allow_random_init=True))
