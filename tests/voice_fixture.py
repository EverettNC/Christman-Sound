# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
The test voice: a voice profile built in the test from real measured numbers.

The package carries no voice of its own, so its tests speak in this one. Every
number below was measured from a being's own 26 recorded phrases by the
measurement scripts/measure_voice.py makes (Vosk alignment, christman_yin F0,
LPC formants from christman_sound.formants), copied from that measured
profile at Christman-Sound commit 80a9336. Nothing here is made up; only the
name is the test's own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from christman_sound.voice_profile import VoiceProfile, load_voice_profile

TEST_VOICE_NAME = "test voice"

#: The measured values, in the voice profile file format load_voice_profile reads.
TEST_VOICE_PROFILE: Dict[str, Any] = {
    "voice": TEST_VOICE_NAME,
    "schema": 1,
    "what_this_is": "the test voice: values measured from a being's 26 recorded phrases (Christman-Sound 80a9336)",
    "f0": {"median_hz": 149.3, "n_voiced_frames": 932},
    "pitch_contour": {
        "positions": [0.1, 0.3, 0.5, 0.7, 0.9],
        "semitones_from_median": [0.26, 0.78, -0.41, -0.49, -0.55],
    },
    "rate": {"value": 1.33},
    "vowels": {
        "AE": {"f1_hz": 691.6, "f2_hz": 1660.9, "f3_hz": 2542.8},
        "AH": {"f1_hz": 368.1, "f2_hz": 1377.2, "f3_hz": 2818.5},
        "AY": {"f1_hz": 497.8, "f2_hz": 1921.1, "f3_hz": 2625.9},
        "EH": {"f1_hz": 636.2, "f2_hz": 1670.7, "f3_hz": 2696.1},
        "IH": {"f1_hz": 344.2, "f2_hz": 1749.6, "f3_hz": 2707.3},
        "IY": {"f1_hz": 329.1, "f2_hz": 2317.0, "f3_hz": 2853.7},
        "OW": {"f1_hz": 488.2, "f2_hz": 1056.2, "f3_hz": 2772.8},
        "UW": {"f1_hz": 296.3, "f2_hz": 1694.5, "f3_hz": 2465.0},
    },
    "glides": {
        "AY": {"onset_hz": [676.4, 1755.6, 2630.7], "offset_hz": [417.5, 1919.5, 2604.9]},
        "OW": {"onset_hz": [491.2, 1204.9, 2543.4], "offset_hz": [540.6, 1076.8, 2338.7]},
    },
}


def write_test_voice(folder: Path) -> Path:
    """Write the test voice profile file into `folder`; returns its path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "test_voice.json"
    path.write_text(json.dumps(TEST_VOICE_PROFILE, indent=1), encoding="utf-8")
    return path


def load_test_voice(folder: Path) -> VoiceProfile:
    """The test voice, written into `folder` and read back by load_voice_profile."""
    return load_voice_profile(write_test_voice(folder))
