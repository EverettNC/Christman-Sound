# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
AlphaVox's measured voice in the sovereign synthesizer (2026-10-28).

christman_sound/alphavox_voice.json is measured from AlphaVox's own 26
recorded phrases by scripts/measure_alphavox_voice.py. These tests hold:

  * christman_synth_voiced with Codec War's own rows is christman_synth,
    sample for sample: the new input changes nothing unless it is used;
  * the measured vowels reach the output: the synthesized vowel, measured
    with christman_sound.formants, sits at AlphaVox's measured F1-F3 and
    not at Codec War's;
  * every value in the profile is the median of the measurements written
    beside it, recomputed here from the file itself;
  * the refusals: bad formant rows (code 9), a voice plus f0/rate, a
    profile missing a value;
  * the measurement itself, re-run from the recordings with Everett's Vosk,
    reproduces the committed profile. That run needs the recordings, the
    Vosk model and a Python with vosk; without them it reports NOT RUN.

Real runs only: the C library is built by scripts/build_dsp.py here.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from christman_sound.dsp import ChristmanDSP, SynthRefused  # noqa: E402
from christman_sound.formants import measure_formants  # noqa: E402
from christman_sound import synth as cs_synth  # noqa: E402
from christman_sound.voice_profile import (  # noqa: E402
    ALPHAVOX_PROFILE_PATH,
    VoiceProfileError,
    load_alphavox_voice,
    load_voice_profile,
)

PROFILE = json.loads(ALPHAVOX_PROFILE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dsp(tmp_path_factory) -> ChristmanDSP:
    out = tmp_path_factory.mktemp("lib") / "christman_dsp.so"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_dsp.py"), "--output", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"build_dsp.py failed:\n{proc.stdout}\n{proc.stderr}"
    engine = ChristmanDSP(out)
    assert engine.available, engine.error
    assert engine.synth_available, engine.synth_error
    return engine


PHONES = [("SIL", 0.05), ("HH", 0.06), ("UW", 0.12), ("L", 0.07), ("IY", 0.1), ("S", 0.09), ("SIL", 0.05)]


# ------------------------------------------------------------ the C input


def test_voiced_with_codec_war_rows_is_christman_synth_sample_for_sample(dsp):
    table = dsp.synth_phone_table()
    rows_table = dsp.synth_phone_formants()
    ids = [table[p][0] for p, _ in PHONES]
    durs = [d for _, d in PHONES]
    for sr in (16000, 22050):
        plain = dsp.synth(ids, durs, 149.3, 1.0, sr)
        rows = [rows_table[p][0] for p, _ in PHONES]
        voiced = dsp.synth(ids, durs, 149.3, 1.0, sr, None, rows)
        assert list(plain) == list(voiced), sr
    # and the table read back from the C is Codec War's (phonemes.ts:19, :33)
    assert rows_table["AA"] == ((730.0, 1090.0, 2440.0, 70.0, 90.0), True)
    assert rows_table["UW"] == ((300.0, 870.0, 2240.0, 60.0, 80.0), True)
    assert rows_table["S"][1] is False


def test_bad_formant_row_is_refused_with_code_9(dsp):
    table = dsp.synth_phone_table()
    ids = [table["SIL"][0], table["AA"][0]]
    for bad in (0.0, -5.0, float("nan"), 8000.0):
        with pytest.raises(SynthRefused) as exc:
            dsp.synth(ids, [0.1, 0.1], 149.3, 1.0, 16000, None, [(730, 1090, 2440, 70, 90), (bad, 1090, 2440, 70, 90)])
        assert exc.value.code == 9, bad
    with pytest.raises(ValueError):
        dsp.synth(ids, [0.1, 0.1], 149.3, 1.0, 16000, None, [(730, 1090, 2440, 70, 90)])


# ------------------------------------------------------- AlphaVox's vowels


def _vowel_medians(dsp, voice):
    """Synthesize a held UW and measure it with christman_sound.formants."""
    phones = [("SIL", 0.05), ("UW", 0.5), ("SIL", 0.05)]
    if voice is None:
        r = cs_synth.synthesize(phones=phones, f0=149.3, engine=dsp)
    else:
        r = cs_synth.synthesize(phones=phones, voice=voice, engine=dsp)
    ph = r.phones[1]
    a = measure_formants(list(r.samples), r.sample_rate)
    fr = [f for f in a.frames if ph.start + 0.03 <= f.time_s <= ph.start + ph.dur - 0.03
          and f.f1_hz and f.f2_hz and f.f3_hz]
    assert len(fr) >= 10, len(fr)
    return tuple(statistics.median(getattr(f, k) for f in fr) for k in ("f1_hz", "f2_hz", "f3_hz"))


def test_alphavox_vowel_reaches_the_output_and_codec_wars_does_not(dsp):
    """
    UW is the vowel AlphaVox's measurement moved most from Codec War's
    table (F2 870 Hz in the table). The synthesized UW, measured, must sit
    within 8 % of AlphaVox's measured F2 and F3 and within 15 % of its F1
    (F1 has few harmonics under it at 149 Hz; formants.py states that limit).
    """
    voice = load_alphavox_voice()
    want = voice.vowel_formants["UW"]
    got = _vowel_medians(dsp, voice)
    old = _vowel_medians(dsp, None)
    print(f"UW AlphaVox measured {want}, synthesized with it {tuple(round(x) for x in got)}, "
          f"Codec War's table voice {tuple(round(x) for x in old)}")
    assert abs(got[0] - want[0]) / want[0] <= 0.15
    assert abs(got[1] - want[1]) / want[1] <= 0.08
    assert abs(got[2] - want[2]) / want[2] <= 0.08
    assert abs(old[1] - 870.0) / 870.0 <= 0.08  # the table voice is where it was
    assert abs(got[1] - want[1]) < abs(old[1] - want[1])


def test_voice_is_the_one_source_of_pitch_and_rate(dsp):
    voice = load_alphavox_voice()
    for extra in ({"f0": 150.0}, {"rate": 1.0}, {"f0_contour": [150.0] * 3}):
        with pytest.raises(ValueError):
            cs_synth.synthesize(phones=[("SIL", 0.05), ("AA", 0.2), ("SIL", 0.05)], voice=voice, engine=dsp, **extra)
    r = cs_synth.synthesize(phones=[("SIL", 0.05), ("AA", 0.2), ("SIL", 0.05)], voice=voice, engine=dsp)
    assert r.voice == "AlphaVox" and r.rate == voice.rate
    assert abs(r.phones[1].dur - 0.2 / voice.rate) < 1e-12


# ------------------------------------------- every value beside its source


def test_profile_is_alphavox_and_names_its_sources():
    assert PROFILE["voice"] == "AlphaVox"
    files = PROFILE["sources"]["recordings"]["files"]
    assert len(files) == 26
    assert all(len(f["sha256"]) == 64 for f in files)
    assert PROFILE["sources"]["recordings"]["directory"] == "data/phrases/alphavox_verified"
    assert PROFILE["sources"]["recordings"]["repository"] == "EverettNC/AlphaVox"
    assert PROFILE["sources"]["aligner"]["model"] == "vosk-model-en-us-0.22"
    text = ALPHAVOX_PROFILE_PATH.read_text(encoding="utf-8")
    for wrong in ("Everett's recorded", "his recorded", "Everett's voice", "his formants", "Everett's formants"):
        assert wrong not in text, wrong


def test_each_vowel_is_the_median_of_its_tokens():
    names = {f["name"] for f in PROFILE["sources"]["recordings"]["files"]}
    for group in ("vowels", "measured_not_applied"):
        for v, e in PROFILE[group].items():
            toks = e["tokens"]
            assert e["n_tokens"] == len(toks)
            assert e["n_frames"] == sum(t["n_frames"] for t in toks)
            for k in ("f1_hz", "f2_hz", "f3_hz"):
                assert e[k] == round(statistics.median(t[k] for t in toks), 1), (v, k)
            assert all(t["phrase"] in names and t["vowel"] == v and t["n_frames"] > 0 for t in toks)
    assert all(e["n_tokens"] >= 2 for e in PROFILE["vowels"].values())
    assert all(e["n_tokens"] < 2 for e in PROFILE["measured_not_applied"].values())
    # nothing applied, held back and unmeasured at once; together they cover Codec War's 15 vowels
    applied, held, none = set(PROFILE["vowels"]), set(PROFILE["measured_not_applied"]), set(PROFILE["not_measured"])
    assert not (applied & held) and not (applied & none) and not (held & none)
    assert len(applied | held | none) == 15


def test_pitch_and_rate_are_the_medians_written_beside_them():
    c = PROFILE["pitch_contour"]
    per = c["per_phrase"].values()
    for k, value in enumerate(c["semitones_from_median"]):
        col = [p["semitones"][k] for p in per if p["semitones"][k] is not None]
        assert len(col) == c["n_phrases_per_position"][k]
        assert value == round(statistics.median(col), 2)
    r = PROFILE["rate"]
    assert r["value"] == round(statistics.median(p["ratio"] for p in r["per_phrase"].values()), 3)
    assert r["n_phrases"] == len(r["per_phrase"])
    assert 0.6 <= r["value"] <= 1.6
    assert PROFILE["f0"]["median_hz"] == 149.3


def test_profile_refusals(tmp_path):
    good = json.loads(ALPHAVOX_PROFILE_PATH.read_text(encoding="utf-8"))
    cases = [
        ("f0", lambda d: d.pop("f0")),
        ("rate", lambda d: d["rate"].update(value=2.5)),
        ("pitch_contour", lambda d: d["pitch_contour"]["positions"].pop()),
        ("f2_hz", lambda d: next(iter(d["vowels"].values())).pop("f2_hz")),
        ("vowels", lambda d: d.update(vowels={})),
    ]
    for name, spoil in cases:
        d = json.loads(json.dumps(good))
        spoil(d)
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        with pytest.raises(VoiceProfileError):
            load_voice_profile(p)
    with pytest.raises(VoiceProfileError):
        load_voice_profile(tmp_path / "absent.json")


# ------------------------------------------------- the measurement, re-run


def test_measurement_from_alphavox_recordings_reproduces_the_profile(tmp_path):
    """
    Re-runs scripts/measure_alphavox_voice.py on AlphaVox's recordings with
    Everett's Vosk and compares every applied value with the committed file.
    Needs ALPHAVOX_PHRASES_DIR (AlphaVox data/phrases/alphavox_verified),
    CHRISTMAN_VOSK_MODEL (vosk-model-en-us-0.22), CHRISTMAN_LEXICON_PATH and
    CHRISTMAN_VOSK_PYTHON (a Python that imports vosk).
    """
    need = {k: os.environ.get(k) for k in
            ("ALPHAVOX_PHRASES_DIR", "CHRISTMAN_VOSK_MODEL", "CHRISTMAN_LEXICON_PATH", "CHRISTMAN_VOSK_PYTHON")}
    missing = [k for k, v in need.items() if not v]
    if missing:
        pytest.skip("not run: needs " + ", ".join(missing) +
                    " (AlphaVox's recordings, the Vosk model, its lexicon, a Python with vosk); runs on Everett's Mac")
    out = tmp_path / "alphavox_voice.json"
    proc = subprocess.run(
        [need["CHRISTMAN_VOSK_PYTHON"], str(ROOT / "scripts" / "measure_alphavox_voice.py"),
         "--phrases", need["ALPHAVOX_PHRASES_DIR"], "--vosk-model", need["CHRISTMAN_VOSK_MODEL"],
         "--lexicon", need["CHRISTMAN_LEXICON_PATH"], "--out", str(out)],
        capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    fresh = json.loads(out.read_text(encoding="utf-8"))
    assert [f["sha256"] for f in fresh["sources"]["recordings"]["files"]] == \
           [f["sha256"] for f in PROFILE["sources"]["recordings"]["files"]]
    # Vosk's word times on the same file move by up to one 30 ms frame from
    # run to run (measured 2026-10-28: 3 to 4 of the 26 phrases), and the
    # values move with them. Measured spread over three runs: 10 Hz on one
    # vowel's F3, 0.06 semitones on one contour point, rate unchanged. The
    # tolerances below are a few times that; a real change is far larger.
    assert fresh["f0"]["median_hz"] == PROFILE["f0"]["median_hz"]
    assert abs(fresh["rate"]["value"] - PROFILE["rate"]["value"]) <= 0.02
    for x, y in zip(fresh["pitch_contour"]["semitones_from_median"], PROFILE["pitch_contour"]["semitones_from_median"]):
        assert abs(x - y) <= 0.25
    for group in ("vowels", "measured_not_applied", "not_measured"):
        assert set(fresh[group]) == set(PROFILE[group]), group
    for v, e in PROFILE["vowels"].items():
        for k in ("f1_hz", "f2_hz", "f3_hz"):
            assert abs(fresh["vowels"][v][k] - e[k]) <= 0.02 * e[k], (v, k, fresh["vowels"][v][k], e[k])
