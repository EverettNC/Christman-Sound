# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
A caller's measured voice in the sovereign synthesizer.

The package carries no voice profile and no default voice. These tests speak
in the test voice (tests/voice_fixture.py: numbers measured from a being's
own recordings) and hold:

  * christman_synth_voiced with Codec War's own rows is christman_synth,
    sample for sample: the new input changes nothing unless it is used;
  * the measured vowels reach the output: the synthesized vowel, measured
    with christman_sound.formants, sits at the voice's measured F1-F3 and
    not at Codec War's;
  * no voice, no sound: synthesize and speak refuse without the caller's
    voice, and the package ships no profile of its own;
  * the profile's fingerprint is the hash of the file passed;
  * the refusals: bad formant rows (code 9), a voice plus f0/rate, a
    profile missing a value;
  * the measurement itself (scripts/measure_voice.py), run on a being's
    recordings with christman_sound.ear, writes every value as the median of the
    measurements beside it. That run needs the recordings and the model
    folder the ear reads; without them it reports NOT RUN.

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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vosk_models import require_full_model  # noqa: E402
sys.path.insert(0, str(ROOT))

from christman_sound.dsp import ChristmanDSP, SynthRefused  # noqa: E402
from christman_sound.formants import measure_formants  # noqa: E402
from christman_sound import synth as cs_synth  # noqa: E402
from christman_sound import voice_profile as vp  # noqa: E402
from christman_sound.voice_profile import VoiceProfileError, load_voice_profile  # noqa: E402
from voice_fixture import TEST_VOICE_NAME, TEST_VOICE_PROFILE, load_test_voice, write_test_voice  # noqa: E402


@pytest.fixture
def voice(tmp_path):
    return load_test_voice(tmp_path)


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


# ------------------------------------------------------- the voice's vowels


def _vowel_medians(dsp, voice):
    """Synthesize a held UW and measure it with christman_sound.formants."""
    phones = [("SIL", 0.05), ("UW", 0.5), ("SIL", 0.05)]
    if voice is None:
        r = cs_synth.render_codecwar_table(phones=phones, f0=TEST_VOICE_PROFILE["f0"]["median_hz"], rate=1.0, engine=dsp)
    else:
        r = cs_synth.synthesize(phones=phones, voice=voice, engine=dsp)
    ph = r.phones[1]
    a = measure_formants(list(r.samples), r.sample_rate)
    fr = [f for f in a.frames if ph.start + 0.03 <= f.time_s <= ph.start + ph.dur - 0.03
          and f.f1_hz and f.f2_hz and f.f3_hz]
    assert len(fr) >= 10, len(fr)
    return tuple(statistics.median(getattr(f, k) for f in fr) for k in ("f1_hz", "f2_hz", "f3_hz"))


def test_measured_vowel_reaches_the_output_and_codec_wars_does_not(dsp, voice):
    """
    UW is the vowel the measurement moved most from Codec War's table (F2
    870 Hz in the table). The synthesized UW, measured, must sit within 8 %
    of the voice's measured F2 and F3 and within 15 % of its F1 (F1 has few
    harmonics under it at 149 Hz; formants.py states that limit).
    """
    want = voice.vowel_formants["UW"]
    got = _vowel_medians(dsp, voice)
    old = _vowel_medians(dsp, None)
    print(f"UW measured {want}, synthesized with it {tuple(round(x) for x in got)}, "
          f"Codec War's bare table {tuple(round(x) for x in old)}")
    assert abs(got[0] - want[0]) / want[0] <= 0.15
    assert abs(got[1] - want[1]) / want[1] <= 0.08
    assert abs(got[2] - want[2]) / want[2] <= 0.08
    assert abs(old[1] - 870.0) / 870.0 <= 0.08  # the bare table is where it was
    assert abs(got[1] - want[1]) < abs(old[1] - want[1])


def test_voice_is_the_one_source_of_pitch_and_rate(dsp, voice):
    for extra in ({"f0": 150.0}, {"rate": 1.0}, {"f0_contour": [150.0] * 3}):
        with pytest.raises(ValueError):
            cs_synth.synthesize(phones=[("SIL", 0.05), ("AA", 0.2), ("SIL", 0.05)], voice=voice, engine=dsp, **extra)
    r = cs_synth.synthesize(phones=[("SIL", 0.05), ("AA", 0.2), ("SIL", 0.05)], voice=voice, engine=dsp)
    assert r.voice == TEST_VOICE_NAME and r.rate == voice.rate
    assert abs(r.phones[1].dur - 0.2 / voice.rate) < 1e-12
    assert r.renderer == cs_synth.RENDERER_CLEAR


# ------------------------------------------------- no voice, no sound


def test_no_voice_is_refused(dsp, tmp_path):
    phones = [("SIL", 0.05), ("AA", 0.2), ("SIL", 0.05)]
    with pytest.raises(cs_synth.NoVoice, match="no default voice"):
        cs_synth.synthesize(phones=phones, voice=None, engine=dsp)
    with pytest.raises(TypeError):
        cs_synth.synthesize(phones=phones, engine=dsp)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        cs_synth.speak(phones=phones, engine=dsp)  # type: ignore[call-arg]
    with pytest.raises(cs_synth.NoVoice):
        cs_synth.synthesize(phones=phones, voice=149.3, engine=dsp)  # type: ignore[arg-type]
    with pytest.raises(VoiceProfileError):
        cs_synth.synthesize(phones=phones, voice=tmp_path / "absent.json", engine=dsp)


def test_the_package_ships_no_voice():
    assert not list((ROOT / "christman_sound").glob("*.json"))
    loaders = [n for n in dir(vp) if n.startswith("load_")]
    assert loaders == ["load_voice_profile"], loaders  # one loader, and it takes the caller's path
    assert vp.__all__ == ["VoiceProfileError", "VoiceProfile", "load_voice_profile"]


def test_a_path_is_a_voice_and_says_whose(dsp, tmp_path):
    path = write_test_voice(tmp_path)
    phones = [("SIL", 0.05), ("AA", 0.2), ("SIL", 0.05)]
    by_path = cs_synth.synthesize(phones=phones, voice=str(path), engine=dsp)
    by_obj = cs_synth.synthesize(phones=phones, voice=load_voice_profile(path), engine=dsp)
    assert list(by_path.samples) == list(by_obj.samples)
    assert by_path.voice == TEST_VOICE_NAME


def test_fingerprint_is_the_hash_of_the_passed_profile(tmp_path):
    import hashlib

    path = write_test_voice(tmp_path / "a")
    v = load_voice_profile(path)
    assert v.fingerprint() == hashlib.sha256(path.read_bytes()).hexdigest()
    d = json.loads(path.read_text(encoding="utf-8"))
    d["rate"]["value"] = 1.2
    other = tmp_path / "b.json"
    other.write_text(json.dumps(d), encoding="utf-8")
    assert load_voice_profile(other).fingerprint() != v.fingerprint()
    built = vp.VoiceProfile(name=v.name, path=None, f0_median_hz=v.f0_median_hz, contour_points=v.contour_points,
                            rate=v.rate, vowel_formants=v.vowel_formants, glides=v.glides)
    assert len(built.fingerprint()) == 64 and built.fingerprint() != v.fingerprint()
    assert cs_synth.voice_fingerprint(v) == v.fingerprint()


def test_profile_refusals(tmp_path):
    good = TEST_VOICE_PROFILE
    cases = [
        ("f0", lambda d: d.pop("f0")),
        ("rate", lambda d: d["rate"].update(value=2.5)),
        ("pitch_contour", lambda d: d["pitch_contour"]["positions"].pop()),
        ("f2_hz", lambda d: next(iter(d["vowels"].values())).pop("f2_hz")),
        ("vowels", lambda d: d.update(vowels={})),
        ("glide_offset", lambda d: d["glides"]["AY"].pop("offset_hz")),
        ("glide_short", lambda d: d["glides"]["AY"]["onset_hz"].pop()),
        ("glide_zero", lambda d: d["glides"]["AY"]["onset_hz"].__setitem__(0, 0)),
        ("glide_not_gliding", lambda d: d["glides"].update(AA=d["glides"]["AY"])),
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


# ------------------------------------------------- the measurement, run


def _check_medians(profile: dict) -> None:
    """Every value in a measured profile is the median of the measurements written beside it."""
    names = {f["name"] for f in profile["sources"]["recordings"]["files"]}
    for group in ("vowels", "measured_not_applied"):
        for v, e in profile[group].items():
            toks = e["tokens"]
            assert e["n_tokens"] == len(toks)
            assert e["n_frames"] == sum(t["n_frames"] for t in toks)
            for k in ("f1_hz", "f2_hz", "f3_hz"):
                assert e[k] == round(statistics.median(t[k] for t in toks), 1), (v, k)
            assert all(t["phrase"] in names and t["vowel"] == v and t["n_frames"] > 0 for t in toks)
    assert all(e["n_tokens"] >= 2 for e in profile["vowels"].values())
    assert all(e["n_tokens"] < 2 for e in profile["measured_not_applied"].values())
    applied, held, none = set(profile["vowels"]), set(profile["measured_not_applied"]), set(profile["not_measured"])
    assert not (applied & held) and not (applied & none) and not (held & none)
    assert len(applied | held | none) == 15
    for group in ("glides", "glides_not_applied"):
        for v, e in profile[group].items():
            toks = e["tokens"]
            assert e["n_tokens"] == len(toks)
            assert all(t["phrase"] in names and t["n_frames"] >= 3 for t in toks)
            for k in ("onset_hz", "offset_hz"):
                assert e[k] == [round(statistics.median(t[k][j] for t in toks), 1) for j in range(3)], (v, k)
    c = profile["pitch_contour"]
    for k, value in enumerate(c["semitones_from_median"]):
        col = [p["semitones"][k] for p in c["per_phrase"].values() if p["semitones"][k] is not None]
        assert len(col) == c["n_phrases_per_position"][k]
        assert value == round(statistics.median(col), 2)
    r = profile["rate"]
    assert r["value"] == round(statistics.median(p["ratio"] for p in r["per_phrase"].values()), 3)
    assert r["n_phrases"] == len(r["per_phrase"])
    assert 0.6 <= r["value"] <= 1.6


def test_measurement_from_a_beings_recordings(tmp_path):
    """
    Runs scripts/measure_voice.py on a being's recordings with
    christman_sound.ear (2026-10-29: the old recognizer is removed), loads
    what it wrote, and checks every value against the
    measurements written beside it. With CHRISTMAN_VOICE_REFERENCE_PROFILE
    (that being's earlier measured profile) it also compares every applied
    value with it. Needs CHRISTMAN_VOICE_RECORDINGS (the folder of recorded
    phrases; CHRISTMAN_VOICE_PREFIX is their file name prefix),
    CHRISTMAN_VOSK_MODEL (the vosk-model-en-us-0.22 folder the ear reads) and
    CHRISTMAN_LEXICON_PATH. It runs on this Python, the one with
    Christman-Sound installed and the ear's C library built.
    """
    need = {k: os.environ.get(k) for k in
            ("CHRISTMAN_VOICE_RECORDINGS", "CHRISTMAN_VOSK_MODEL", "CHRISTMAN_LEXICON_PATH")}
    missing = [k for k, v in need.items() if not v]
    if missing:
        pytest.skip("not run: needs " + ", ".join(missing) +
                    " (a being's recordings, the model folder, its lexicon); runs on Everett's Mac")
    require_full_model(need["CHRISTMAN_VOSK_MODEL"])
    out = tmp_path / "measured_voice.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "measure_voice.py"),
         "--recordings", need["CHRISTMAN_VOICE_RECORDINGS"], "--prefix", os.environ.get("CHRISTMAN_VOICE_PREFIX", ""),
         "--name", "measured voice", "--vosk-model", need["CHRISTMAN_VOSK_MODEL"],
         "--lexicon", need["CHRISTMAN_LEXICON_PATH"], "--out", str(out)],
        capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    fresh = json.loads(out.read_text(encoding="utf-8"))
    assert fresh["voice"] == "measured voice"
    assert fresh["sources"]["script"] == "scripts/measure_voice.py"
    _check_medians(fresh)
    load_voice_profile(out)
    ref_path = os.environ.get("CHRISTMAN_VOICE_REFERENCE_PROFILE")
    if not ref_path:
        return
    ref = json.loads(Path(ref_path).expanduser().read_text(encoding="utf-8"))
    # The old recognizer's word times on the same file moved by up to one
    # 30 ms frame from run to run (measured 2026-10-28: 3 to 4 of 26
    # phrases), and the values moved with them: over three runs, 10 Hz on
    # one vowel's F3, 0.06 semitones on one contour point, rate unchanged.
    # The tolerances below are a few times that; a real change is far
    # larger. The ear's run to run is not yet measured here, so the
    # tolerances stay as they were.
    assert fresh["f0"]["median_hz"] == ref["f0"]["median_hz"]
    assert abs(fresh["rate"]["value"] - ref["rate"]["value"]) <= 0.02
    for x, y in zip(fresh["pitch_contour"]["semitones_from_median"], ref["pitch_contour"]["semitones_from_median"]):
        assert abs(x - y) <= 0.25
    for group in ("vowels", "glides"):
        assert set(fresh[group]) == set(ref[group]), group
    for v, e in ref["vowels"].items():
        for k in ("f1_hz", "f2_hz", "f3_hz"):
            assert abs(fresh["vowels"][v][k] - e[k]) <= 0.02 * e[k], (v, k, fresh["vowels"][v][k], e[k])
    for v, e in ref["glides"].items():
        for k in ("onset_hz", "offset_hz"):
            for got, want in zip(fresh["glides"][v][k], e[k]):
                assert abs(got - want) <= 0.02 * want, (v, k, got, want)
