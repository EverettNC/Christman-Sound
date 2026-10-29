# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
The sovereign synthesizer (christman_synth.c, the C port of Codec War), the
Vosk lexicon loader, and the Python entry points. Real runs only: the library
is built by scripts/build_dsp.py at the start, every number is measured from
the samples it produced, and pitch is measured with the package's own
christman_yin. A test that cannot run here reports NOT RUN with the reason
(tests/conftest.py); none passes by skipping.

Two environment hooks:
  CODECWAR_SRC   path to a CODECWAR5TH checkout. With node on PATH, the
                 comparison test runs Codec War's own synth.ts (node strips the
                 TypeScript types itself) and compares every sample.
  (no variable)  playback goes to the default device; with none, NOT RUN.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from christman_sound.dsp import ChristmanDSP, PlaybackUnavailable, SynthRefused  # noqa: E402
from christman_sound.lexicon import (  # noqa: E402
    DURATION_REASONS,
    NOT_MAPPED,
    VOSK_TO_CODECWAR,
    LexiconFormatError,
    LexiconUnavailable,
    PhoneNotMapped,
    WordNotInLexicon,
    load_lexicon,
)
from christman_sound.synth import (  # noqa: E402
    NothingToSay,
    UnknownPhone,
    render_codecwar_table,
    speak,
    text_to_phones,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_fixture import load_test_voice  # noqa: E402

#: Codec War's default pitch (synth.ts:31) and rate: the values its own
#: synth.ts is compared at. The bare table takes them by name; it has no default.
CW_F0 = 148.0
CW_RATE = 1.0


def bare_table(text=None, **kwargs):
    """Codec War's bare table, at Codec War's own pitch and rate unless the test names others."""
    kwargs.setdefault("f0", CW_F0)
    kwargs.setdefault("rate", CW_RATE)
    return render_codecwar_table(text, **kwargs)

# Codec War's table durations (phonemes.ts:18-57), written out here so the
# length test does not read the expected values from the code it is testing.
CW_DUR = {
    "SIL": 0.06, "AA": 0.1, "AE": 0.1, "AH": 0.07, "AO": 0.1, "AW": 0.14, "AY": 0.14,
    "EH": 0.08, "ER": 0.12, "EY": 0.13, "IH": 0.07, "IY": 0.1, "OW": 0.13, "OY": 0.14,
    "UH": 0.07, "UW": 0.1, "B": 0.06, "D": 0.05, "G": 0.06, "P": 0.06, "T": 0.05,
    "K": 0.06, "F": 0.09, "V": 0.07, "TH": 0.08, "DH": 0.06, "S": 0.09, "Z": 0.07,
    "SH": 0.1, "ZH": 0.08, "HH": 0.06, "CH": 0.09, "JH": 0.08, "M": 0.07, "N": 0.07,
    "NG": 0.08, "L": 0.07, "R": 0.07, "W": 0.07, "Y": 0.07,
}

# A small lexicon in the exact align_lexicon.txt format of vosk-model-en-us-0.22.
# The first two lines are the ones quoted from Everett's file. "butterd" and
# "bearshort" are control rows for the duration tests: the same phones with a
# plain d and a plain E, so the flap and long-E rows can be measured against them.
FIXTURE_LEXICON = """\
christman christman k_B r_I I_I s_I t_I m_I V_I n_E
hello hello h_B @_I l_I oU_E
filament filament f_B I_I l_I @_I m_I @_I n_I t_E
booth booth b_B u_I T_E
i i aI_S
butter butter b_B V_I 4_I @`_E
bear bear b_B E:_I r_E
butterd butterd b_B V_I d_I @`_E
bearshort bearshort b_B E_I r_E
uh uh UHH_S
qq qq Q_S
"""


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def dsp(tmp_path_factory) -> ChristmanDSP:
    """Build the library with the repository's own build script, then load it."""
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


@pytest.fixture()
def lexicon_file(tmp_path) -> Path:
    path = tmp_path / "align_lexicon.txt"
    path.write_text(FIXTURE_LEXICON, encoding="utf-8")
    return path


def _expected_text_timeline(words_phones):
    """Codec War's textToPhones rules, phonemes.ts:228-242, from CW_DUR."""
    seq = [("SIL", 0.12)]
    for phones in words_phones:
        for i, p in enumerate(phones):
            f = 1.05 if (i == 0 or i == len(phones) - 1) else 0.95
            seq.append((p, CW_DUR[p] * f))
        seq.append(("SIL", 0.08))
    return seq


def _expected_length(durs, rate, sr):
    """synth.ts:38 with the rate division of synth.ts:105."""
    t = 0.0
    for d in durs:
        t += d / rate
    return max(1, math.floor(t * sr) + math.floor(sr / 12))


def _yin_frames(dsp, samples, sr, start_s, end_s, frame=2048, hop=512):
    a, b = int(start_s * sr), int(end_s * sr)
    return [dsp.yin(samples[i:i + frame], sr) for i in range(a, b - frame, hop)]


# ------------------------------------------------------------ build + table

def test_library_built_and_phone_table_is_codec_wars(dsp):
    table = dsp.synth_phone_table()
    assert len(table) == 40
    assert {name: dur for name, (_, dur) in table.items()} == CW_DUR
    # Index order is Codec War's PHONE_INDEX: SIL is 0, AA is 1, Y is last.
    assert table["SIL"][0] == 0 and table["AA"][0] == 1 and table["Y"][0] == 39


def test_mapping_reaches_every_codec_war_phone(dsp):
    targets = {pid for pid, _ in VOSK_TO_CODECWAR.values()}
    assert targets == set(dsp.synth_phone_table()), "every Codec War phone must be reachable"
    assert not set(VOSK_TO_CODECWAR) & set(NOT_MAPPED)
    assert len(VOSK_TO_CODECWAR) == 44 and len(NOT_MAPPED) == 10
    # Everett's ruling, 2026-10-18: only the flap and the long E carry a
    # factor other than 1.0, each with its reason written down.
    scaled = {v: f for v, (_, f) in VOSK_TO_CODECWAR.items() if f != 1.0}
    assert scaled == {"4": 0.5, "E:": 1.625}
    assert VOSK_TO_CODECWAR["4"] == ("D", 0.5) and VOSK_TO_CODECWAR["E:"] == ("EH", 1.625)
    assert set(DURATION_REASONS) == {"4", "E:"}
    assert "4" not in NOT_MAPPED and "E:" not in NOT_MAPPED


# ---------------------------------------------------------------- pitch

# Tolerances, and why:
#   * Codec War's vibrato swings the pitch +/-1.2 % at 5.1 Hz (synth.ts:57).
#     A 2048-sample frame sees part of a vibrato cycle, so one frame may read
#     up to about that much off. Across the ~0.8 s window measured there are
#     four vibrato cycles, so the median sits on the requested pitch.
#   * christman_yin reports a whole-number lag with no interpolation: it takes
#     the first lag under its threshold and walks down to the local minimum.
#     With vibrato moving the true period, the median frame can land one whole
#     lag away. One lag step, taken at the shorter lag, is f0 / (sr - f0):
#     0.63 % at 100 Hz/16 kHz, 1.91 % at 300 Hz/16 kHz.
#   MEDIAN: within f0/(sr - f0) + 0.3 % (one lag step, plus what is left of
#     the vibrato in a median of ~21-31 frames).
#   EVERY FRAME: within 3 %. Vibrato (1.2 %) plus one lag step is 3.1 % at
#     the very worst corner (300 Hz/16 kHz); the largest seen across the grid
#     is 2.44 % (EH at 100 Hz/16 kHz). 3 % still catches any octave or
#     subharmonic error, which is 50 % or more.
PITCH_GRID = [
    (vowel, f0, sr)
    for sr in (16000, 22050)
    for vowel in ("AA", "IY", "UW", "EH", "OW")
    for f0 in (148.0, 220.0, 300.0)
] + [
    (vowel, f0, sr)
    for sr in (16000, 22050)
    for vowel in ("IY", "UW", "EH", "OW")
    for f0 in (100.0, 120.0)
]


@pytest.mark.parametrize("vowel,f0,sr", PITCH_GRID)
def test_sustained_vowel_pitch_matches_request(dsp, vowel, f0, sr):
    r = bare_table(phones=[("SIL", 0.03), (vowel, 1.0), ("SIL", 0.03)], f0=f0, sample_rate=sr, engine=dsp)
    est = _yin_frames(dsp, r.samples, sr, 0.13, 0.93)
    assert len(est) >= 20
    median = statistics.median(est)
    median_tol = f0 / (sr - f0) + 0.003
    worst = max(abs(e - f0) / f0 for e in est)
    print(f"{vowel} sr={sr} requested={f0:.1f} Hz measured median={median:.2f} Hz "
          f"({(median - f0) / f0 * 100:+.2f} %, tol {median_tol * 100:.2f} %) "
          f"worst frame {worst * 100:.2f} % of {len(est)} frames")
    assert abs(median - f0) / f0 <= median_tol
    assert worst <= 0.03, est


@pytest.mark.parametrize("f0,sr", [(100.0, 16000), (120.0, 16000), (100.0, 22050)])
def test_open_vowel_aa_at_low_pitch_finding(dsp, f0, sr):
    """
    FINDING, not a pass on the pitch claim. Codec War's AA at low pitch is
    less periodic frame to frame than YIN's 0.15 threshold demands: at
    100 Hz/16 kHz the normalised difference at the true period is about
    0.25. The likely cause is Codec War's vibrato together with its radiation
    stage (synth.ts:152), which lifts AA's high formants more at 16 kHz than
    at 22050. The TypeScript original gives identical samples, so this is the
    synthesizer's own behaviour, not the port's. What this test holds: every
    frame christman_yin calls voiced reads the true period or a whole
    multiple of it (f0, f0/2, f0/3) — the waveform repeats at the requested
    pitch. How many frames read unvoiced is printed for Everett.
    """
    r = bare_table(phones=[("SIL", 0.03), ("AA", 1.0), ("SIL", 0.03)], f0=f0, sample_rate=sr, engine=dsp)
    est = _yin_frames(dsp, r.samples, sr, 0.13, 0.93)
    voiced = [e for e in est if e > 0]
    at_f0 = [e for e in voiced if abs(e - f0) / f0 <= 0.03]
    print(f"AA sr={sr} requested={f0:.0f} Hz: {len(est)} frames, {len(est) - len(voiced)} unvoiced, "
          f"{len(at_f0)} at f0, {len(voiced) - len(at_f0)} at a multiple of the period: "
          f"{sorted(round(e, 1) for e in voiced if e not in at_f0)}")
    for e in voiced:
        assert any(abs(e - f0 / k) / (f0 / k) <= 0.03 for k in (1, 2, 3)), (e, est)


def test_pitch_contour_hook_follows_each_phone(dsp):
    """The one parameter Codec War lacks: f0 per phone."""
    sr = 16000
    phones = [("SIL", 0.03), ("IY", 0.6), ("IY", 0.6), ("SIL", 0.03)]
    contour = [150.0, 130.0, 210.0, 150.0]
    r = bare_table(phones=phones, f0_contour=contour, sample_rate=sr, engine=dsp)
    first = statistics.median(_yin_frames(dsp, r.samples, sr, 0.10, 0.60))
    second = statistics.median(_yin_frames(dsp, r.samples, sr, 0.70, 1.20))
    print(f"contour: requested 130.0 then 210.0 Hz, measured {first:.2f} then {second:.2f} Hz")
    assert abs(first - 130.0) / 130.0 <= 130 / (sr - 130) + 0.003
    assert abs(second - 210.0) / 210.0 <= 210 / (sr - 210) + 0.003
    assert r.f0 == tuple(contour)


# ---------------------------------------------------------------- length

@pytest.mark.parametrize("sr", [16000, 22050])
@pytest.mark.parametrize("rate", [0.6, 1.0, 1.25, 1.6])
def test_output_length_tracks_durations_and_rate(dsp, sr, rate):
    phones = [("SIL", 0.03), ("AA", 0.4), ("S", 0.09), ("IY", 0.25), ("SIL", 0.03)]
    r = bare_table(phones=phones, rate=rate, sample_rate=sr, engine=dsp)
    expected = _expected_length([d for _, d in phones], rate, sr)
    speech = len(r.samples) - math.floor(sr / 12)  # synth.ts:38 adds a 1/12 s tail
    print(f"sr={sr} rate={rate}: {len(r.samples)} samples, expected {expected}; "
          f"speech part {speech / sr:.4f} s vs requested {0.8 / rate:.4f} s")
    assert len(r.samples) == expected
    # floor() of a duration summed phone by phone can drop just under one
    # sample (0.8 s / 1.6 summed in pieces is 0.4999999...): Codec War does
    # the same. Never more than one sample.
    assert abs(speech / sr - sum(d for _, d in phones) / rate) <= 1.0 / sr + 1e-9
    starts = [p.start for p in r.phones]
    assert starts == sorted(starts) and abs(r.phones[-1].start + r.phones[-1].dur - 0.8 / rate) < 1e-9


# ------------------------------------------------------------ text phrase

@pytest.mark.parametrize("rate", [1.0, 1.5])
def test_short_phrase_from_inline_lexicon(dsp, lexicon_file, rate):
    sr = 16000
    r = bare_table("Hello, Christman.", rate=rate, lexicon_path=lexicon_file, engine=dsp)
    hello = ["HH", "AH", "L", "OW"]
    christman = ["K", "R", "IH", "S", "T", "M", "AH", "N"]
    expected = _expected_text_timeline([hello, christman])
    assert [p.id for p in r.phones] == [p for p, _ in expected]
    for got, (_, d) in zip(r.phones, expected):
        assert abs(got.dur - d / rate) < 1e-12
    assert len(r.samples) == _expected_length([d for _, d in expected], rate, sr)
    peak = max(abs(s) for s in r.samples)
    lead = r.samples[: int(0.12 / rate * sr)]
    rms = math.sqrt(sum(s * s for s in r.samples) / len(r.samples))
    print(f"rate={rate}: {len(r.samples)} samples ({r.duration:.3f} s), peak {peak:.4f}, rms {rms:.4f}, "
          f"lead silence max {max(abs(s) for s in lead):.2e}")
    assert abs(peak - 0.7) < 1e-6  # synth.ts:83 normalises the peak to 0.7
    assert max(abs(s) for s in lead) == 0.0  # the opening SIL is silent, not hiss
    assert rms > 0.05


def test_write_wav_round_trips_with_stdlib(dsp, lexicon_file, tmp_path):
    r = bare_table("hello", lexicon_path=lexicon_file, engine=dsp)
    path = r.write_wav(tmp_path / "hello.wav")
    with wave.open(str(path), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert w.getnframes() == len(r.samples)


# ------------------------------------------------------------ refusals

@pytest.mark.parametrize("sr", [16000, 22050])
@pytest.mark.parametrize(
    "word,control,position,cw_phone,cw_dur,factor",
    [
        # butter = b V 4 @` : the flap is an inside phone (Codec War x 0.95).
        ("butter", "butterd", 2, "D", CW_DUR["D"], 0.5),
        # bear = b E: r : the long E is an inside phone (x 0.95).
        ("bear", "bearshort", 1, "EH", CW_DUR["EH"], 1.625),
    ],
)
def test_flap_and_long_e_speak_at_their_factor(dsp, lexicon_file, sr, word, control, position,
                                               cw_phone, cw_dur, factor):
    """
    Everett's ruling, 2026-10-18: the flap is D x 0.5 and the long E is EH x
    1.625; neither refuses. Measured from the output: the word and its control
    row differ only in that one phone, so the difference in sample count is
    that phone's change in length. floor() at the end of each run may move
    either total by under one sample, so the allowance is 1 sample.
    """
    r = bare_table(word, lexicon_path=lexicon_file, sample_rate=sr, engine=dsp)
    ref = bare_table(control, lexicon_path=lexicon_file, sample_rate=sr, engine=dsp)
    seg = r.phones[1 + position]       # +1 for the opening SIL
    ref_seg = ref.phones[1 + position]
    assert seg.id == ref_seg.id == cw_phone
    expected_seg = cw_dur * 0.95 * factor
    assert abs(seg.dur - expected_seg) < 1e-12
    measured_delta = len(r.samples) - len(ref.samples)
    expected_delta = (expected_seg - cw_dur * 0.95) * sr
    measured_seg = ref_seg.dur + measured_delta / sr
    print(f"{word} sr={sr}: {cw_phone} x {factor} -> segment {expected_seg * 1000:.2f} ms expected; "
          f"measured from the samples {measured_seg * 1000:.2f} ms "
          f"({measured_delta:+d} samples vs control, expected {expected_delta:+.2f}); "
          f"measured factor {measured_seg / (cw_dur * 0.95):.4f}")
    assert abs(measured_delta - expected_delta) <= 1.0
    assert abs(max(abs(x) for x in r.samples) - 0.7) < 1e-6  # it spoke: peak-normalised output, not silence


def test_non_speech_phone_raises_typed_failure(dsp, lexicon_file):
    with pytest.raises(PhoneNotMapped) as exc:
        bare_table("uh", lexicon_path=lexicon_file, engine=dsp)
    assert exc.value.phone == "UHH"


def test_phone_outside_inventory_raises_typed_failure(dsp, lexicon_file):
    with pytest.raises(PhoneNotMapped) as exc:
        bare_table("qq", lexicon_path=lexicon_file, engine=dsp)
    assert "not in the known" in str(exc.value)


def test_missing_lexicon_raises_typed_failure(dsp, monkeypatch, tmp_path):
    monkeypatch.delenv("CHRISTMAN_LEXICON_PATH", raising=False)
    with pytest.raises(LexiconUnavailable) as exc:
        bare_table("hello", engine=dsp)
    print(exc.value)
    with pytest.raises(LexiconUnavailable):
        bare_table("hello", lexicon_path=tmp_path / "no_such_lexicon.txt", engine=dsp)
    monkeypatch.setenv("CHRISTMAN_LEXICON_PATH", str(tmp_path / "also_missing.txt"))
    with pytest.raises(LexiconUnavailable):
        bare_table("hello", engine=dsp)


def test_env_var_names_the_lexicon(dsp, monkeypatch, lexicon_file):
    monkeypatch.setenv("CHRISTMAN_LEXICON_PATH", str(lexicon_file))
    assert len(bare_table("hello", engine=dsp).samples) > 0


def test_word_not_in_lexicon_is_named_not_spelled(dsp, lexicon_file):
    with pytest.raises(WordNotInLexicon) as exc:
        bare_table("hello zebra quokka", lexicon_path=lexicon_file, engine=dsp)
    assert exc.value.words == ["zebra", "quokka"]


def test_malformed_lexicon_line_is_refused(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("hello hello h_B @_I l_I oU_E\nbroken\n", encoding="utf-8")
    with pytest.raises(LexiconFormatError, match=":2:"):
        load_lexicon(bad)


def test_engine_refusals_carry_their_reason(dsp, lexicon_file):
    vowel = [("SIL", 0.03), ("AA", 0.2), ("SIL", 0.03)]
    for kwargs, code in (({"rate": 2.0}, 5), ({"rate": 0.5}, 5), ({"sample_rate": 44100}, 6),
                         ({"f0": 0.0}, 4), ({"f0": 9000.0}, 4)):
        with pytest.raises(SynthRefused) as exc:
            bare_table(phones=vowel, engine=dsp, **kwargs)
        assert exc.value.code == code, (kwargs, exc.value)
    with pytest.raises(SynthRefused) as exc:
        bare_table(phones=[("AA", 0.0)], engine=dsp)
    assert exc.value.code == 3
    with pytest.raises(UnknownPhone):
        bare_table(phones=[("QQ", 0.1)], engine=dsp)
    with pytest.raises(NothingToSay):
        text_to_phones("  ... ?!  ", load_lexicon(lexicon_file), dsp)


# ------------------------------------------------------------ sovereignty

@pytest.mark.needs_dsp
def test_synth_path_imports_no_outside_audio_or_numeric_package(tmp_path):
    """Run the whole text-to-samples path with numpy, scipy, soundfile,
    sounddevice, pygame and librosa made unimportable."""
    lex = tmp_path / "lex.txt"
    lex.write_text(FIXTURE_LEXICON, encoding="utf-8")
    code = f"""
import builtins, sys
BLOCK = {{"numpy", "scipy", "soundfile", "sounddevice", "pygame", "librosa"}}
real = builtins.__import__
def guard(name, *a, **k):
    if name.split(".")[0] in BLOCK:
        raise ImportError("blocked " + name)
    return real(name, *a, **k)
builtins.__import__ = guard
sys.path.insert(0, {str(ROOT)!r})
from christman_sound.synth import render_codecwar_table
r = render_codecwar_table("hello christman", f0=148.0, rate=1.0, lexicon_path={str(lex)!r})
r.write_wav({str(tmp_path / "x.wav")!r})
print(len(r.samples), sorted(m for m in sys.modules if m.split(".")[0] in BLOCK))
"""
    env = dict(os.environ)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    count, loaded = proc.stdout.strip().split(" ", 1)
    assert int(count) > 0 and loaded == "[]"


# ------------------------------------------------------------ playback

@pytest.mark.plays_aloud
def test_speak_plays_through_christman_play(dsp, tmp_path):
    try:
        r = speak(phones=[("SIL", 0.03), ("AA", 0.3), ("SIL", 0.03)], voice=load_test_voice(tmp_path), engine=dsp)
    except PlaybackUnavailable as exc:
        pytest.skip(f"not run: no speaker reachable here ({dsp.play_backend()}): {exc}")
    assert len(r.samples) > 0


# ------------------------------------------------------------ against Codec War

_DRIVER = r"""
import { writeFileSync } from "node:fs";
const [dir, out, specJson] = process.argv.slice(2);
const m = await import(dir + "/synth.ts");
const spec = JSON.parse(specJson);
const r = spec.text !== undefined
  ? m.synthesize(spec.text, { f0: spec.f0, rate: spec.rate })
  : m.synthesizePhones(spec.phones.map(([id, dur]) => ({ id, start: 0, dur })), { f0: spec.f0 });
writeFileSync(out, Buffer.from(r.pcm.buffer));
"""


def test_matches_codec_war_typescript_sample_for_sample(dsp, lexicon_file, tmp_path):
    """
    Runs Codec War's own synth.ts under node (node strips the TypeScript
    types; nothing is installed or rewritten) and compares every sample.
    Allowed difference: 1e-5. Node's Math.sin/cos/exp and the C library's
    may differ in the last bit of a double; a porting error would be orders
    of magnitude larger. The count of exactly equal samples is printed.
    """
    import array

    src = os.environ.get("CODECWAR_SRC")
    node = shutil.which("node")
    if not src or not node:
        pytest.skip("not run: needs node on PATH and CODECWAR_SRC set to a CODECWAR5TH checkout "
                    f"(node={'found' if node else 'missing'}, CODECWAR_SRC={'set' if src else 'unset'})")
    audio = Path(src) / "src" / "lib" / "audio"
    if not (audio / "synth.ts").is_file():
        pytest.skip(f"not run: {audio / 'synth.ts'} not found")
    driver = tmp_path / "driver.mjs"
    driver.write_text(_DRIVER, encoding="utf-8")

    def reference(spec):
        out = tmp_path / "ref.f32"
        subprocess.run([node, str(driver), str(audio), str(out), json.dumps(spec)], check=True,
                       capture_output=True, text=True)
        ref = array.array("f")
        ref.frombytes(out.read_bytes())
        return ref

    every_phone = [[p, 0.11] for p in CW_DUR]
    cases = [
        ("all 40 phones, f0 148", {"phones": every_phone, "f0": 148},
         bare_table(phones=[tuple(x) for x in every_phone], f0=148, engine=dsp).samples),
        ("sustained AA, f0 200", {"phones": [["SIL", 0.03], ["AA", 0.5], ["SIL", 0.03]], "f0": 200},
         bare_table(phones=[("SIL", 0.03), ("AA", 0.5), ("SIL", 0.03)], f0=200, engine=dsp).samples),
        # Text path: Codec War's own word list vs our Vosk-lexicon path for
        # the same words, rate 1.3 and f0 130 — checks the duration rules too.
        ("text 'hello filament booth', f0 130, rate 1.3", {"text": "hello filament booth", "f0": 130, "rate": 1.3},
         bare_table("hello filament booth", f0=130, rate=1.3, lexicon_path=lexicon_file, engine=dsp).samples),
    ]
    for label, spec, ours in cases:
        ref = reference(spec)
        assert len(ours) == len(ref), (label, len(ours), len(ref))
        diffs = [abs(a - b) for a, b in zip(ours, ref)]
        exact = sum(1 for d in diffs if d == 0.0)
        print(f"{label}: {len(ref)} samples, {exact} identical, max abs difference {max(diffs):.3e}")
        assert max(diffs) <= 1e-5, label
