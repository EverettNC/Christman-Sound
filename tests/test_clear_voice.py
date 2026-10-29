# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
christman_synth_clear: a being's measured voice, rendered so the words can
be understood (2026-10-28).

Through christman_synth_voiced, Everett's Filament ear heard a being's free
text as "the": 4 of 53 words on twelve sentences. These tests hold:

  * christman_synth and christman_synth_voiced are not changed (their own
    tests, test_sovereign_synth.py and test_voice_profile.py, still run);
  * the clear renderer keeps the contract: the same length, the same
    refusals, and one more (code 10, a bad glide row);
  * the vowel's resonances reach the output: a held AA's spectrum sits up
    among its formants (the old renderer's sat at less than half the
    height), and a vowel's F1-F3, measured, sit at the voice's rows;
  * S and SH put their noise where it belongs (S high, SH lower);
  * AY glides from the voice's measured onset to its measured offset;
  * a measured voice is rendered by the clear renderer, and says so;
  * words: with Everett's Vosk model on the machine, twelve sentences are
    synthesized in the test voice and recognized; the clear renderer must
    be heard far better than christman_synth_voiced. Without the model or a
    Python vosk, NOT RUN.

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

from christman_sound import synth as cs_synth  # noqa: E402
from christman_sound.dsp import ChristmanDSP, SynthRefused  # noqa: E402
from christman_sound.formants import measure_formants  # noqa: E402
from voice_fixture import TEST_VOICE_NAME, load_test_voice  # noqa: E402


@pytest.fixture(scope="module")
def voice(tmp_path_factory):
    """The test voice (tests/voice_fixture.py): measured numbers, a neutral name."""
    return load_test_voice(tmp_path_factory.mktemp("voice"))


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
    assert engine.synth_clear_available, engine.synth_clear_error
    return engine


def _spectrum(samples, sr, a, b):
    """Power per FFT bin over [a, b) seconds, Hann frames of 512 with hop 128."""
    import numpy as np

    x = np.asarray(samples[int(a * sr): int(b * sr)], dtype=float)
    acc = np.zeros(257)
    for i in range(0, len(x) - 512, 128):
        acc += np.abs(np.fft.rfft(x[i:i + 512] * np.hanning(512))) ** 2
    return np.fft.rfftfreq(512, 1 / sr), acc


def _centroid(samples, sr, a, b):
    f, p = _spectrum(samples, sr, a, b)
    return float((f * p).sum() / p.sum())


# ------------------------------------------------------------- the contract


def test_clear_keeps_the_length_and_the_refusals(dsp):
    table = dsp.synth_phone_table()
    ids = [table[p][0] for p in ("SIL", "AA", "S", "SIL")]
    durs = [0.05, 0.2, 0.1, 0.05]
    for sr in (16000, 22050):
        for rate in (0.6, 1.0, 1.33, 1.6):
            plain = dsp.synth(ids, durs, 149.3, rate, sr)
            clear = dsp.synth(ids, durs, 149.3, rate, sr, clear=True)
            assert len(clear) == len(plain)
            assert max(abs(v) for v in clear) == pytest.approx(0.7, abs=1e-6)
    cases = [
        (dict(phone_ids=[99], durations=[0.1]), 2),
        (dict(phone_ids=[ids[1]], durations=[0.0]), 3),
        (dict(phone_ids=[ids[1]], durations=[0.1], f0=-1.0), 4),
        (dict(phone_ids=[ids[1]], durations=[0.1], rate=2.0), 5),
        (dict(phone_ids=[ids[1]], durations=[0.1], sample_rate=44100), 6),
    ]
    for kw, code in cases:
        args = dict(phone_ids=None, durations=None, f0=149.3, rate=1.0, sample_rate=16000)
        args.update(kw)
        with pytest.raises(SynthRefused) as exc:
            dsp.synth(args["phone_ids"], args["durations"], args["f0"], args["rate"], args["sample_rate"], clear=True)
        assert exc.value.code == code, kw


def test_bad_glide_row_is_refused_with_code_10(dsp):
    table = dsp.synth_phone_table()
    ids = [table["SIL"][0], table["AY"][0], table["SIL"][0]]
    good = (0.0,) * 8
    for bad in ((676, 1756, 2631, 418, 1920, 2605, 0.8, 0.2),   # start after end
                (676, 1756, 2631, 418, 1920, 9000, 0.2, 0.8),   # above half the sample rate
                (676, 0.0, 2631, 418, 1920, 2605, 0.2, 0.8),    # a zero frequency in a glide
                (676, 1756, 2631, 418, 1920, 2605, -0.1, 0.8)):
        with pytest.raises(SynthRefused) as exc:
            dsp.synth(ids, [0.05, 0.2, 0.05], 149.3, 1.0, 16000, clear=True, glides=[good, bad, good])
        assert exc.value.code == 10, bad
    with pytest.raises(ValueError):
        dsp.synth(ids, [0.05, 0.2, 0.05], 149.3, 1.0, 16000, glides=[good, good, good])  # glides need clear


# ------------------------------------------------------ what reaches the ear


def test_a_held_vowel_is_its_resonances_not_the_swell(dsp, voice):
    """
    The old renderer's resonators (gain 1 - r, synth.ts:122-129) let the
    vowel's formants cancel and the voicing's low end win. Measured on his
    Mac on 2026-10-28, a held AA's spectral centroid: christman_synth_voiced
    334 Hz, christman_synth_clear 708 Hz (AA's F1 is 730 Hz). Held here: the
    clear AA above 450 Hz and at least 1.8 times the old one.
    """
    phones = [("SIL", 0.05), ("AA", 0.5), ("SIL", 0.05)]
    clear = cs_synth.synthesize(phones=phones, voice=voice, engine=dsp)
    table = dsp.synth_phone_formants()
    rows = voice.formant_rows([p for p, _ in phones], table)
    ids = [dsp.synth_phone_table()[p][0] for p, _ in phones]
    old = dsp.synth(ids, [d for _, d in phones], voice.f0_median_hz, voice.rate, 16000, None, rows)
    ph = clear.phones[1]
    c_new = _centroid(list(clear.samples), 16000, ph.start + 0.05, ph.start + ph.dur - 0.05)
    c_old = _centroid(list(old), 16000, ph.start + 0.05, ph.start + ph.dur - 0.05)
    print(f"held AA spectral centroid: christman_synth_voiced {c_old:.0f} Hz, christman_synth_clear {c_new:.0f} Hz")
    assert c_new > 450
    assert c_new >= 1.8 * c_old


@pytest.mark.parametrize("vowel", ["UW", "IY", "AE"])
def test_measured_vowels_sit_at_the_voices_formants(dsp, vowel, voice):
    want = voice.vowel_formants[vowel]
    r = cs_synth.synthesize(phones=[("SIL", 0.05), (vowel, 0.5), ("SIL", 0.05)], voice=voice, engine=dsp)
    ph = r.phones[1]
    a = measure_formants(list(r.samples), r.sample_rate)
    fr = [f for f in a.frames if ph.start + 0.06 <= f.time_s <= ph.start + ph.dur - 0.03
          and f.f1_hz and f.f2_hz and f.f3_hz]
    assert len(fr) >= 10, len(fr)
    got = tuple(statistics.median(getattr(f, k) for f in fr) for k in ("f1_hz", "f2_hz", "f3_hz"))
    print(f"{vowel}: measured {want}, christman_synth_clear {tuple(round(x) for x in got)}")
    # F1 within 15 % (few harmonics under it at 149 Hz, as formants.py says), F2 and F3 within 8 %.
    assert abs(got[0] - want[0]) / want[0] <= 0.15
    assert abs(got[1] - want[1]) / want[1] <= 0.08
    assert abs(got[2] - want[2]) / want[2] <= 0.08


def test_s_hisses_high_and_sh_lower(dsp, voice):
    cents = {}
    for fric in ("S", "SH"):
        r = cs_synth.synthesize(phones=[("SIL", 0.05), ("AA", 0.15), (fric, 0.3), ("AA", 0.15), ("SIL", 0.05)],
                                voice=voice, engine=dsp)
        ph = r.phones[2]
        cents[fric] = _centroid(list(r.samples), 16000, ph.start + 0.04, ph.start + ph.dur - 0.04)
    print(f"spectral centroid: S {cents['S']:.0f} Hz, SH {cents['SH']:.0f} Hz")
    assert cents["S"] > 4000
    assert 2000 < cents["SH"] < cents["S"]


def test_ay_glides_from_the_measured_onset_to_offset(dsp, voice):
    assert "AY" in voice.glides, "the profile carries the measured AY glide"
    (on, off) = voice.glides["AY"]
    r = cs_synth.synthesize(phones=[("SIL", 0.05), ("AY", 0.6), ("SIL", 0.05)], voice=voice, engine=dsp)
    ph = r.phones[1]
    a = measure_formants(list(r.samples), r.sample_rate)
    fr = [f for f in a.frames if ph.start <= f.time_s <= ph.start + ph.dur and f.f1_hz and f.f2_hz]
    first = [f for f in fr if f.time_s < ph.start + 0.15 * ph.dur]
    last = [f for f in fr if f.time_s > ph.start + 0.85 * ph.dur]
    assert first and last
    f1_on = statistics.median(f.f1_hz for f in first)
    f1_off = statistics.median(f.f1_hz for f in last)
    f2_on = statistics.median(f.f2_hz for f in first)
    f2_off = statistics.median(f.f2_hz for f in last)
    print(f"AY measured onset {on} offset {off}; synthesized F1 {f1_on:.0f}->{f1_off:.0f}, F2 {f2_on:.0f}->{f2_off:.0f}")
    assert f1_on > f1_off + 100      # the jaw closes: F1 falls
    assert f2_off > f2_on + 80       # the tongue fronts: F2 rises
    assert abs(f2_on - on[1]) / on[1] <= 0.08 and abs(f2_off - off[1]) / off[1] <= 0.08


def test_a_measured_voice_is_rendered_clear_and_says_so(dsp, voice):
    r = cs_synth.synthesize(phones=[("SIL", 0.05), ("AA", 0.2), ("SIL", 0.05)], voice=voice, engine=dsp)
    assert r.renderer == cs_synth.RENDERER_CLEAR and r.voice == TEST_VOICE_NAME
    r = cs_synth.render_codecwar_table(phones=[("SIL", 0.05), ("AA", 0.2), ("SIL", 0.05)], f0=149.3, rate=1.0, engine=dsp)
    assert r.renderer == cs_synth.RENDERER_CODECWAR and r.voice is None


# ------------------------------------------------------------------- words

SENTENCES = [
    "i want to go home", "please help me", "i am hungry", "can you open the door",
    "thank you very much", "where is my mother", "turn off the light", "i feel sick today",
    "good morning how are you", "the dog is sleeping on the bed", "my head hurts",
    "i would like some water please",
]


def _word_errors(ref: str, hyp: str):
    r, h = ref.split(), hyp.split()
    d = [[i + j if i * j == 0 else 0 for j in range(len(h) + 1)] for i in range(len(r) + 1)]
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)], len(r)


def test_free_text_is_heard_word_for_word_far_more_often(dsp, tmp_path, voice):
    """
    Twelve sentences in the test voice, recognized by Everett's Vosk
    (vosk-model-en-us-0.22, run directly). Measured on his Mac on 2026-10-28:
    christman_synth_voiced 3 of 53 words (5.7%); christman_synth_clear 25 of
    53 (47.2%). Held here: the clear renderer at least 40% and at least 30
    points above the old one. Not yet clear speech: about half the words. Needs CHRISTMAN_VOSK_MODEL,
    CHRISTMAN_LEXICON_PATH and CHRISTMAN_VOSK_PYTHON (a Python with vosk).
    """
    need = {k: os.environ.get(k) for k in ("CHRISTMAN_VOSK_MODEL", "CHRISTMAN_LEXICON_PATH", "CHRISTMAN_VOSK_PYTHON")}
    missing = [k for k, v in need.items() if not v]
    if missing:
        pytest.skip("not run: needs " + ", ".join(missing) +
                    " (Everett's Vosk model, its lexicon, a Python with vosk); runs on Everett's Mac")
    require_full_model(need["CHRISTMAN_VOSK_MODEL"])
    table = dsp.synth_phone_table()
    formant_table = dsp.synth_phone_formants()
    wavs = {"clear": [], "voiced": []}
    for i, s in enumerate(SENTENCES):
        r = cs_synth.synthesize(s, voice=voice, lexicon_path=need["CHRISTMAN_LEXICON_PATH"], engine=dsp)
        p = tmp_path / f"clear_{i}.wav"
        r.write_wav(p)
        wavs["clear"].append(str(p))
        names = [ph.id for ph in r.phones]
        durs = [ph.dur * r.rate for ph in r.phones]
        old = dsp.synth([table[n][0] for n in names], durs, voice.f0_median_hz, r.rate, r.sample_rate,
                        list(r.f0), voice.formant_rows(names, formant_table))
        p = tmp_path / f"voiced_{i}.wav"
        cs_synth.write_wav(p, old, r.sample_rate)
        wavs["voiced"].append(str(p))
    hear = (
        "import json, sys, wave, vosk\n"
        "vosk.SetLogLevel(-1)\n"
        "m = vosk.Model(sys.argv[1])\n"
        "out = []\n"
        "for p in sys.argv[2:]:\n"
        "    w = wave.open(p, 'rb'); sr = w.getframerate(); pcm = w.readframes(w.getnframes())\n"
        "    rec = vosk.KaldiRecognizer(m, sr); rec.AcceptWaveform(pcm + b'\\x00\\x00' * (sr // 2))\n"
        "    out.append(json.loads(rec.FinalResult()).get('text', ''))\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run([need["CHRISTMAN_VOSK_PYTHON"], "-c", hear, need["CHRISTMAN_VOSK_MODEL"]]
                          + wavs["clear"] + wavs["voiced"], capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stderr[-2000:]
    heard = json.loads(proc.stdout.strip().splitlines()[-1])
    acc = {}
    for k, part in (("clear", heard[:len(SENTENCES)]), ("voiced", heard[len(SENTENCES):])):
        errs = total = 0
        for ref, hyp in zip(SENTENCES, part):
            e, n = _word_errors(ref, hyp)
            errs += e
            total += n
            print(f"{k:6} {ref!r:36} heard {hyp!r}")
        acc[k] = 1 - errs / total
    print(f"word accuracy: christman_synth_voiced {acc['voiced']:.1%}, christman_synth_clear {acc['clear']:.1%}")
    assert acc["clear"] >= 0.40
    assert acc["clear"] >= acc["voiced"] + 0.30
