# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
The mouth wired to Everett's synthesizer (2026-10-18):
SpeechSynthesisEngine._render -> christman_sound.lexicon -> christman_synth
-> samples -> the cache file -> christman_play.

Real runs only. The C library is built with the repository's own
scripts/build_dsp.py at the start of the module and installed as the shared
loader's instance, so the mouth reaches it the way it does in use. Pitch is
measured from the output with the package's own christman_yin. A test that
cannot run here reports NOT RUN with the reason (tests/conftest.py).

Pitch tolerance, taken from tests/test_sovereign_synth.py and its stated
reasons (Codec War's +/-1.2 % vibrato; christman_yin's whole-number lag):
    MEDIAN of voiced frames: within f0 / (sr - f0) + 0.3 %
    EVERY voiced frame:      within 3 %
Frames are 2048 samples with a 512-sample hop, as in the synth tests. A
phrase is not a sustained vowel: frames over a consonant or a silence come
back unvoiced from christman_yin (0.0) or outside 50-600 Hz, and are set
aside as pitch_contour does. The count of voiced frames is printed and must
be at least 20.
"""

from __future__ import annotations

import array
import math
import os
import statistics
import subprocess
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import christman_sound.dsp as dsp_mod  # noqa: E402
from christman_sound.dsp import ChristmanDSP  # noqa: E402
from christman_sound.lexicon import (  # noqa: E402
    LexiconUnavailable,
    PhoneNotMapped,
    WordNotInLexicon,
)
from christman_sound import synth as cs_synth  # noqa: E402
from christman_voice_sdk.engines.synthesis_result import SynthesisUnavailable  # noqa: E402
from christman_voice_sdk.synthesis import voice_synthesis as vs  # noqa: E402

# The same inline align_lexicon.txt format as vosk-model-en-us-0.22. The first
# two lines are the ones quoted from Everett's file in christman_sound/lexicon.py.
FIXTURE_LEXICON = """\
christman christman k_B r_I I_I s_I t_I m_I V_I n_E
hello hello h_B @_I l_I oU_E
booth booth b_B u_I T_E
uh uh UHH_S
"""

PHRASE = "Hello, Christman. Hello, Christman."
F0 = 149.3


@pytest.fixture(scope="module")
def built_dsp(tmp_path_factory) -> ChristmanDSP:
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
def dsp(built_dsp, monkeypatch) -> ChristmanDSP:
    """The built library as the shared loader's instance, as the mouth reaches it in use."""
    monkeypatch.setattr(dsp_mod, "_instance", built_dsp)
    return built_dsp


@pytest.fixture()
def lexicon_file(tmp_path) -> Path:
    path = tmp_path / "align_lexicon.txt"
    path.write_text(FIXTURE_LEXICON, encoding="utf-8")
    return path


@pytest.fixture()
def mouth(dsp, lexicon_file, tmp_path) -> vs.SpeechSynthesisEngine:
    return vs.SpeechSynthesisEngine(cache_dir=str(tmp_path / "cache"), lexicon_path=str(lexicon_file))


def _voiced_yin(dsp, samples, sr, frame=2048, hop=512):
    est = [dsp.yin(samples[i:i + frame], sr) for i in range(0, len(samples) - frame, hop)]
    return est, [e for e in est if 50.0 <= e <= 600.0]


# ------------------------------------------------------------------ _render

def test_render_returns_real_samples_from_christman_synth(mouth, dsp, lexicon_file):
    r = mouth._render(PHRASE, None, "en")
    assert r.engine == "christman"
    assert isinstance(r.audio, array.array) and r.audio.typecode == "f"
    assert r.sample_rate == cs_synth.CODECWAR_SAMPLE_RATE
    assert r.degraded is False and r.error_reason is None
    # The same samples christman_synth gives for this text at this pitch: the
    # mouth adds nothing and takes nothing away.
    direct = cs_synth.synthesize(PHRASE, f0=F0, lexicon_path=lexicon_file, engine=dsp)
    assert list(r.audio) == list(direct.samples)
    assert abs(r.duration - len(r.audio) / r.sample_rate) < 1e-12
    peak = max(abs(s) for s in r.audio)
    rms = math.sqrt(sum(s * s for s in r.audio) / len(r.audio))
    print(f"_render: {len(r.audio)} samples ({r.duration:.3f} s) at {r.sample_rate} Hz, "
          f"peak {peak:.4f}, rms {rms:.4f}, synthesis {r.synthesis_time * 1000:.1f} ms")
    assert abs(peak - 0.7) < 1e-6  # synth.ts:83 normalises the peak to 0.7: it spoke
    assert rms > 0.05


def test_f0_constant_is_everetts_measured_median_with_its_source():
    assert vs.EVERETT_F0_MEDIAN_HZ == 149.3
    src = (ROOT / "christman_voice_sdk" / "synthesis" / "voice_synthesis.py").read_text(encoding="utf-8")
    at = src.index("\nEVERETT_F0_MEDIAN_HZ = 149.3")  # the assignment, not the docstring mention
    beside = src[max(0, at - 400):at]
    for words in ("26 recorded phrases", "alphavox_verified", "christman_yin", "formants.py",
                  "2026-10-18", "per-phrase contours"):
        assert words in beside, words


def test_output_pitch_is_149_3_hz_by_christman_yin(mouth, dsp):
    r = mouth._render(PHRASE, None, "en")
    sr = r.sample_rate
    est, voiced = _voiced_yin(dsp, r.audio, sr)
    median = statistics.median(voiced)
    median_tol = F0 / (sr - F0) + 0.003
    worst = max(abs(e - F0) / F0 for e in voiced)
    print(f"requested {F0} Hz at {sr} Hz: {len(voiced)} voiced of {len(est)} frames, "
          f"median {median:.2f} Hz ({(median - F0) / F0 * 100:+.2f} %, tol {median_tol * 100:.2f} %), "
          f"worst voiced frame {worst * 100:.2f} %")
    assert len(voiced) >= 20
    assert abs(median - F0) / F0 <= median_tol
    assert worst <= 0.03, voiced


def test_generate_speech_audio_writes_the_samples_and_reports_christman(mouth, tmp_path):
    out = tmp_path / "out.wav"
    path = mouth.generate_speech_audio(PHRASE, play_audio=False, output_path=str(out))
    assert path == str(out)
    assert mouth.last_engine == "christman" and mouth.last_error is None
    r = mouth._render(PHRASE, None, "en")
    with wave.open(str(out), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, r.sample_rate)
        assert w.getnframes() == len(r.audio)
    samples, rate, channels = vs._read_wav(out)
    assert (rate, channels, len(samples)) == (r.sample_rate, 1, len(r.audio))
    # 16-bit round trip: every sample within half a step of what the synth made.
    assert max(abs(a - b) for a, b in zip(samples, r.audio)) <= 0.5 / 32767 + 1 / 32768


def test_readiness_is_true_only_with_library_and_lexicon(dsp, lexicon_file, monkeypatch):
    ready, reason = vs.synthesis_readiness(str(lexicon_file))
    assert ready is True and reason.startswith("christman:") and "149.3" in reason
    monkeypatch.delenv("CHRISTMAN_LEXICON_PATH", raising=False)
    ready, reason = vs.synthesis_readiness()
    assert ready is False and "no lexicon configured" in reason


# ------------------------------------------------------------------ refusals

def test_missing_word_raises_the_typed_error_through_every_public_entry_point(mouth, dsp, lexicon_file,
                                                                               tmp_path, monkeypatch):
    """WordNotInLexicon, unchanged, naming the words; nothing spelled, nothing written."""
    text = "hello zebra quokka"
    cache = Path(mouth.cache_dir)

    def check(excinfo):
        err = excinfo.value
        assert type(err) is WordNotInLexicon, type(err)
        assert not isinstance(err, SynthesisUnavailable), "not wrapped"
        assert err.words == ["zebra", "quokka"]

    with pytest.raises(WordNotInLexicon) as e1:
        mouth._render(text, None, "en")
    check(e1)
    for play in (False, True):
        with pytest.raises(WordNotInLexicon) as e2:
            mouth.generate_speech_audio(text, play_audio=play)
        check(e2)
        assert mouth.last_engine == "none" and "zebra" in (mouth.last_error or "")
    with pytest.raises(WordNotInLexicon) as e3:
        mouth.generate_emotion_adjusted_speech(text)
    check(e3)

    # Module-level synthesize_speech, through the shared engine and the env var.
    monkeypatch.setenv("CHRISTMAN_LEXICON_PATH", str(lexicon_file))
    monkeypatch.setattr(vs, "DEFAULT_CACHE_DIR", cache)
    monkeypatch.setattr(vs, "_speech_synthesis_engine", None)
    with pytest.raises(WordNotInLexicon) as e4:
        vs.synthesize_speech(text)
    check(e4)

    # christman_sound's own entry points: synthesize, and speak (which raises
    # before it reaches the speaker, so this runs without an audio device).
    with pytest.raises(WordNotInLexicon) as e5:
        cs_synth.synthesize(text, f0=F0, lexicon_path=lexicon_file, engine=dsp)
    check(e5)
    with pytest.raises(WordNotInLexicon) as e6:
        cs_synth.speak(text, f0=F0, lexicon_path=lexicon_file, engine=dsp)
    check(e6)

    assert not cache.exists() or list(cache.iterdir()) == [], "no WAV for words that were not voiced"


def test_missing_lexicon_and_non_speech_phone_surface_unchanged(dsp, lexicon_file, tmp_path):
    no_lex = vs.SpeechSynthesisEngine(cache_dir=str(tmp_path / "c1"),
                                      lexicon_path=str(tmp_path / "no_such_lexicon.txt"))
    with pytest.raises(LexiconUnavailable) as e1:
        no_lex.generate_speech_audio("hello", play_audio=False)
    assert type(e1.value) is LexiconUnavailable
    good = vs.SpeechSynthesisEngine(cache_dir=str(tmp_path / "c2"), lexicon_path=str(lexicon_file))
    with pytest.raises(PhoneNotMapped) as e2:
        good.generate_speech_audio("uh", play_audio=False)
    assert type(e2.value) is PhoneNotMapped and e2.value.phone == "UHH"
    with pytest.raises(cs_synth.NothingToSay):
        good.generate_speech_audio(" ?! ", play_audio=False)
    for d in ("c1", "c2"):
        assert not (tmp_path / d).exists() or list((tmp_path / d).iterdir()) == []


def test_other_language_is_refused_not_spoken_in_english(mouth):
    with pytest.raises(vs.NothingVoiced) as excinfo:
        mouth.generate_speech_audio("hello", language="es", play_audio=False)
    assert "English" in excinfo.value.reason and "'es'" in excinfo.value.reason
    assert mouth._render("hello", None, "en-US").engine == "christman"


def test_emotion_params_are_not_applied_and_that_is_recorded(mouth):
    plain = mouth._render("hello", None, "en")
    assert mouth.last_emotion_unapplied is None
    asked = {"emotion": "happy", "exaggeration": 0.5}
    moved = mouth._render("hello", asked, "en")
    assert mouth.last_emotion_unapplied == asked
    assert list(moved.audio) == list(plain.audio), "nothing invented from the emotion"


# ------------------------------------------------------------------ sovereignty

def test_mouth_path_imports_no_outside_audio_or_numeric_package(built_dsp, lexicon_file, tmp_path):
    """
    Text -> lexicon -> christman_synth -> cache WAV -> the reader that feeds
    christman_play, with numpy, scipy, soundfile, sounddevice, pygame and
    librosa made unimportable.
    """
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
import os
os.environ["CHRISTMAN_DSP_LIB"] = {str(built_dsp.path)!r}
from pathlib import Path
from christman_voice_sdk.synthesis import voice_synthesis as vs
eng = vs.SpeechSynthesisEngine(cache_dir={str(tmp_path / "cache")!r}, lexicon_path={str(lexicon_file)!r})
path = eng.generate_speech_audio("hello christman", play_audio=False)
samples, rate, ch = vs._read_wav(Path(path))
print(eng.last_engine, len(samples), sorted(m for m in sys.modules if m.split(".")[0] in BLOCK))
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    engine, count, loaded = proc.stdout.strip().splitlines()[-1].split(" ", 2)
    assert engine == "christman" and int(count) > 0 and loaded == "[]", proc.stdout


def test_stdlib_wav_reader_gives_the_numpy_readers_values(tmp_path):
    """The reader moved off numpy; every value must be the float32 numpy gave."""
    try:
        import numpy as np
    except ImportError:
        pytest.skip("not run: numpy not installed here; it is needed only to compute the old reader's values")
    rng = np.random.default_rng(20261018)
    cases = {
        2: rng.integers(-32768, 32767, 4000, dtype=np.int16),
        4: rng.integers(-2**31, 2**31 - 1, 4000, dtype=np.int32),
        1: rng.integers(0, 255, 4000, dtype=np.uint8),
    }
    for width, data in cases.items():
        path = tmp_path / f"w{width}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(width)
            w.setframerate(16000)
            w.writeframes(data.astype(data.dtype.newbyteorder("<")).tobytes())
        if width == 2:
            old = data.astype(np.float32) / 32768.0
        elif width == 4:
            old = data.astype(np.float32) / 2147483648.0
        else:
            old = (data.astype(np.float32) - 128.0) / 128.0
        new, rate, ch = vs._read_wav(path)
        assert (rate, ch) == (16000, 1)
        assert np.array_equal(np.frombuffer(new.tobytes(), dtype=np.float32), old.astype(np.float32)), width
    # IEEE float WAV (format tag 3), the kind the stdlib wave module refuses.
    import struct
    f = rng.uniform(-1, 1, 3000).astype("<f4")
    body = f.tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, 24000, 24000 * 4, 4, 32)
    riff = b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(body)) + b"WAVE"
    riff += b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(body)) + body
    fpath = tmp_path / "float.wav"
    fpath.write_bytes(riff)
    new, rate, ch = vs._read_wav(fpath)
    assert (rate, ch) == (24000, 1)
    assert np.array_equal(np.frombuffer(new.tobytes(), dtype=np.float32), f.astype(np.float32))


# ------------------------------------------------------------------ the service

def test_tts_service_voices_and_never_serves_another_voices_file(dsp, lexicon_file, tmp_path, monkeypatch):
    try:
        import flask  # noqa: F401
    except ImportError:
        pytest.skip("not run: flask not installed in this environment")
    from christman_voice_sdk.synthesis import tts_service as ts

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHRISTMAN_LEXICON_PATH", str(lexicon_file))
    monkeypatch.setattr(vs, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(vs, "_speech_synthesis_engine", None)
    # A file at the key the service used before 2026-10-18, as an old renderer
    # would have left it. It must not be served as this voice.
    audio_dir = tmp_path / "static" / "audio"
    audio_dir.mkdir(parents=True)
    old_key = ts._hash_content("hello", "None", "en", "{}")
    (audio_dir / f"{old_key}.wav").write_bytes(b"not this voice")

    client = ts.app.test_client()
    status = client.get("/status").get_json()
    assert status["synthesis_available"] is True and status["synthesis_reason"] is None
    r = client.post("/generate", json={"text": "hello"})
    assert r.status_code == 200, r.get_json()
    url = r.get_json()["audio_url"]
    assert url != f"/audio/{old_key}.wav"
    with wave.open(str(audio_dir / url.rsplit("/", 1)[1]), "rb") as w:
        assert w.getnframes() > 0 and w.getframerate() == 16000

    r = client.post("/generate", json={"text": "hello zebra"})
    assert r.status_code == 422
    body = r.get_json()
    assert body["available"] is False and body["kind"] == "WordNotInLexicon" and body["words"] == ["zebra"]

    monkeypatch.setenv("CHRISTMAN_LEXICON_PATH", str(tmp_path / "gone.txt"))
    r = client.post("/generate", json={"text": "booth"})
    assert r.status_code == 503 and r.get_json()["available"] is False


# ------------------------------------------------------------------ playback

def test_generate_speech_audio_plays_through_christman_play(mouth, dsp):
    try:
        mouth.generate_speech_audio("hello", play_audio=True)
    except vs.PlaybackFailed as exc:
        pytest.skip(f"not run: no speaker reachable here ({dsp.play_backend()}); "
                    f"playback is checked on Everett's Mac: {exc}")
    mouth.stop_audio()
    assert mouth.last_engine == "christman" and mouth.last_playback_error is None
