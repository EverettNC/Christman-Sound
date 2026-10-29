# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Three safety fixes, each proven against the real code (Lane 6, 2026-10-28).

1. timbre/voicepack.py: a hostile voicepack zip cannot write outside the
   folder it unpacks into. Absolute paths, drive letters, ".." steps and
   symlinks are refused before anything is extracted.
2. audio/config.py: building a Config creates no folder, inside the package
   or anywhere. Folders are made under the SDK's data folder when asked.
3. audio/word_ear_receiver.py: no default port, never the package's own
   corti_receiver port, and never a port the caller names as held.
"""

from __future__ import annotations

import re
import socket
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ------------------------------------------------------------------ voicepack

def _voicepack():
    for module in ("torch",):
        try:
            __import__(module)
        except ImportError:
            pytest.skip(f"not run: {module} is not installed here; voicepack imports timbre_modeler, which imports it")
    from christman_voice_sdk.timbre import voicepack
    return voicepack


def _zip(path: Path, members: dict[str, bytes], symlink: str | None = None) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, "/etc/passwd")
    return path


HOSTILE = {
    "dotdot": "../escaped.txt",
    "nested_dotdot": "reference_audio/../../escaped.txt",
    "backslash_dotdot": "reference_audio\\..\\..\\escaped.txt",
    "absolute": "/tmp/christman_voicepack_escape.txt",
    "backslash_absolute": "\\escaped.txt",
    "drive": "C:/escaped.txt",
}


@pytest.mark.parametrize("label", sorted(HOSTILE))
def test_hostile_voicepack_is_refused_and_nothing_is_written(label, tmp_path):
    vp = _voicepack()
    packs = tmp_path / "home" / "voicepacks"
    packs.mkdir(parents=True)
    member = HOSTILE[label]
    archive = _zip(tmp_path / f"{label}.voicepack",
                   {"metadata.json": b"{}", member: b"written where it should not be"})

    builder = vp.VoicepackBuilder(output_dir=packs)
    with pytest.raises(vp.UnsafeVoicepack) as info:
        builder.load(archive)
    assert repr(member) in str(info.value)

    cache = packs.parent / "cache" / archive.stem
    assert list(cache.iterdir()) == [], "a member was extracted before the archive was refused"
    assert not (tmp_path / "escaped.txt").exists()
    assert not (packs.parent / "escaped.txt").exists()
    assert not Path("/tmp/christman_voicepack_escape.txt").exists()


def test_symlink_member_is_refused(tmp_path):
    vp = _voicepack()
    packs = tmp_path / "voicepacks"
    archive = _zip(tmp_path / "link.voicepack", {"metadata.json": b"{}"}, symlink="reference_audio/link")
    with pytest.raises(vp.UnsafeVoicepack, match="symlink"):
        vp.VoicepackBuilder(output_dir=packs).load(archive)


def test_ordinary_members_pass_the_check(tmp_path):
    vp = _voicepack()
    archive = _zip(tmp_path / "ok.voicepack", {
        "metadata.json": b"{}",
        "voice_profile.json": b"{}",
        "reference_audio/sample_000.wav": b"RIFF",
        "emotion_models/m.pt": b"x",
    })
    with zipfile.ZipFile(archive) as z:
        vp.check_members(z, tmp_path / "dest")  # does not raise


def test_voicepack_builder_creates_nothing_when_built(tmp_path, monkeypatch):
    vp = _voicepack()
    monkeypatch.setenv("CHRISTMAN_AI_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    builder = vp.VoicepackBuilder()
    assert builder.output_dir == tmp_path / "home" / "voice_sdk" / "voicepacks"
    assert sorted(p.name for p in tmp_path.iterdir()) == [], "construction made a folder"


def test_voicepack_metadata_does_not_claim_measurements():
    vp = _voicepack()
    meta = vp.VoicepackMetadata(name="x")
    assert meta.training_hours is None and meta.quality_score is None


# --------------------------------------------------------------------- config

def test_config_creates_no_folder(tmp_path, monkeypatch):
    from christman_voice_sdk.audio import config as cfg

    made = []
    real_mkdir = Path.mkdir

    def watch(self, *args, **kwargs):
        made.append(self)
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setenv("CHRISTMAN_AI_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "mkdir", watch)
    conf = cfg.get_config()
    assert made == [], f"get_config() created {made}"

    package = Path(cfg.__file__).resolve().parent.parent
    for d in (conf.models_dir, conf.data_dir, conf.logs_dir):
        assert package not in d.parents, f"{d} is inside the package"
        assert d.parent == tmp_path / "home" / "voice_sdk"
        assert not d.exists()

    made_dir = conf.ensure_dir(conf.models_dir)
    assert made_dir.is_dir() and made and made[0] == conf.models_dir
    assert all(d == conf.models_dir or d in conf.models_dir.parents for d in made)
    assert stat.S_IMODE(made_dir.stat().st_mode) == 0o700


def test_importing_the_capture_client_writes_nothing_into_the_package(tmp_path):
    """voice_capture_client runs get_config() at import; the package stays as it was."""
    package = ROOT / "christman_voice_sdk"
    before = {p.name for p in package.iterdir()}
    env = {"CHRISTMAN_AI_HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    proc = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
         "import christman_voice_sdk.integration.voice_capture_client"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    if proc.returncode != 0 and "No module named" in proc.stderr:
        pytest.skip(f"not run: a dependency of voice_capture_client is missing here: {proc.stderr.strip()[-200:]}")
    assert proc.returncode == 0, proc.stderr[-2000:]
    after = {p.name for p in package.iterdir()}
    assert after - before - {"__pycache__"} == set(), f"import created {sorted(after - before)}"
    assert not (tmp_path / "home").exists(), "import created the SDK's data folder"


def test_no_default_model_id_in_the_config():
    from christman_voice_sdk.audio import config as cfg

    assert "jonatasgrosman" not in repr(cfg.Config()._get_default_config())


# ------------------------------------------------------------------- word-ear

def test_word_ear_has_no_default_port(monkeypatch):
    from christman_voice_sdk.audio import word_ear_receiver as wer

    monkeypatch.delenv(wer.PORT_ENV, raising=False)
    assert not hasattr(wer, "DEFAULT_PORT")
    with pytest.raises(ValueError, match="no default port"):
        wer.resolve_port(None)


def test_word_ear_refuses_the_package_own_corti_port(monkeypatch):
    from christman_voice_sdk.audio import corti_receiver as cr
    from christman_voice_sdk.audio import word_ear_receiver as wer

    monkeypatch.delenv(wer.REFUSED_PORTS_ENV, raising=False)
    assert set(wer.PACKAGE_PORTS) == {cr.DEFAULT_PORT}
    with pytest.raises(ValueError, match="is taken by corti_receiver"):
        wer.resolve_port(cr.DEFAULT_PORT)
    monkeypatch.setenv(wer.PORT_ENV, str(cr.DEFAULT_PORT))
    with pytest.raises(ValueError, match="is taken by corti_receiver"):
        wer.resolve_port(None)


def test_word_ear_refuses_the_ports_the_caller_names(monkeypatch):
    from christman_voice_sdk.audio import word_ear_receiver as wer

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        held = s.getsockname()[1]
    monkeypatch.delenv(wer.REFUSED_PORTS_ENV, raising=False)
    assert wer.resolve_port(held) == held  # the package itself does not hold it
    with pytest.raises(ValueError, match=r"is taken by the caller \(refused_ports=\)"):
        wer.resolve_port(held, refused=[held])
    monkeypatch.setenv(wer.REFUSED_PORTS_ENV, f" {held} ,")
    with pytest.raises(ValueError, match="is taken by the caller"):
        wer.resolve_port(held)
    with pytest.raises(ValueError, match="is taken by the caller"):
        wer.create_receiver(port=held)
    monkeypatch.setenv(wer.REFUSED_PORTS_ENV, "not-a-port")
    with pytest.raises(ValueError, match="not a port number"):
        wer.resolve_port(held)


def test_word_ear_takes_the_port_it_is_given(monkeypatch):
    from christman_voice_sdk.audio import word_ear_receiver as wer

    monkeypatch.setenv(wer.PORT_ENV, "not-a-port")
    with pytest.raises(ValueError, match="not a port number"):
        wer.resolve_port(None)
    with pytest.raises(ValueError, match="not a port number"):
        wer.resolve_port(70000)

    monkeypatch.delenv(wer.REFUSED_PORTS_ENV, raising=False)
    with socket.socket() as s:  # a free loopback port, not one of the refused ones
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    monkeypatch.setenv(wer.PORT_ENV, str(free))
    assert wer.resolve_port(None) == free

    class NoEngine:
        available = False
        unavailable_reason = "test: no recognizer"

    server, _receiver = wer.create_receiver(port=0, engine=NoEngine())
    try:
        host, bound = server.server_address[:2]
        assert host == "127.0.0.1" and bound not in wer.refused_ports() and bound != 0
    finally:
        server.server_close()


# ---------------------------------------------------------------------- words

LANE6_FILES = [
    *sorted((ROOT / "christman_voice_sdk" / "utils").glob("*.py")),
    *sorted((ROOT / "christman_voice_sdk" / "music").glob("*.py")),
    ROOT / "christman_voice_sdk" / "timbre" / "shorty_emotion.py",
    ROOT / "christman_voice_sdk" / "timbre" / "voicepack.py",
    ROOT / "christman_voice_sdk" / "audio" / "config.py",
    ROOT / "christman_sound" / "__init__.py",
    ROOT / "christman_ocr_shared.py",
]


def test_the_people_served_are_clients_in_these_files():
    """Everett: clients, never "users". (word_ear_receiver reads the
    is_user_speech property of recognition_result.py, which is not this lane's.)"""
    pattern = re.compile(r"\busers?\b|\buser'?s\b|user_input|user_said", re.I)
    hits = []
    for path in LANE6_FILES:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line.replace("/Users/", "")):
                hits.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
    assert hits == [], "\n".join(hits)


def test_grounder_reads_the_one_breathing_number():
    from christman_voice_sdk.tone import emotion_quantifier
    from christman_voice_sdk.utils import grounder

    assert grounder._THRESHOLD_SOURCE == "emotion_quantifier.BREATHING_RISE"
    assert grounder.BREATHING_FIRST_THRESHOLD == emotion_quantifier.BREATHING_RISE
