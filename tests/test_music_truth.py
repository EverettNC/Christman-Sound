# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
music/ reports only what it does (Lane 6, 2026-10-28).

christman_studio.py builds arrangement data and makes no sound. It used to
return "quality": "professional" from a mix that was never made and
{"lufs": -14.0, ...} from a master that measured nothing. music_engine.py
had a sing_melody() that sang nothing. Mixing, mastering and singing are
not built, and the code says so (NOT_BUILT) instead of claiming them.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MUSIC = ROOT / "christman_voice_sdk" / "music"


def test_no_claimed_quality_or_loudness_anywhere_in_music():
    for path in MUSIC.glob("*.py"):
        if path.name == "resonance_player.py":
            continue  # measures its levels for real (peak and RMS in dBFS)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
                      and n.body and isinstance(n.body[0], ast.Expr)
                      and isinstance(n.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                low = node.value.lower()
                assert "professional" not in low, f"{path.name}:{node.lineno} {node.value!r}"
                assert "lufs" not in low, f"{path.name}:{node.lineno} {node.value!r}"


def test_studio_says_what_is_not_built_and_has_no_mix_or_master(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from christman_voice_sdk.music import christman_studio as cs

    assert {"mixing", "mastering", "singing"} <= set(cs.NOT_BUILT)
    studio = cs.ChristmanMusicStudio()
    assert not hasattr(studio, "mix_project") and not hasattr(studio, "master_track")
    assert list(tmp_path.iterdir()) == [], "constructing the studio made a folder"

    stats = studio.get_studio_stats()
    assert stats["audio_rendered"] is False and "mixing" in stats["not_built"]
    assert "studio_settings" not in stats and "sample_library_size" not in stats


def test_create_beat_returns_the_arrangement_with_the_callers_velocity():
    from christman_voice_sdk.music import christman_studio as cs

    project = cs.create_beat("test", "electronic")
    assert project["audio_rendered"] is False
    hits = project["tracks"]["Drums"]["notes"]
    assert len(hits) == 4 * 4  # four hits a bar, four bars
    assert {h["velocity"] for h in hits} == {100}

    studio = cs.ChristmanMusicStudio()
    studio.create_project("p")
    out = studio.program_beat("Drums", [1, 0, 1, 0], bars=1, velocity=90)
    assert {b["velocity"] for b in out["beats"]} == {90} and len(out["beats"]) == 8


def test_music_engine_does_not_claim_to_sing_and_writes_only_where_told(tmp_path, monkeypatch):
    monkeypatch.setenv("CHRISTMAN_AI_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    from christman_voice_sdk.music.music_engine import ChristmanMusicEngine

    engine = ChristmanMusicEngine()
    assert not hasattr(engine, "sing_melody")
    assert sorted(p.name for p in tmp_path.iterdir()) == [], "construction made a folder"

    melody = engine.generate_melody("calm", length=4)
    assert len(melody) == 4
    saved = tmp_path / "home" / "voice_sdk" / "music" / "musical_memory.json"
    assert saved.is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["home"]
