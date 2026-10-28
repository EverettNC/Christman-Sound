# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
The README's file lists match the files that exist (Lane 6, 2026-10-28).

For every package section ("#### Audio (`audio/`)" and the rest) and for
christman_sound, each module listed exists, and each module that exists is
listed. The README never shows a person's name as a voice profile, and it
does not say speech is unavailable while the mouth is wired.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
SDK = ROOT / "christman_voice_sdk"


def _sections() -> dict[str, list[str]]:
    """Package folder -> the module names its README section lists as bullets."""
    found: dict[str, list[str]] = {}
    current = None
    for line in README.splitlines():
        head = re.match(r"^#### .*\(`(\w+)/`\)", line)
        if head:
            current = head.group(1)
            found[current] = []
            continue
        if line.startswith("#"):
            current = None
            continue
        if current and line.startswith("- "):
            found[current].extend(re.findall(r"`(\w+\.py)`", line.split(" — ")[0]))
    return found


def _modules(folder: Path) -> set[str]:
    return {p.name for p in folder.glob("*.py") if p.name != "__init__.py"}


def test_every_sdk_package_has_a_section():
    packages = {p.name for p in SDK.iterdir() if (p / "__init__.py").is_file()}
    assert set(_sections()) == packages


def test_each_section_lists_exactly_the_modules_that_exist():
    for folder, listed in _sections().items():
        assert set(listed) == _modules(SDK / folder), (
            f"{folder}/: listed but missing {sorted(set(listed) - _modules(SDK / folder))}; "
            f"present but unlisted {sorted(_modules(SDK / folder) - set(listed))}")


def test_christman_sound_section_matches():
    body = README.split("### christman_sound — the voice", 1)[1].split("\n### ", 1)[0]
    listed = set(re.findall(r"^- `(\w+\.py)`", body, flags=re.M))
    assert listed == _modules(ROOT / "christman_sound")


def test_c_sources_listed_are_the_ones_that_exist():
    body = README.split("### C engine", 1)[1].split("\n### ", 1)[0]
    listed = set(re.findall(r"^- `(\w+\.c)`", body, flags=re.M))
    assert listed == {p.name for p in ROOT.glob("*.c")}


def test_no_persons_voice_profile_and_no_stale_claims():
    assert "--profile everett" not in README and "--capture everett" not in README
    assert "unavailable until his own synthesizer is wired" not in README
    assert "Until the sovereign synthesizer" not in README
    assert "falls back to local TTS" not in README
    assert "ChristmanSpeechToSpeech" not in README
