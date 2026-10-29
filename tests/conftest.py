# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Test reporting rules for this repository (Everett, 2026-10-11).

A test that cannot run in the current environment must say so. It never
passes, and it never disappears into a bare "s". Such a test calls

    pytest.skip("not run: <what is missing and where it does run>")

and this file prints every one of them under a NOT RUN banner at the end of
the run, whatever flags pytest was given. pyproject.toml also sets `-ra`,
so pytest's own summary lists the reason as well. Two places, on purpose.

NOTHING PLAYS OUT LOUD UNLESS ASKED (2026-10-29)
------------------------------------------------
Test runs on Everett's Mac played speech through his speakers. Now a test
that plays through the speakers must carry `@pytest.mark.plays_aloud`, and
it runs only when CHRISTMAN_SOUND_PLAY_ALOUD=1 is set; otherwise it is NOT
RUN, with that reason. Every other test is guarded: if it reaches the real
christman_play (the C entry that drives the speakers), it fails at once,
before any sample is played, naming the marker it needs. A test that swaps
in its own christman_play (tests/test_playback_honest.py) is not stopped,
because nothing it plays reaches a speaker.
"""

from __future__ import annotations

import ctypes
import os

import pytest

NOT_RUN_PREFIX = "not run:"
PLAY_ALOUD_ENV = "CHRISTMAN_SOUND_PLAY_ALOUD"
PLAY_ALOUD_MARK = "plays_aloud"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        f"{PLAY_ALOUD_MARK}: plays sound through the speakers; runs only when {PLAY_ALOUD_ENV}=1",
    )


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    if os.environ.get(PLAY_ALOUD_ENV) == "1":
        return
    skip = pytest.mark.skip(
        reason=f"{NOT_RUN_PREFIX} plays through the speakers; set {PLAY_ALOUD_ENV}=1 to run it")
    for item in items:
        if item.get_closest_marker(PLAY_ALOUD_MARK):
            item.add_marker(skip)


def _drives_the_speakers(lib) -> bool:
    """True when lib.christman_play is the library's own entry, not a test's stand-in."""
    try:
        own = ctypes.CDLL(lib._name).christman_play
    except (OSError, AttributeError):
        return False
    here = ctypes.cast(lib.christman_play, ctypes.c_void_p).value
    return here == ctypes.cast(own, ctypes.c_void_p).value


@pytest.fixture(autouse=True)
def _speakers_only_when_marked(request, monkeypatch):
    if request.node.get_closest_marker(PLAY_ALOUD_MARK):
        yield
        return
    try:
        from christman_sound import dsp as dsp_mod
    except ImportError:
        yield
        return
    play = dsp_mod.ChristmanDSP.play

    def guarded(self, *args, **kwargs):
        if _drives_the_speakers(self._require()):
            pytest.fail(
                f"{request.node.nodeid} reached the speakers (christman_play) without "
                f"@pytest.mark.{PLAY_ALOUD_MARK}. Render to a file (play_audio=False), or mark "
                f"the test so it runs only when {PLAY_ALOUD_ENV}=1.", pytrace=False)
        return play(self, *args, **kwargs)

    monkeypatch.setattr(dsp_mod.ChristmanDSP, "play", guarded)
    yield


def _skip_message(report) -> str:
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # noqa: ARG001
    not_run = []
    for report in terminalreporter.stats.get("skipped", []):
        message = _skip_message(report)
        if NOT_RUN_PREFIX in message:
            reason = message.split(NOT_RUN_PREFIX, 1)[1].strip()
            not_run.append((report.nodeid, reason))
    if not not_run:
        return
    terminalreporter.write_sep(
        "=", f"NOT RUN: {len(not_run)} test(s) made no claim in this environment"
    )
    for nodeid, reason in not_run:
        terminalreporter.write_line(f"NOT RUN  {nodeid}")
        terminalreporter.write_line(f"         {reason}")
