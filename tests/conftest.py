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
"""

from __future__ import annotations

NOT_RUN_PREFIX = "not run:"


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
