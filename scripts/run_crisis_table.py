#!/usr/bin/env python3
# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Run Everett's crisis table through the hearing turn and print the numbers.

The table lives in tests/test_harm_frame_gate.py (MUST CALL A HUMAN /
MUST NOT CALL A HUMAN / his rulings). This script does not test the gate in
isolation — it sends every sentence through `Hearing.hear(text=...)`, the
same path a being uses, with a crisis callback that records whether a human
was called, and reports:

    turn_status   decision.safety.status from the full hearing turn
    gate_status   harm_frame_safety(sentence).status, the gate alone
    human_called  the crisis callback was reached

Expected: MUST CALL -> hard_block and human_called; MUST NOT CALL -> neither.

    python3 scripts/run_crisis_table.py            # table + counts, exit 1 on any mismatch
    python3 scripts/run_crisis_table.py --json out.json

Run it on org main after a merge and diff the JSON against the PR's run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_harm_frame_gate import (  # noqa: E402
    MUST_CALL, MUST_NOT_CALL, RULED_CALL, RULED_NO_CALL,
)
from christman_voice_sdk.audio.hearing import Hearing, harm_frame_safety  # noqa: E402

SECTIONS = [
    ("MUST CALL A HUMAN", MUST_CALL, True),
    ("MUST NOT CALL A HUMAN", MUST_NOT_CALL, False),
    ("RULED: CALL", RULED_CALL, True),
    ("RULED: NO CALL", RULED_NO_CALL, False),
]


def _head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except Exception as exc:  # the number still gets reported without it
        return f"unknown ({exc})"


def run() -> Dict[str, Any]:
    reached: List[str] = []

    def human(result, context) -> bool:
        reached.append(result.status)
        return True          # a human was reached, so the turn may complete

    ear = Hearing(crisis_callback=human)
    rows: List[Dict[str, Any]] = []
    for section, sentences, expect_call in SECTIONS:
        for sentence in sentences:
            reached.clear()
            turn = ear.hear(text=sentence)
            turn_status = turn.decision.safety.status if turn.decision is not None else "no_text"
            gate_status = harm_frame_safety(sentence).status
            called = bool(reached)
            ok = (called == expect_call) and ((turn_status == "hard_block") == expect_call)
            rows.append({
                "section": section,
                "sentence": sentence,
                "turn_status": turn_status,
                "gate_status": gate_status,
                "human_called": called,
                "expected_call": expect_call,
                "ok": ok,
            })

    counts: Dict[str, Dict[str, int]] = {}
    for section, _, _ in SECTIONS:
        mine = [r for r in rows if r["section"] == section]
        counts[section] = {
            "sentences": len(mine),
            "hard_block": sum(r["turn_status"] == "hard_block" for r in mine),
            "human_called": sum(r["human_called"] for r in mine),
            "ok": sum(r["ok"] for r in mine),
            "mismatch": sum(not r["ok"] for r in mine),
        }
    return {"commit": _head(), "rows": rows, "counts": counts,
            "total": len(rows), "mismatches": sum(not r["ok"] for r in rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, help="write the full result here")
    args = parser.parse_args()

    result = run()
    print(f"commit {result['commit']}")
    print(f"{'section':22} {'sentence':44} {'turn':14} {'gate':14} {'called':6} {'ok'}")
    for r in result["rows"]:
        print(f"{r['section']:22} {r['sentence'][:44]!r:46} {r['turn_status']:14} "
              f"{r['gate_status']:14} {str(r['human_called']):6} {'ok' if r['ok'] else 'MISMATCH'}")
    print()
    for section, c in result["counts"].items():
        print(f"{section:22} sentences={c['sentences']:2} hard_block={c['hard_block']:2} "
              f"human_called={c['human_called']:2} ok={c['ok']:2} mismatch={c['mismatch']}")
    print(f"TOTAL {result['total']} sentences, {result['mismatches']} mismatch(es)")

    if args.json:
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.json}")
    return 1 if result["mismatches"] else 0


if __name__ == "__main__":
    sys.exit(main())
