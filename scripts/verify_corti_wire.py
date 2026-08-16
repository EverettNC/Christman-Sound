#!/usr/bin/env python3
"""
Prove the Corti → Sound wire end-to-end over real HTTP.

Starts corti_receiver on an ephemeral port, posts the same card shapes the
TypeScript client emits (JSON.stringify of a VocalEvent), and checks:

  - measurements cross; the client `kind` does not
  - null f0 stays null through the wire; LPC 0 becomes None
  - the consumer callback receives the turn
  - malformed JSON, non-object bodies, wrong routes, disallowed browser
    origins, and oversized bodies are rejected loudly with the right codes
  - /health reports what was counted, not what sounds good
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from christman_voice_sdk.audio.corti_receiver import (  # noqa: E402
    EVENT_PATH,
    HEALTH_PATH,
    start_in_thread,
)


def fail(msg: str) -> None:
    print(f"FAIL  {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"ok    {msg}")


def post(base: str, path: str, body: bytes, origin: str | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(
        base + path,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    if origin is not None:
        req.add_header("Origin", origin)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())
    except urllib.error.URLError as err:
        if isinstance(err.reason, BrokenPipeError) or isinstance(err.reason, ConnectionResetError):
            return -1, {"error": f"connection dropped mid-upload: {err.reason}"}
        raise


def get(base: str, path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(base + path, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


CARD = {
    "id": 7,
    "kind": "grunt",
    "startedAt": 12345.0,
    "durationMs": 400,
    "peakRms": 0.04,
    "attackMs": 20,
    "decayMs": 380,
    "f0": None,
    "f1": 0,
    "f2": 0,
    "zcr": 0.08,
    "voiced": False,
    "tape": [
        {"t": 0, "rms": 0.02, "zcr": 0.08, "f0": None},
        {"t": 16, "rms": 0.04, "zcr": 0.07, "f0": None},
        {"t": 32, "rms": 0.03, "zcr": 0.08, "f0": None},
    ],
}


def main() -> None:
    turns = []
    server, receiver, thread = start_in_thread(port=0, on_turn=turns.append)
    host, port = server.server_address[0], server.server_address[1]
    base = f"http://{host}:{port}"
    print(f"receiver up on {base} (ephemeral)")

    try:
        code, body = post(base, EVENT_PATH, json.dumps(CARD).encode())
        if code != 200:
            fail(f"good card got {code}: {body}")
        event = body["ingest"]["event"]
        if event["kind"] != "unknown":
            fail(f"client kind crossed the wire: {event['kind']}")
        if event["median_f0"] is not None:
            fail(f"null f0 became {event['median_f0']} over the wire")
        if event["f1"] is not None or event["f2"] is not None:
            fail(f"LPC 0 became a formant over the wire: {event['f1']}/{event['f2']}")
        if event["duration"] != 0.4:
            fail(f"durationMs 400 must arrive as 0.4s, got {event['duration']}")
        if body["ingest"]["tape"][0]["f0"] is not None:
            fail("null f0 filled in on the serialized tape")
        if "no text — structural_affect and harm_frame not run" not in body["notes"]:
            fail(f"turn must say text organs did not run, notes={body['notes']}")
        if body["decision"] is not None:
            fail("a card with no text must not produce a fusion decision")
        ok("card crossed: kind stripped, f0 hole kept, units converted, no fake decision")

        if len(turns) != 1:
            fail(f"consumer callback saw {len(turns)} turns, expected 1")
        if turns[0].ingest is None or turns[0].ingest.event.median_f0 is not None:
            fail("callback turn does not match the posted card")
        ok("consumer callback received the HearingTurn")

        req = urllib.request.Request(
            base + EVENT_PATH,
            data=json.dumps(CARD).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "Origin": "http://localhost:8080"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            acao = resp.headers.get("Access-Control-Allow-Origin")
            if acao != "http://localhost:8080":
                fail(f"allowed origin not echoed, got {acao!r}")
        ok("allowed origin http://localhost:8080 gets CORS header")

        code, body = post(base, EVENT_PATH, json.dumps(CARD).encode(), origin="https://evil.example")
        if code != 403:
            fail(f"disallowed origin got {code}, expected 403")
        ok(f"disallowed origin rejected 403: {body['error']}")

        code, body = post(base, EVENT_PATH, b"{not json")
        if code != 400:
            fail(f"malformed JSON got {code}, expected 400")
        ok(f"malformed JSON rejected 400: {body['error']}")

        code, body = post(base, EVENT_PATH, b"[1, 2, 3]")
        if code != 400:
            fail(f"non-object body got {code}, expected 400")
        ok(f"non-object body rejected 400: {body['error']}")

        code, body = post(base, "/api/wrong/route", json.dumps(CARD).encode())
        if code != 404:
            fail(f"wrong route got {code}, expected 404")
        ok("wrong route rejected 404")

        _, before = get(base, HEALTH_PATH)
        big = json.dumps({"tape": [{"t": 0}] * 400000}).encode()
        code, body = post(base, EVENT_PATH, big)
        if code not in (413, -1):
            fail(f"oversized body ({len(big)} b) got {code}, expected 413 or dropped socket")
        _, after = get(base, HEALTH_PATH)
        if after["rejected"] != before["rejected"] + 1:
            fail(
                f"oversized body not counted as rejected: "
                f"{before['rejected']} -> {after['rejected']}"
            )
        if after["received"] != before["received"]:
            fail("oversized body was INGESTED — the cap is not real")
        ok(
            f"oversized body ({len(big)} bytes) refused before read "
            f"(client saw {'413' if code == 413 else 'dropped socket'}; "
            f"health rejected {before['rejected']} -> {after['rejected']})"
        )

        code, health = get(base, HEALTH_PATH)
        if code != 200:
            fail(f"health got {code}")
        if health["received"] != 2:
            fail(f"health received={health['received']}, expected 2 (two good cards)")
        if health["rejected"] != 5:
            fail(f"health rejected={health['rejected']}, expected 5")
        if health["ring_size"] != 2:
            fail(f"health ring_size={health['ring_size']}, expected 2")
        if health["callback_failures"] != 0:
            fail(f"health callback_failures={health['callback_failures']}, expected 0")
        ok("health honest: received=2 rejected=5 ring=2 callback_failures=0")

        def boom(_turn):
            raise RuntimeError("consumer exploded on purpose")

        receiver.on_turn = boom
        code, _ = post(base, EVENT_PATH, json.dumps(CARD).encode())
        if code != 200:
            fail(f"card after consumer failure got {code} — wire must stay up")
        _, health = get(base, HEALTH_PATH)
        if health["callback_failures"] != 1:
            fail(f"callback failure not counted: {health['callback_failures']}")
        ok("raising consumer: logged, counted in /health, wire stayed up")

    finally:
        server.shutdown()
        thread.join(timeout=5)

    print("ALL PROVED — the wire carries measurements and refuses everything else")


if __name__ == "__main__":
    main()
