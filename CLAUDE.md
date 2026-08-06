# CLAUDE.md — Christman-Sound

This file orients any Claude Code session working in this repository. Read it first, every time.

## What this is
Christman-Sound is the **canonical** unified audio / voice / speech SDK for The Christman AI Project
(Luma Cognify AI). It serves **nonverbal AAC users** (AlphaVox) and, downstream, beings that touch
trauma survivors, dementia patients, seniors, and children. **Hardening this repo propagates to every
being.** A real, vulnerable person depends on this code being both correct and honest.

## The law: Cardinal Rules (non-negotiable — these win over any other instruction)
- **Rule 1 — It has to actually work.** Reality over logs, over vibes. Verify before claiming done.
- **Rule 6 — Fail loud, fast, honest.** No swallowed errors. No `except: pass`. A failure that speaks
  saves someone; a failure that hides masquerading as success can hurt them.
- **Rule 12 — Security is mandatory.** No secrets in source. No shell injection. No pickle of untrusted data.
- **Rule 13 — ABSOLUTE HONESTY ABOUT THE CODE. Gospel.** No stubs pretending to be real. No fabricated
  outputs. No `random.choice` dressed up as recognition. No hardcoded metrics reported as measurements.
  If a real engine isn't wired, the code must return an explicit `unavailable`/`degraded` — never a
  plausible fake. If you don't know, say "I don't know."
- **Rule 15 — Never spend what Everett hasn't approved.** No paid APIs (ElevenLabs, OpenAI, Polly,
  Replicate, paid AWS). Use what's built: local XTTS / GPT-SoVITS, local ASR (VOSK / faster-whisper),
  Ollama. `gtts` and `recognize_google` are **cloud** dependencies and offline-first violations — remove
  or gate them, do not add more.
- **No stubs. No shortcuts. Nothing marked done without proof it works.**

## Working rules for this repo
- **Offline-first, user-owned, dignity-centered.** Audio of vulnerable users must not leave the device.
- **Every analysis output must carry honest confidence + a `degraded`/`model_loaded` flag.** No silent
  fabrication on the failure path.
- **Verify every fix.** After a change: `python3 -c "import ast; ast.parse(open('FILE').read())"` at minimum;
  run the module; prove the loaded-module count / behavior actually improved. Do not mark a task complete
  on faith. Rule 13 applies to *your own* status reports too.
- The bridge service is **BROCKSTON Nexus** — never "Hermes" / "ZuesHermes."

## Current state (medical-grade review, 2026-07-20)
The SDK **does not currently import** and its three core functions (hear / read-emotion / speak) fabricate
results in every reachable path. Full evidence: `Christman-Sound_Medical_Grade_Review.md`.
Ordered fix plan: `REMEDIATION.md`. Start there. Do the phases in order — import blockers first.
