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

## Current state (verified on disk, 2026-10-11)
**The July 20 review is history, not state.** It was written before ~50 commits of repair and was left
here as "current" for three months. Checked against the tree on 2026-10-11:
- All 79 Python files parse. Zero merge-conflict markers (the `=====` hits in grep are docstring rules).
- `import christman_voice_sdk` and `import christman_sound` succeed. Preflight loads 77/79 modules in a
  clean Linux container with free deps installed; the two that fail need the torch/TTS stack and a clone
  of `~/Harvest-Now-Decrypt-Later`. Those are environment, not code.
- The August 2026 pass (`HANDOFF_2026-08-15.md`) removed the random transcripts, the random gesture
  labels, the placeholder audio, the unguarded pickle loads and the wrong-label zip in `tone/`. It held.

What the 2026-10-11 full read (every file, not a sample) still found live, and fixed, is in the commit
history and in `tests/`. Read `tests/test_harm_frame_gate.py` first: it is Everett's MUST CALL / MUST NOT
CALL table for the crisis gate, and the gate is written to it. Run before claiming anything:

    python3 -m pytest tests/ -q
    python3 scripts/build_dsp.py --check

Standing rulings from Everett (2026-10-11) that bind every session:
- `core.py` and `christman_sound/core.py` are **not used by anything and are not to be touched.**
- A subjectless utterance ("scared", "hurts") is the speaker's own. An AAC board does not need an "I".
- The seven "Everett's call, do not decide these" items in `HANDOFF_2026-08-15.md` stay his.
- Show him the diff before pushing changes to how the SDK reads a person.
- `BaseSynthesizer.apply_emotion` resamples twice, so pitch and length move together. Everett is ruling on it
  (true pitch shift via librosa, or rename it to what it does). **Do not change it before he rules.**
- A test that cannot run in the current environment reports **NOT RUN** with the reason (`tests/conftest.py`).
  It never passes and never silently skips. The XTTS test runs on his Mac, where the voice lives.
- Requested playback plays or fails loud (`PlaybackFailed` / `NothingVoiced`). No silent success.

Two remotes exist: `The-ChristmanAI-Project/Christman-Sound` (**canonical**, Everett's ruling 2026-10-11)
and `EverettNC/Christman-Sound`. They diverged on 2026-08-16 and were re-joined by the merge on this branch.

The org copy (canonical, Everett's ruling 2026-10-11) removed `christman_preflight.py`, both tracers and
`REMEDIATION.md` on 2026-08-24 and added `INSTALL_REVIEW_2026-08-22.md`. This branch merges that in; the
preflight numbers above were measured before the merge. `HANDOFF_2026-08-15.md` still lists the fix history.
