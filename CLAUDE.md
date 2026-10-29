# CLAUDE.md — Christman-Sound

This file orients any Claude Code session working in this repository. Read it first, every time.

## What this is
Christman-Sound is the audio, voice and speech package of The Christman AI Project (Luma Cognify AI):
the C engine (measurement, the synthesizer, the frame vocoder), the voice synthesis path, ToneScore and
voice measurement, and the hearing modules. A being passes its own voice profile; the package holds no
being's identity. The people it serves include nonverbal clients, trauma survivors, dementia clients,
seniors and children. A real, vulnerable person depends on this code being both correct and honest.

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
  Replicate, paid AWS). Use what's built: his own synthesizer (Codec War's, ported in as christman_synth.c), local ASR
  (his Vosk), Ollama. **No outside speech engine** — XTTS / Coqui, GPT-SoVITS, say, espeak, pyttsx3, gTTS
  and pygame are cut (Everett's ruling, 2026-10-17; the cut engines were removed from the tree and git
  history keeps them). `gtts` and `recognize_google` are **cloud** dependencies and offline-first
  violations — remove or gate them, do not add more.
- **No stubs. No shortcuts. Nothing marked done without proof it works.**

## Working rules for this repo
- **Offline-first, client-owned, dignity-centered.** Audio of vulnerable clients must not leave the device.
- **Every analysis output must carry honest confidence + a `degraded`/`model_loaded` flag.** No silent
  fabrication on the failure path.
- **Verify every fix.** After a change: `python3 -c "import ast; ast.parse(open('FILE').read())"` at minimum;
  run the module; prove the behavior actually improved. Do not mark a task complete on faith. Rule 13
  applies to *your own* status reports too.
- **Only sound lives here.** Another being's code, another project's engine, and old review or handoff
  paperwork do not belong in this repository. Git history keeps what was removed.

Run before claiming anything:

    python3 -m pytest tests/ -q
    python3 scripts/build_dsp.py --check

Read `tests/test_harm_frame_gate.py` first when touching the crisis gate: it is Everett's MUST CALL /
MUST NOT CALL table, and the gate is written to it.

Standing rulings from Everett that bind every session:
- `christman_sound/core.py` is **not used by anything and is not to be touched** (2026-10-11).
- A subjectless utterance ("scared", "hurts") is the speaker's own. An AAC board does not need an "I".
- The seven "Everett's call, do not decide these" items recorded in the August 2026 handoff (in git
  history) stay his.
- Show him the diff before pushing changes to how the SDK reads a person.
- A test that cannot run in the current environment reports **NOT RUN** with the reason (`tests/conftest.py`).
  It never passes and never silently skips. Playback through christman_play is checked on his Mac, where the sound card is.
- Requested playback plays or fails loud (`PlaybackFailed` / `NothingVoiced`). No silent success.

Two remotes exist. `EverettNC/Christman-Sound` is where all work is done first; it goes to
`The-ChristmanAI-Project/Christman-Sound` only once tested and proven (Everett's ruling 2026-10-17).
