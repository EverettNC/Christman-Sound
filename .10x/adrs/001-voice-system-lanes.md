# ADR-001: One voice system across three repos, built in parallel lanes

**Status:** Accepted
**Date:** 2026-10-28
**Feature:** voice-system
**Author:** 10x-Team (Architect + Staff Engineer), on Everett Christman's rulings

## Context
AlphaVox's free-text voice came out "super robotic and nothing like" its recordings.
Four read-only reviews (sdk-review A to D, 2026-10-28) found the causes:
- Only four measured things reach the sound: F0 median, a 5-point pitch shape, rate 1.33, vowel F1-F3.
- Jitter, shimmer, HNR and breath are never measured correctly. A wrong import path in timbre_modeler.py
  falls back to fixed values 0.0 / 0.0 / 15.0. The C library loaders look in the wrong folder, so pitch reads 0.0.
- No renderer accepts breath, jitter, shimmer, intonation, cadence, prosody, articulation, resonance, timbre or affect.
- The Voice Creation Center's controls never reached any renderer. That is fixed in VCC main 5940b21:
  it now reads synthesize()'s signature and reports each control applied or not applied.
- Codec War's Booth is the source of christman_synth.c, but the clear renderer's fixes were built in C only.

Everett's rulings:
- The voice runs on the Christman Voice SDK, the Voice Creation Center and Codec War together.
- Anything new is built in CODECWAR5TH first. Christman-Sound carries the runtime port, tested sample for sample.
- Offline only. No outside speech engine, no browser speech, no downloads.
- His DSP's YIN is the pitch tracker ("We have our own DSP for that exact reason").
- His voice is never used for any being. Recordings are each being's own.
- Merge immediately; nothing left unmerged.
- Do not touch core.py or christman_sound/core.py.
- Show him the diff before changing how the SDK reads a person.

## Decision
Work runs in parallel lanes. Each lane owns its files exactly, so lanes never edit the same file.

| Lane | Owns | Delivers |
|------|------|----------|
| 1 Mouth controls | CODECWAR5TH src/lib/audio/synth.ts, phonemes.ts; Christman-Sound christman_synth.c, christman_sound/synth.py, dsp.py, voice_profile.py | synthesize() keywords breath, jitter, shimmer, intonation, cadence, prosody, articulation, resonance, timbre, affect. Built in the Booth first, then ported. Profile values map to the controls. |
| 2 Measurement | christman_voice_sdk/audio/* measuring code (not harm_frame.py or hearing.py), timbre/timbre_modeler.py, tone/tonescore_engine.py, tone/tone_analyzer.py, integration/voice_capture_client.py, scripts/measure_alphavox_voice.py, christman_sound/alphavox_voice.json | One DSP loader. YIN as the only pitch tracker. Praat-grade jitter, shimmer, HNR and breath. No silent 0.0. The measured fields jitter_local_pct, shimmer_local_pct, hnr_db, breath_ratio and bw written into each profile. |
| 3 Alignment | CODECWAR5TH new modules: src/lib/audio/align.ts, formant-track.ts (plus tests), and the Filament's timing reply | Real formant tracks and phone timings from recordings. Needed for stitching. |
| 4 Codec War repairs | CODECWAR5TH booth-room.tsx (Host voice removal), encoder/stream.ts (GB18030), engine.ts (mic canal, unused imports), studio.tsx (mic stop), dispersion-room.tsx and war-room.tsx ("sealed" text), startup.sh, encoder.ts labels | Offline guard test. Each bug fixed with a test. |
| 5 AlphaVox pass-through | AlphaVox backend routes and services on the voice path; Christman-Sound christman_voice_sdk/synthesis/* (voice_synthesis.py, speech_response.py, shorty_voice_engine_v3.py, voice_synthesis_orchestrator.py, tts_service.py) | Voice, emotion and every control pass from AlphaVox into synthesize(), or are refused with the reason. VOICE_NAME comes from the caller's profile. No path still says "not wired yet". |
| 6 Packaging and truth | pyproject.toml, setup.py, py.typed, README.md, christman_sound/__init__.py docs, models/documents/README.md, music/*, utils/*, timbre/shorty_emotion.py, timbre/voicepack.py, integration/speech_integration.py, christman_voice_sdk/audio/config.py | Install works on Everett's Intel Mac. No outside download. Documents match the code. Zip extraction checked. No "everett" voice profile. |
| 7 Stitched voice (after 2 and 3) | CODECWAR5TH sixth room, plus its Christman-Sound port | Dense analysis and resynthesis from each being's own recordings. Fidelity gate plus intelligibility gate. |
| QA (after 1, 2 and 5) | read-only | End-to-end check from AlphaVox to sound. Fidelity numbers against the recordings. |

Shared contract:
- Profile fields: jitter_local_pct, shimmer_local_pct, hnr_db, breath_ratio, bw.
- Keyword names: as in the table above.
- Controls take floats in 0..1, except affect and resonance, which are lowercase strings.
- A control the renderer cannot apply is refused, never silently ignored.

## Alternatives Considered
| Alternative | Pros | Cons | Why Not |
|-------------|------|------|---------|
| One builder does everything in sequence | No merge collisions | Hours of serial work; Everett waits on it | Too slow. The work splits cleanly by file. |
| Lanes split by repo only | Simple ownership | Christman-Sound holds four lanes' work; one lane would be huge | File-level ownership is finer and still collision-free |
| Many builders with no file ownership | Fastest start | Conflicting edits to the same files; lost work | Ownership by file is the guard |

## Consequences
### Positive
- Up to six builders at once.
- Each lane is testable on its own.
- The contract fixes the names before code exists.
### Negative
- Lanes 1, 2 and 5 depend on the shared contract. A name change has to go through the orchestrator.
### Risks
- Rebase conflicts on Christman-Sound main. Mitigation: small PRs, rebase right before merge, and run the full suite after rebasing.
- Lane 7 and QA wait on earlier lanes.

## Dependencies
- Lane 5's pass-through goes live once Lane 1's keywords land.
- Lane 7 needs Lanes 2 and 3.
- QA needs Lanes 1, 2 and 5.
