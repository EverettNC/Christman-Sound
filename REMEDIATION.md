# REMEDIATION.md — Christman-Sound hardening plan

Ordered, evidence-backed, Rule-13-compliant. Do phases **in order**. Verify every step (no "done" without
proof). Every line item cites the file:line from the medical-grade review. Nothing here requires a paid service.

Legend: `[ ]` todo · each item = exact defect → required fix → how to prove it.

---

## PHASE 0 — Make it import (nothing works until this is green)
Verify each with: `python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" <file>`

- [ ] `core.py:67` — resolve merge conflict (keep HEAD side; logger must be defined before use ~line 60/88). Prove: parses + `python3 -c "import core"` in the SDK context.
- [ ] `setup.py:19` — resolve merge conflict (keep HEAD dependency pins: numpy<2, transformers<5, praat-parselmouth, torchaudio). Prove: `pip install -e .` succeeds.
- [x] `crypto_bridge.py:15` — parses. Loads official HNDL (`~/Harvest-Now-Decrypt-Later`, https://github.com/The-ChristmanAI-Project/Harvest-Now-Decrypt-Later.git). `HybridPQCipher(768)` encrypt/decrypt roundtrip proved; `HybridSigner` reports `classical_rsa_pss` when oqs is absent. Bad signature raises. Does not import a nonexistent module.
- [ ] `CHRISTMAN_EAR_CANAL/__init__.py:34` — resolve merge conflict (`unmatched ']'`). Prove: `import CHRISTMAN_EAR_CANAL` works.
- [ ] `christman_preflight.py:205` — resolve merge conflict (keep HEAD: parselmouth/torchaudio mappings). Then **delete the 3 duplicate broken copies** `christman_preflight0.py`, `christman_preflight2.py`, `christman_preflight3.py`. Prove: preflight runs and reports.
- [ ] `christman_voice_sdk/_init_.py` → rename to `__init__.py`. Prove: `import christman_voice_sdk; christman_voice_sdk.__version__` resolves.
- [ ] Run the (now-fixed) preflight across the tree; record baseline loaded-module count.

## PHASE 1 — Stop the lying (Rule 13 — highest patient-safety priority)
Replace every fabricated output with EITHER a real engine OR an explicit honest error. No `random.choice`, no hardcoded metrics, no placeholder bytes on any path a user reaches.

- [ ] `audio/enhanced_speech_recognition.py:195,256` — delete the `random.choice(sample_phrases)` transcript path; return `{"error":"ASR not implemented","status":"error"}` until a real engine is wired.
- [ ] `audio/sound_recognition_service.py:49,98` — stop labeling the simulated engine "real"; never emit random `"distress"`/`"help"` events. Gate all simulation behind an explicit opt-in flag.
- [ ] `engines/gpt_sovits_engine.py:118,267` — the fallback must NOT return looped-reference/sine audio as a successful `SynthesisResult`. Return `degraded=True` (or raise a typed `DegradedSynthesisError`) the caller surfaces in UI. Delete the `np.random.randn` "gentle noise" branch.
- [ ] `engines/base_synthesizer.py:99` — remove hardcoded `{speaker_similarity:0.95, naturalness_mos:4.5, clarity:0.90}`; return `None`/real metrics only.
- [ ] `nonverbal/engine_temporal.py:369-370` — delete the random-label/random-confidence branch; no-model must return the SAME safe default as model-missing (confidence 0.0, "I don't understand").
- [ ] `nonverbal/nonverbal_engine.py:353-357` — delete the `random.uniform(-0.05,0.05)` confidence noise (3 sites).
- [x] `nonverbal/cochlear_sync_tts.py` — **deleted from the tree.** Placeholder TTS is gone. Do not reintroduce.
- [ ] `tone/tonescore_engine.py:102` — delete fabricated accuracy block + false "CREMA-D/RAVDESS" claim (or check in the real eval harness that produced numbers).
- [ ] `music/christman_studio.py:313,353` — remove fabricated `"professional"`/LUFS metrics from functions that write no audio; label the module symbolic or implement real rendering.
- [ ] `timbre/timbre_modeler.py:247` & `timbre/shorty_emotion.py:158` — stop returning `np.random.randn` / raw-embedding-dims as speaker identity / emotion scores; raise `NotImplementedError` or return honest `unavailable`.

## PHASE 2 — Fix the emotion label mapping (wrong labels = wrong care decisions)
- [ ] `tone/tonescore_engine.py:144`, `tone/tone_analyzer.py:136`, `tone/christman_tone_engine_v2.py:31` — build the label list from `model.config.id2label` at load time; never hardcode 7/11 labels against a 4-class head. Prove: print id2label, assert output keys == model classes.
- [ ] All three — on model load/infer failure return `{"status":"unavailable","emotions":None}` (not flat `0.14` scores); callers must handle it. Remove the "anger"-by-tiebreak default.
- [ ] `christman_tone_engine_v2.py:52` & `audio_processor.py:87` — fix stereo downmix axis (`axis=1`, not `axis=0`).

## PHASE 3 — Wire ONE real engine each (offline, free — Rule 15 clean)
- [ ] Real local ASR: integrate VOSK or faster-whisper behind `speech_integration.py`; fix the wrong import name at `speech_integration.py:67`; make engine-load failure LOUD. Prove: speak a known phrase, get correct text.
- [ ] Real local TTS: make XTTS (`xtts_engine.py`) the working default; fix the <3s reference bug (`xtts_engine.py:113`); build the real fallback chain XTTS → GPT-SoVITS → offline `say`/pyttsx3 with an explicit degraded signal at each hop. Prove: synthesize a sentence, hear the sentence.
- [ ] Remove/gate cloud paths: `gtts` (`tts_service.py`, `voice_synthesis.py`, `voice_diagnostics.py`, `setup.py`) and `recognize_google` (`speech_recognition_engine.py`). Prove: works with network disabled.
- [ ] Compile + wire `christman_dsp.c` (`cc -shared -O2 -o christman_dsp.so christman_dsp.c -lm`); fix `DSP_LIB_PATH` (points inside SDK, .so is at repo root); surface `_dsp_ok=False` in results. First fix the C bugs below.

## PHASE 4 — Security (Rule 12)
- [ ] `christman_ocr_shared.py:242,244` — replace `os.system(f'say "{text}"')` with `subprocess.run(["say", text], shell=False)`. Same for espeak branch. (Also `voice_capture_client.py:189` — add `import os`; use subprocess list form.)
- [ ] `timbre/voicepack.py:247` — implement real encryption (e.g. `cryptography.Fernet` + user key) or raise `NotImplementedError`; never log "encrypted" for a copy. Verify checksums in `validate()`.
- [ ] Replace `pickle.load` of models/voicepacks (`engine_temporal.py` ×6, `timbre_modeler.py:436`) with safetensors/JSON, or verify a signed hash before load.
- [ ] `christman_dsp.c:114` — guard LPC size_t underflow (`if ((size_t)order >= length) return zero-filled`). `christman_dsp.c:110` — initialize `out_a` before any early return. `:136` — guard `error <= 0`.
- [ ] `voice_capture_client.py:168` — require `wss://` + endpoint allow-list for voice biometrics; never plaintext `ws://`.
- [ ] `tts_service.py:361` — remove `debug=True` from any Flask `app.run`.

## PHASE 5 — Reliability & dignity
- [ ] `core.py:1131` (VAD) — only start the silence countdown after speech energy is first detected; never cut off a slow-to-start speaker.
- [ ] `nonverbal_engine.py:82` — actually ENFORCE `confidence_threshold` at every acting return; below it → "unclear, please confirm." Add a user override / undo path (currently none exists).
- [ ] `nonverbal_engine.py:215` — stop the confidence-only-decays bug; never update confidence from unknown-outcome interactions.
- [ ] `voice_synthesis.py:246` — never mutate spoken text; drive prosody by parameters, not by prepending "[With strong fear]".
- [ ] `base_synthesizer.py:93` — fix normalization (`audio = audio/peak*0.99`); replace resample-based "pitch shift" (`:74`) with real pitch shift or drop the feature.
- [ ] Add locks around shared mutable state (nonverbal engine maps, orchestrator voice state, capture callback lists).
- [ ] Temp/cache hygiene: uuid names + restrictive perms + retention policy for `/tmp/christman_sdk`, `voice_cache/`, `static/audio/`. No vulnerable-user audio left world-readable.
- [ ] Persistent per-interpretation audit log (input → interpretation → confidence → spoken output) so "why did it say that?" is answerable after a restart.

## PHASE 6 — Consistency / cleanup (Rule 4, Rule 10)
- [ ] Unify the four loggers on the `engines/logger.py` dual-import pattern.
- [ ] Fix CHRISTMAN_EAR_CANAL adapters: `EAR.py:32` (`listen` doesn't exist), `TONE.py` (wrong API + undefined `logger`), `OCR.py:29` (missing `import asyncio`).
- [ ] Remove hardcoded machine paths (`VOICES.py:45` `/Users/EverettN/vega`; identity mismatches — `christman_studio.py:51` "BROCKSTON", `speech_personality.py:18` Brockston→"Alpha Vox").
- [ ] Delete superseded `tone_classifier.py` (use honest `written_tone.py`); adopt its NON-CLAIMS docstring as the template for every analysis module.

---

## Definition of done (per item)
1. Change made. 2. File parses. 3. Module imports. 4. Behavior demonstrably correct (real input → real output,
or honest `unavailable` on the failure path). 5. No new Rule 13 violation introduced. 6. Preflight loaded-count
did not regress. Only then mark `[x]`.
