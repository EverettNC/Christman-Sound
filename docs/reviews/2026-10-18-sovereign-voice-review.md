# Sovereign voice: QA and security review

Branch: claude/sovereign-voice, cut from origin/main f9eb077.
Date: 2026-10-18. Reviewer: Claude, as QA and as security reviewer.
Scope: the combined branch: the three Claude branches merged, the mouth wired to christman_synth, and everything that path touches.

## 1. What is on the branch

Commits ahead of origin/main, oldest first:

- 19b89cd. From claude/cut-outside-voices. It puts christman_sound/core.py back to 99a6cce and brings CLAUDE.md in line with the 2026-10-17 rulings.
- 6f5d3d9. Merge of claude/cut-outside-voices.
- c6233d6. SynthesisResult saves christman_synth's array('f') with the standard library.
- cbd66b0. SpeechSynthesisEngine._render wired: text, then lexicon, then christman_synth, then samples.
- 557c5e2. TTS service: the cache key carries the voice, and lexicon failures get a plain answer.
- 0a07b6e. tests/test_sovereign_voice.py.
- 81bcc6e. A lexicon line that is not UTF-8 becomes a LexiconFormatError naming its line. This was found during this review.

How the merges went:

- claude/sovereign-synth and claude/formant-frames were already fully inside origin/main, through PRs #10, #11 and #13. Merging them was "Already up to date". No commit was made and nothing conflicted.
- claude/cut-outside-voices had one commit main lacked: 19b89cd, pushed about a minute after PR #12 merged. It merged with no conflicts.
- There was no dsp.py conflict. Main already holds the sovereign-synth dsp.py, and formants.py runs on it: tests/test_formants.py passes.

core.py: the branch's christman_sound/core.py is byte-identical to 99a6cce, Everett's text (sha256 9d011ce2...). It is NOT byte-identical to current main f9eb077 (sha256 6611b55a...). Main carries f546841, which cut lines out of core.py against the "never touched" ruling. No line of core.py was written by hand on this branch. Which version stands is listed under Everett's decisions below.

## 2. Test results (real runs, this machine, Linux, Python 3.13.16)

The command was `python3 -m pytest -q`, run after `python3 scripts/build_dsp.py`. CODECWAR_SRC pointed at a fresh CODECWAR5TH clone so the sample-for-sample test against Codec War's own synth.ts could run under node 22.

| | passed | failed | NOT RUN | collected |
|---|---|---|---|---|
| origin/main f9eb077 | 146 | 3 | 5 | 154 |
| claude/sovereign-voice | 160 | 3 | 6 | 169 |

Regressions against main: none. Every test outcome on main is the same on the branch except one test. test_synthesis.py::test_mouth_refuses_with_a_typed_failure_and_writes_nothing held the old "not wired" refusal. It was replaced by two tests: the no-lexicon case and the no-library case. Both pass.

The same 3 tests fail on main and on the branch. Each fails because a module is not installed here: not a pass, and not a NOT RUN.

- tests/test_audio_lifecycle.py::test_silent_recording_has_no_pitch. websockets is missing; voice_capture_client.py:89 imports it.
- tests/test_audio_lifecycle.py::test_tone_is_measured_not_defaulted. Same cause.
- tests/test_synthesis.py::test_phoneme_confidence_is_none_not_a_constant. soundfile is missing.

Installing them was not permitted in this session, so these 3 make no claim either way. See finding 12.

NOT RUN on the branch, with reasons:

- test_sovereign_voice.py::test_generate_speech_audio_plays_through_christman_play. There is no audio device here: christman_play (alsa) refused with code 4, "the audio device could not be opened". Playback is checked on Everett's Mac.
- test_sovereign_synth.py::test_speak_plays_through_christman_play. Same reason.
- test_no_outside_voices.py::test_local_speak_fallback_raises_the_typed_failure. websockets is not installed.
- test_no_outside_voices.py::test_ocr_read_aloud_raises_when_derek_is_not_there. websockets is not installed.
- test_synthesis.py::test_shorty_quantifies_then_refuses_to_render. torch is not installed.
- test_synthesis.py::test_orchestrator_embeds_emotion_then_refuses_to_render. torch is not installed.

Also not run anywhere in this review: the mouth against Everett's real align_lexicon.txt (vosk-model-en-us-0.22, about 484,651 lines). That file lives on his Mac and is not in this repo. Load time, memory, and real-word coverage are therefore unmeasured. Every lexicon test used the inline fixture.

Measured numbers from the branch:

- _render on "Hello, Christman. Hello, Christman.": 36149 samples (2.259 s) at 16000 Hz. Peak 0.7000 (Codec War's normalisation), rms 0.0527, synthesis 3.3 ms. The samples are identical, one by one, to christman_sound.synth.synthesize on the same text at f0 149.3.
- Output pitch by christman_yin: 2048-sample frames, 512 hop, voiced frames only (50 to 600 Hz). 29 of 67 frames were voiced. The median was 149.53 Hz against 149.3 requested: +0.16 %, inside the synth tests' stated tolerance of 1.24 %, which is f0/(sr - f0) + 0.3 %. The worst voiced frame was 1.10 % off; the limit is 3 %.
- Codec War's own synth.ts against christman_synth:
  - All 40 phones: 71732 of 71732 samples identical.
  - Sustained AA: 10293 of 10293 identical.
  - Text "hello filament booth" at rate 1.3: 19554 of 19554 identical.
  - Maximum difference in all three: 0.
- The whole path ran with numpy, scipy, soundfile, sounddevice, pygame and librosa blocked: text, then lexicon, then christman_synth, then the cache WAV, then the WAV reader that feeds christman_play. None of them loaded.
- The stdlib WAV reader returns exactly the float32 values the old numpy reader gave. Checked for 16-bit, 32-bit, 8-bit and IEEE-float WAVs.

The 149.3 Hz value came from this session's measurement of his 26 phrases. It was not measured again in this review.

## 3. Findings, most serious first

### 1. core.py on this branch is not main's core.py, and it still names outside engines and a network download

**Decision for Everett.**

- What it contains:
  - christman_sound/core.py:118 imports pygame.
  - core.py:169 imports XTTSEngine from `.engines.xtts_engine`. christman_sound/engines/ does not exist, so this import always fails.
  - core.py:371-372 call `from_pretrained` for wav2vec2, which downloads from Hugging Face.
- How reachable it is:
  - Nothing in Christman-Sound imports core.py.
  - AlphaVox's live surface (backend/app) does not import it either; sound_package.py reports `core_py_used: False`.
  - tests/test_no_outside_voices.py skips core.py by name, with the ruling written in.
  - So none of these lines can run unless something imports core.py.
- What still points readers at it: christman_sound/__init__.py:11-13 still says AlphaVox's routers import `christman_sound.core`. That is no longer true, and it invites someone to wire core.py back in.

### 2. Free text now has a voice, but it is Codec War's formant voice at Everett's pitch, not Everett's voice

**Decision for Everett.**

- christman_sound/synth.py:35-36 says this in its own words.
- AlphaVox's standing rule: AlphaVox speaks in Everett's voice or it does not speak. Its CLAUDE.md also names the engines it accepts, "xtts" or "recorded".
- Once CHRISTMAN_LEXICON_PATH is set on the Mac, every AlphaVox path that reaches voice_synthesis.synthesize_speech will produce audio reported as engine "christman". Those paths include AlphaVox's tts router, stream_gateway, and sound_package.speak.
- Whether "christman" passes AlphaVox's impostor-voice gate is his ruling. Until he rules, leave CHRISTMAN_LEXICON_PATH unset on any machine a client uses. With it unset the mouth answers with LexiconUnavailable and makes no sound.

### 3. Playback reports success after 150 ms

- voice_synthesis.py:581-586: play_audio_file joins the playback thread for 0.15 s and then returns True.
- A device failure after that window is written to `_play_error` and logged. The caller has already been told it played, so `_play_or_raise` returns as a success.
- This is a window where success can be reported falsely. It predates this branch and was not changed here, because it touches how playback reports. It needs Everett's word before it changes.

### 4. A client's words stay on disk with no limit

- voice_synthesis.py:427-429: every call to generate_speech_audio without an output_path creates a new `NamedTemporaryFile(delete=False)` in ~/.christman_ai/voice_cache. Nothing deletes them.
- tts_service keeps its own copies in ./static/audio.
- Each file is a client's words in audio. The files are 0600. The cache directory is created 0700, but its parents are created with the default mode.
- Both the retention policy and the disk growth need a ruling.

### 5. The TTS service has no limit on text length

- tts_service.py, /generate: the text length is not capped.
- christman_sound.synth walks every token in Python. dsp.py:413 then allocates 4 bytes for each output sample.
- One large POST body can tie up memory and CPU on the device.
- Mitigation in place: the service binds to loopback by default (tts_service.py:89).

### 6. The lexicon cache never evicts

- christman_sound/lexicon.py:359: each distinct (path, size, mtime) keeps a full lexicon in memory for the life of the process. For the real lexicon that is about 484k entries per version.
- This only grows when the file changes while the process runs.

### 7. How the native library is loaded (ctypes)

- christman_sound/dsp.py:144-155 and :215: the first loadable christman_dsp*.so is loaded from, in order:
  1. $CHRISTMAN_DSP_LIB
  2. the package directory
  3. the repository root
  4. christman_voice_sdk/
- A library's constructor runs at load. Anyone who can write to one of those directories can therefore run native code in every being that imports the mouth.
- Keep install directories writable only by the owner. Control of the environment variable already means control of the process.
- On Linux, christman_dsp.c:336 calls `dlopen("libasound.so.2")` by name. That is the normal lookup, and it follows the library search path.
- The ctypes buffers were checked:
  - synth() checks that the phone, duration and contour lengths agree before the call.
  - The C code (christman_synth.c:230-250) checks every phone id, duration, f0, rate and sample rate. It refuses with code 7 unless out_len equals the computed length.
  - play() passes the true buffer length.
  - No out-of-bounds write was found.
- One remaining edge: christman_synth_length (christman_synth.c:208-212) casts `floor(t * sample_rate)` to size_t with no ceiling. A caller of the public `synthesize(phones=...)` API can pass a finite but enormous duration; that cast is then undefined behaviour, or Python raises MemoryError allocating the buffer. The text path cannot reach this, because durations come from the C table.

### 8. The lexicon path

- christman_sound/lexicon.py:369: the path comes only from the caller's argument or $CHRISTMAN_LEXICON_PATH, never from an HTTP request.
- The file is opened read-only and parsed with str.split. It is never executed or evaluated.
- An unknown phone raises PhoneNotMapped. Nothing is dropped and nothing is substituted.
- Fixed on this branch (81bcc6e): a byte that was not UTF-8 escaped as a bare UnicodeDecodeError. That error is not a LexiconError, so `synthesis_readiness()` raised instead of returning the reason, and the mouth's constructor crashed. It is now a LexiconFormatError naming the file and line, with a test.

### 9. TTS service details

- tts_service.py:201: the 500 branch returns `str(e)` to the requester, which can expose filesystem paths.
- :160: `send_file("static/index.html")` resolves against whatever directory the process started in.
- :97: md5 is used for the cache key. It is not a security use, and on a loopback service a collision is not a practical attack.
- :167: `reference_audio_path` from the request is recorded and placed in the cache key, but never opened, because christman_synth has no reference-voice input.
- Fixed on this branch (557c5e2):
  - The cache key now includes the engine, f0 and the lexicon file's identity. Before, a WAV left by the old XTTS or `say` renderer under the same text would have been served as this voice with status "ok". A test proves the old-key file is not served.
  - Lexicon failures now get 503 or 422 with `available: false` instead of a bare 500.

### 10. Other speaking paths still give "sovereign synthesizer not wired yet"

**Decision for Everett.**

These were kept cut, as instructed:

- speech_response.py:40
- shorty_voice_engine_v3.py:128
- voice_synthesis_orchestrator.py:16 and :150
- voice_capture_client.py:321
- christman_ocr_shared.py:295

Now that the mouth renders, the reason they give is no longer true. Whether each one is routed to the mouth, or given a reason that is true, is Everett's call.

### 11. Outside network and outside binaries still in the tree (none of them speech engines)

- christman_voice_sdk/timbre/shorty_emotion.py:59-60: wav2vec2 `from_pretrained`, which downloads from Hugging Face the first time ShortyVoiceEngineV3 is constructed.
- christman_ocr_shared.py:270: websocket to Derek. ws:// is loopback only, and anything else must be wss:// (:70-72).
- christman_ocr_shared.py:183: `scrot` for screen capture.
- phoneme_labeler.py:95 and :124: runs `mfa` from PATH with list arguments and no shell.
- scripts/verify_corti_wire.py: urllib, as a manual check script.
- scripts/build_dsp.py:54 and :62: runs $CC or cc with list arguments.

What is not in the tree, outside quarantine/:

- No `shell=True` anywhere.
- No `os.system`. Every occurrence is a quoted removal inside a docstring: voice_capture_client.py:37, and voice_synthesis.py:82 for the old say/espeak subprocess.
- `recognize_google` appears only in a docstring that records its removal (speech_recognition_engine.py:33).
- quarantine/ is not packaged (pyproject includes only christman_voice_sdk* and christman_sound*) and is not collected by pytest (testpaths = tests).
- The new path (voice_synthesis, then christman_sound.synth, lexicon and dsp) opens no socket and starts no subprocess.

### 12. Tests that fail instead of reporting NOT RUN

The three failures in section 2 crash on a missing module. They should report NOT RUN with the reason, as tests/conftest.py requires. They make no false claim, but they turn a red run into noise.

### 13. Packaging

- christman_voice_sdk/__init__.py imports audio/config.py, which imports `yaml` at line 38. PyYAML is not declared in pyproject.toml, and the mouth cannot be imported without it.
- pyproject.toml still declares numpy<2, librosa, soundfile, torch, torchaudio and transformers. The sovereign path needs none of them. Trimming the list is Everett's call, since other modules still use them.

### 14. Words

- 16 Python files outside quarantine carry patent docket numbers in their headers. Examples:
  - christman_voice_sdk/__init__.py:8
  - christman_voice_sdk/synthesis/__init__.py:5
  - christman_voice_sdk/engines/__init__.py:5
  - christman_sound/core.py:34
- That is against the standing rule that headers say "patent pending" only where a document calls for it, and never carry a number. core.py cannot be touched. The other headers fall under No Erasure, so changing them is Everett's call.
- "user" appears in older history notes: voice_synthesis.py:107-136 and tts_service.py:45. Nothing added on this branch uses it.

### 15. Emotion is not applied

- christman_synth has no emotion input.
- emotion_params passed to the mouth are logged at WARNING and kept in `last_emotion_unapplied` (voice_synthesis.py:450). The audio is identical with or without them, and a test proves it.
- Mapping emotion onto pitch or rate would be an invented mapping, so none was made. Everett's call.

### 16. A sentence composed for a being

- christman_voice_sdk/integration/christman_speech_to_speech.py:36 builds "I heard you say '...'. How can I help?" as the being's reply.
- It returns None before anything is spoken, because there is no lip-sync engine. It is still words composed on a being's behalf, waiting for a mouth.

## 4. Anything that fakes success?

On the new path, no:

- Every refusal raises a typed error.
- No silence is returned as audio.
- No word is spelled letter by letter.
- No fixed number is reported as a measurement.
- `synthesis_readiness()` is True only when the library, the synthesizer symbols and the lexicon all load.

Outside the new path, the one place found that can report success falsely is the 150 ms playback window in finding 3.

## 5. Waiting on Everett

1. Which core.py stands: 99a6cce, his text, on this branch; or f546841's edit, on main.
2. Whether engine "christman" (Codec War's formant voice at 149.3 Hz) may speak to AlphaVox clients, or stays off until it is shaped toward his voice.
3. Feeding the measured formants (christman_sound/formants.py) into christman_synth's table. This is the next step, and it has not been started.
4. Whether to route speech_response, Shorty, the orchestrator, the capture client and the OCR read-aloud to the mouth.
5. Emotion: whether any emotion maps onto pitch, rate or contour.
6. Retention of rendered WAVs in the voice cache and in static/audio.
7. The 150 ms playback success window (finding 3).
8. Docket numbers in the headers (finding 14).
