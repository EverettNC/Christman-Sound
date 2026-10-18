# Outside voices, quarantined 2026-10-17

## The ruling

Everett's ruling, 2026-10-17: the AlphaVox rebuild cuts every outside speech engine. No Coqui TTS / XTTS, no GPT-SoVITS, no Apple `say`, no espeak, no pyttsx3, no gTTS, no pygame playback. The being speaks in its own voice through Christman-Sound, or it does not speak and says so with a typed failure.

His own synthesizer is being built on the `claude/sovereign-synth` branch. Until it is wired into `SpeechSynthesisEngine._render` (in `christman_voice_sdk/synthesis/voice_synthesis.py`), every speaking path raises `SynthesisUnavailable` (in `christman_voice_sdk/engines/synthesis_result.py`): `available` is False and `reason` is "sovereign synthesizer not wired yet".

Everett deletes; we quarantine. Every file here is byte-identical to `main` at 99a6cce (checked with `cmp` against `git show origin/main:<path>`). Nothing here is imported by the package, and pytest does not collect this folder (`pyproject.toml` sets `testpaths = ["tests"]`).

## The files

### christman_voice_sdk/engines/xtts_engine.py
- What it was: `XTTSEngine`, the Coqui XTTS v2 voice-cloning engine. Default model `tts_models/multilingual/multi-dataset/xtts_v2`.
- What it imported: `torch`, `torchaudio`, `numpy`, `time`, `warnings`, and `from TTS.api import TTS` (Coqui, loaded lazily inside `_load_model`). From the package: `BaseSynthesizer`, `SynthesisResult`, `DegradedSynthesisError` from `engines/base_synthesizer.py`, and `engines/logger.py`.
- Who called it: `synthesis/voice_synthesis.py` (the mouth), `synthesis/shorty_voice_engine_v3.py`, `synthesis/voice_synthesis_orchestrator.py`, and the XTTS fallback in `christman_sound/core.py`. All of those now raise the typed failure instead.
- `_paths.py` used this file as the marker that proved the SDK directory. The marker is now `christman_voice_sdk/__init__.py`.

### christman_voice_sdk/engines/gpt_sovits_engine.py
- What it was: `GPTSoVITSEngine`. Never wired. Its own code said "Loader not wired yet" and raised `NotImplementedError("Real GPT-SoVITS inference is not yet implemented.")`.
- What it imported: `numpy`, `soundfile`, `torch`, `time`. From the package: `BaseSynthesizer`, `SynthesisResult`, `DegradedSynthesisError` from `engines/base_synthesizer.py`, and `engines/logger.py`.
- Who called it: nothing in the package.

### christman_voice_sdk/engines/base_synthesizer.py
- What it was: the abstract base (`BaseSynthesizer`) of the two engines above, plus `SynthesisResult` and `DegradedSynthesisError`.
- What it imported: `abc`, `typing`, `numpy`, `pathlib`, `dataclasses`, `scipy.signal`, `engines/logger.py`; `soundfile` inside `SynthesisResult.save`; `torch` inside `_setup_device`.
- Before it moved, `SynthesisResult` and `DegradedSynthesisError` were copied unchanged into `christman_voice_sdk/engines/synthesis_result.py`, because other modules import them.
- `BaseSynthesizer.apply_emotion` is in this file. It resamples twice, so pitch and length move together. Everett has not ruled on it yet (true pitch shift, or rename it to what it does). It is kept here exactly as it was; nothing was changed in it.

### christman_voice_sdk/synthesis/speech_response.py
- What it was: `speak()`, a wrapper around pyttsx3, the operating system's speech engine (the `nsss` driver on the Mac).
- What it imported: `logging`, `sys`, `pyttsx3`.
- What replaced it: a `speech_response.py` with the same `speak()` name and signature that raises `SynthesisUnavailable`, so callers still import and fail loud at call time.

### tests/test_synthesis_outside_engines.py
- What it was: six tests cut from `tests/test_synthesis.py`: `test_xtts_is_reachable_through_the_package`, `test_import_failure_names_the_real_missing_dependency`, `test_fallback_plays_when_it_can_and_text_goes_on_stdin`, `test_fallback_fails_loud_when_it_cannot_play`, `test_nothing_voiced_raises_when_playback_was_requested`, `test_xtts_path_also_fails_loud_when_it_cannot_play`, with their helpers.
- What they imported: `numpy`, `pytest`, and `christman_voice_sdk.synthesis.voice_synthesis` (its `XTTSEngine`, `_XTTS_IMPORT_ERROR` and the `say`/espeak fallback, all gone). They patched `subprocess.run` and stood in fake XTTS and playback objects.
- This file is the one thing here that is not a byte copy of a whole file: it holds those six tests verbatim under a new header. They would fail if run, because the code they test is no longer in the package.

## Cut in place (not moved, because the files hold Everett's own work)

Each of these opens with a "WHAT CHANGED AND WHY (2026-10-17)" block that quotes the removed lines:

- `christman_voice_sdk/synthesis/voice_synthesis.py`: XTTS import and render, and the `say`/espeak fallback. Playback through `christman_play`, the WAV readers and the cache directory stay.
- `christman_voice_sdk/synthesis/shorty_voice_engine_v3.py`: XTTS construction, `load_voice` and render. The emotion quantification stays.
- `christman_voice_sdk/synthesis/voice_synthesis_orchestrator.py`: the XTTS engine. Voicepack, timbre, emotion embedding and lipsync wiring stay.
- `christman_voice_sdk/synthesis/tts_service.py`: the fixed `"engine": "xtts_v2_local"` status. The cache stays.
- `christman_voice_sdk/integration/voice_capture_client.py`: the `say` / espeak / pyttsx3 fallback. The Derek relay stays.
- `christman_ocr_shared.py`: the `say` / espeak / pyttsx3 read-aloud fallback. The Derek relay stays.
- `christman_voice_sdk/audio/config.py`: the GPT-SoVITS / F5-TTS / StyleTTS2 engine names, checkpoints and sampling settings.
- `christman_sound/core.py`: the pygame and XTTSEngine imports, `XTTS_MODEL`, the base_synthesizer import, the XTTS fallback and the pygame playback, each replaced by a dated comment. Everett ruled this file a symbolic header that is never wired; nothing new was wired to it.
- `_paths.py`: the marker file.
- `setup.py`, `pyproject.toml`: the `pyttsx3` dependency.
