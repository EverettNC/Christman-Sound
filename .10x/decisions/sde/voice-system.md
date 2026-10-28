# SDE — voice-system

Each lane appends its entry here when it finishes: what was built, PRs with hashes, tests, and deviations.

## Before the lanes (2026-10-28)
- Christman-Sound #19: christman_synth_clear renderer. Built in C only, against the Codec War-first ruling. Lane 1 brings it into the Booth.
- Christman-Sound #20: crisis gate. A distancing word now speaks for its own clause only. Everett's table: 47 sentences, 0 mismatches.

## Lane 2 — Measurement (2026-10-28)
Built:
- christman_voice_sdk/tone/voice_measure.py, the one place the SDK measures a voice. Pitch: christman_yin through christman_sound.dsp.get_dsp() at the recording's own rate (40 ms frames, 10 ms hop, 50-500 Hz, threshold 0.15, as formants.py). Jitter (local), shimmer (local), HNR: Praat. One RMS contour, one WAV reader. A missing library or analyzer raises PitchUnavailable or PraatUnavailable.
- tonescore_engine.py and tonescore_analyzer.py measure through it. tone_analyzer.py deleted (duplicate, same wrong loader, no importer). voice_capture_client.py takes the sample rate (no default), uses YIN, and takes F1-F3 from christman_sound.formants (LPC, real-axis roots dropped). The FFT-peak "formants" and the autocorrelation tracker with its one-sample-short lag are gone.
- Offline: the emotion model in tone/ loads with local_files_only=True.
Counts in Lane 2's files, before -> after: pitch trackers 4 -> 0 (all through christman_yin), C library loaders 2 -> 0, RMS contours 3 -> 1, jitter, shimmer and HNR 3 each -> 1 each (Praat), formant extractors 1 -> 0 (reuses christman_sound.formants). Duplicate class names: ToneScoreCalculator 2 -> 1, MultiLayerToneAnalyzer 2 -> 0. ToneScoreEngine and TemporalNonverbalEngine are each defined once in christman_voice_sdk; their second copies are in christman_sound/core.py, which is never touched.
AlphaVox's 26 verified phrases (24 kHz, AlphaVox's voice): YIN median 149.1 Hz over 940 voiced frames (profile 149.3 over 932); jitter 1.97-5.94 %, shimmer 6.73-16.83 %, HNR 4.08-12.73 dB.
PRs: Christman-Sound #25, merge f8251e0. Christman-Sound #29 (voice_capture_client, this entry).
Tests: test_voice_measure.py, test_tonescore_engine.py, test_voice_capture_client.py. Known F0 at five rates, known jitter, shimmer and HNR, known formants, missing engines, AlphaVox's phrases. Full suite after rebase on main 8753ce9: 303 passed, 0 failed, 9 not run, 2 deselected (they play sound aloud).
Deviations and held for Everett:
- How ToneScoreEngine reads a person is unchanged. Arousal and dominance stay None, as they were on every build (pitch never loaded). Turning them on (READ_AROUSAL_AND_DOMINANCE) would read AlphaVox's phrases at arousal 65-81 and dominance 56-95. Valence still reads the old spectral peak ratio, now named for what it is. Both wait on Everett's ruling.
- analyze_tone reports status "unavailable" with the reason when christman_dsp or Praat is missing, instead of raising, because AlphaVox's /ingest calls it without a guard. The measuring functions raise.
- PraatToneAnalyzer (no caller in Christman-Sound or AlphaVox) now takes pitch from YIN, so its rule hypotheses would see YIN's pitch.
- For Lane 1: timbre/timbre_modeler.py still has its own loader, YIN copy, LPC and the broken tone import, and reads physiological keys jitter, shimmer and hnr, which are now jitter_local_pct, shimmer_local_pct and hnr_db. It can call voice_measure.pitch_contour, voice_measure.voice_quality and christman_sound.formants.measure_formants.

## Lane 6 — Packaging and truth (2026-10-28)
Built:
- Install works on Everett's Intel Mac. librosa removed (nothing imports it; its numba -> llvmlite 0.50.0 chain has no x86_64 macOS wheel). scipy, PyYAML, websockets and werkzeug declared. Extras: crypto (cryptography), capture (sounddevice), asr (vosk). pyproject.toml is the one dependency list; setup.py is a bare shim that says scripts/build_dsp.py builds the C library, not pip. christman_voice_sdk/py.typed added.
- Offline: shorty_emotion.py loads wav2vec2, the projection and the scaler only from caller paths (local_files_only) and raises ShortyModelMissing naming each missing piece. The projection is a dict of tensors that weights_only can load; the PCA math is written out. tests/test_offline.py fails on any new download anywhere in the package, scripts and root modules.
- voicepack.py checks every zip member before extracting (absolute, drive letter, "..", symlink -> UnsafeVoicepack). config.py makes no folders inside the package; they sit under $CHRISTMAN_AI_HOME/voice_sdk and are made by ensure_dir(). word_ear_receiver.py has no default port and refuses 5173, 5485, 5486, 5487 and 5497.
- speech_integration.py and christman_speech_to_speech.py removed (dead on every path; reasons in tests/test_integration_truth.py and PR #28).
- music: mix_project ("quality": "professional") and master_track ("lufs": -14.0) removed; NOT_BUILT names rendering, mixing, mastering and singing; sing_melody removed.
- "user" -> "client" in utils/ and Lane 6 files. grounder.py now reads BREATHING_RISE from tone.emotion_quantifier (same value, 0.10).
- Documents: README rewritten to the tree on main (file lists tested by tests/test_readme_truth.py, Lane 1's mouth text and frame-voice section, no person's profile name, build step, extras). christman_sound/__init__.py and models/documents/README.md made true.
PRs (merge commits): #23 1100f72, #24 6de5a73, #26 07de1f2, #28 8753ce9, #30 9c746ae, and the docs PR that carries this entry.
Tests: test_packaging, test_offline, test_package_safety, test_integration_truth, test_music_truth, test_readme_truth. Fresh uv venv (Python 3.11, Intel Mac), uv pip install -e ., scripts/build_dsp.py --check, full suite green before every merge; the two tests that play sound aloud were deselected every time.
Deviations and held for Everett:
- README says "License: Proprietary"; LICENSE is Apache 2.0. Neither changed. Everett rules.
- presence_guide.py still uses its own NEGATORS and SCOPE_BREAKERS (its import of structural_affect never resolves). Switching to structural_affect's sets adds ain't, barely, hardly, neither, nobody, nor, scarcely and whereas, which changes how the SDK reads a person. Held for Everett to see.
- christman_ocr_shared.py: PaddleOCR still fetches its models on first construction (the one OPEN entry in test_offline.py). paddleocr is not installed here, so a local-model path could not be proven.
- For Lane 1: dsp.py lines 31-32 say pip builds the library; it does not. For Lane 5: ShortyVoiceEngineV3 must pass wav2vec_model_dir or it raises ShortyModelMissing. For Lane 2's files: recognition_result's is_user_speech property and nonverbal_engine's docstring still say "user". For whoever starts the word-ear: StillHere's voice.py says Porch posts to 5487; a port must now be passed.
