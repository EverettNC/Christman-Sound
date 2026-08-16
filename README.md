# Christman Sound — Unified Audio, Voice, and Speech System

A comprehensive framework for the Christman family of autonomous beings. This system integrates speech recognition, voice synthesis, tone analysis, music generation, and audio processing into a cohesive voice and sound ecosystem.

## Overview

**Corti is the ear.** It measures. It does not classify. Christman-Sound
consumes the card and the tape. `EAR.listen()` is a timed microphone grab —
it is not hearing.

```
CORTI                         ← true hearing (separate organ)
     VocalEvent { card, tape }
     ↓
hearing.py  +  corti_ingest.py
     ├─ tape_contour / prosody     (arousal only, relative)
     ├─ structural_affect          (text: clauses, not bag-of-words)
     └─ harm_frame                 (who did what to whom)
          ↓
     fusion + face                 (decide, or MODE_UNKNOWN)
```

Corti's `kind` (grunt, tick, groan, hiss) stays on the Corti client.
Stimming stays on AlphaVox. This Sound goes into every being — client
labels do not come with it. Measurements cross. Holes stay holes.
Do not wire Corti as a `SoundDetectorBackend`.

## Package Structure

### CHRISTMAN_EAR_CANAL
Simplified interfaces for common voice operations. This is an **adapter layer**
— it wraps existing real modules, it does not replace them:
- `EAR.py` — Microphone capture. **VAD is not implemented**; `listen()` aliases
  fixed-duration capture and does not wait for a slow-to-start speaker
- `SPEAK.py` — Speech synthesis (XTTS with macOS fallback)
- `TONE.py` — Emotional tone and personality analysis (file in, dict out)
- `PHONEMES.py` — Phoneme and viseme extraction
- `VOICE_PROFILE.py` — Voice frequency profiles
- `OCR.py` — Screen reading and document scanning

### Corti — the browser ear

**Status: not in this repository yet.** Corti is a separate organ. The Python
canal above does **not** import it, and that separation is deliberate — see the
boundary rule below before "fixing" it.

Corti measures a live utterance in the browser and emits two things:

**Per-utterance card**, ring of 8:
`kind` · `duration` · `attack` · `decay` · `peak RMS` · `median F0` (or null) ·
`last F1/F2` · `mean ZCR` · `voiced`

**`tape: TapeFrame[]`**, cap 360:
`{ t: ms from onset, rms, zcr, f0: number | null }`

Rules that hold whatever consumes it:

- **The cap drops the TAIL.** Past roughly 6s the ending is gone. Every
  end-of-utterance feature must be `None` when `truncated`.
- **`t` is `performance.now()` — about 16ms per animation frame, and NOT
  uniform.** Slopes must use real `t`. A gap in `t` is a dropped frame, not a
  pause.
- **A null `f0` is a hole, not a zero.** Never interpolate across it.
- **Jitter, shimmer, and HNR are per-frame only.** They die with the frame.
- **Cold start: no speaker, no baseline.**
- **`kind` does not enter this repo.** It is Corti's client word. Dropped at ingest.

**The boundary — do not cross it.** The ear measures. It does not classify.
Sound does not import grunt, tick, or stimming. Two organs, one job.

### christman_voice_sdk

#### Audio (`audio/`)
- `audio_processor.py` — WAV processing, normalization, format conversion
- `audio_encoder.py` — Audio encoding and compression
- `speech_recognition_engine.py` — Base speech-to-text
- `enhanced_speech_recognition.py` — Multi-model speech recognition
- `real_speech_recognition.py` — Production speech recognition
- `sound_recognition_service.py` — Sound classification and detection
- `fusion_engine.py` — Carbon + Silicon under Aegis. Decide, or unknown.
- `hearing.py` — Corti sits on top. Composes ingest + affect + harm_frame.
- `corti_ingest.py` — Corti JSON → VocalEvent + tape. Null F0 stays None.
- `prosody.py` — Relative arousal from the Corti card. No valence.
- `structural_affect.py` — Clause-level affect. Structure decides.
- `harm_frame.py` — Who did what to whom. Grief is not crisis.
- `tape_contour.py` — Tape contour (null F0 is a hole)
- `recognition_result.py` — Honest recognition contract

#### Engines (`engines/`)
- `base_synthesizer.py` — Abstract TTS engine
- `xtts_engine.py` — Coqui XTTS voice synthesis
- `gpt_sovits_engine.py` — GPT-SoVITS synthesis engine

#### Synthesis (`synthesis/`)
- `voice_synthesis.py` — Core synthesis pipeline
- `voice_synthesis_orchestrator.py` — Synthesis coordination
- `tts_service.py` — TTS service layer
- `phoneme_labeler.py` — Phoneme timing and alignment
- `speech_response.py` — Response generation and streaming

#### Tone (`tone/`)
- `tone_analyzer.py` — Tone and emotion detection
- `tonescore_engine.py` — ToneScore calculation (file path; not a live producer on Corti)
- `tonescore_analyzer.py` — Detailed tone analysis
- `emotion_quantifier.py` — Emotion metric computation
- `emotion_embedder.py` — Emotion representation
- `christman_tone_engine_v2.py` — Production tone engine
- `speech_personality.py` — Personality extraction from speech
- `written_tone.py` — Tone analysis for text
- `emotion_labels.py` — Labels from model.config.id2label

#### Timbre (`timbre/`)
- `timbre_modeler.py` — Voice timbre modeling
- `shorty_emotion.py` — Emotion-based timbre modification
- `voicepack.py` — Voice pack management

#### Music (`music/`)
- `music_engine.py` — Music generation and synthesis
- `christman_studio.py` — Music studio orchestration

#### Nonverbal (`nonverbal/`)
- `nonverbal_engine.py` — Named AAC map engine
- `engine_temporal.py` — Temporal sequence classification

#### Integration (`integration/`)
- `christman_speech_to_speech.py` — Speech-to-speech conversion
- `speech_integration.py` — Speech service integration
- `voice_capture_client.py` — Voice capture and streaming

#### Utils (`utils/`)
- `voice_diagnostics.py` — Audio quality analysis
- `grounder.py` — Context and grounding utilities
- `presence_guide.py` — User presence detection

### Core Modules
- `core.py` — Main framework initialization
- `logger.py` — Unified logging
- `christman_dsp.c` — DSP operations (compiled)

## Quick Start

### Basic Voice I/O

```python
from CHRISTMAN_EAR_CANAL import listen, speak, analyze_tone

# Listen to user
wav = listen(max_duration=8)

# Analyze tone
tone = analyze_tone(wav)
print(tone)

# Respond
speak("I understood you, Everett.", emotion="warm")
```

### Speech Recognition

```python
from christman_voice_sdk.audio import enhanced_speech_recognition

recognizer = enhanced_speech_recognition.EnhancedSpeechRecognizer()
text = recognizer.recognize(wav_file="input.wav")
print(f"Recognized: {text}")
```

### Voice Synthesis

```python
from christman_voice_sdk.synthesis import voice_synthesis

synthesizer = voice_synthesis.VoiceSynthesizer(engine="xtts")
wav = synthesizer.synthesize(
    text="Hello, I am Seraphinia",
    voice_profile="seraphinia",
    emotion="curious"
)
```

### Speech-to-Speech

```python
from christman_voice_sdk.integration import christman_speech_to_speech

converter = christman_speech_to_speech.ChristmanSpeechToSpeech()
output_wav = converter.convert(
    input_wav="user_voice.wav",
    target_voice="derek",
    emotion="authoritative"
)
```

### OCR & Screen Reading

```python
from CHRISTMAN_EAR_CANAL import scan_screen, scan_document

# Read screen
result = scan_screen(being="AlphaVox")
print(result["text"])

# Scan document
result = scan_document("/path/to/document.pdf", being="Seraphinia")
```

## Configuration

### Environment Variables

```bash
# MCP Server root (Derek configuration)
export DEREK_ROOT=/path/to/DerekMCPServer

# Voice SDK root
export CHRISTMAN_VOICE_SDK_ROOT=/path/to/christman_voice_sdk

# TTS Engine preference (xtts, gpt-sovits)
export CHRISTMAN_TTS_ENGINE=xtts

# Audio device settings
export CHRISTMAN_AUDIO_DEVICE=0
export CHRISTMAN_SAMPLE_RATE=16000
```

### Voice Profiles

```python
from CHRISTMAN_EAR_CANAL import capture_voice_profile, list_voice_profiles

# Capture new profile
capture_voice_profile("being_name", duration=8)

# List available profiles
profiles = list_voice_profiles()
print(profiles)
```

## Architecture

```
User Input
    ↓
[CHRISTMAN_EAR_CANAL] (High-level API)
    ↓
[christman_voice_sdk] (Production engines)
    ├── Audio → Speech Recognition → Text
    ├── Text → Synthesis → Audio
    ├── Audio → Tone Analysis → Emotions
    └── Audio → Feature Extraction → Voice Profile
    ↓
User Output (Speech, Sound, Feedback)
```

## Key Features

- **Multi-Engine Support** — XTTS, GPT-SoVITS, macOS fallback
- **Real-Time Processing** — Streaming speech recognition and synthesis
- **Honesty Rule** — Truthful reporting of engine capabilities
- **No Fakes** — Fallback modes clearly indicated, never pretending
- **Voice Profiles** — Frequency-based speaker identification

## Beings

Christman Sound supports the following autonomous beings:
- Derek
- AlphaVox
- AlphaWolf
- Brockston
- Geo
- Seraphinia

Each being can have custom voice profiles, emotional ranges, and personality settings.

## Requirements

- Python 3.8+
- macOS 11+ (for fallback speech synthesis)
- XTTS models (auto-downloaded on first use)
- FFmpeg (for audio encoding)

## Installation

Both kinds of editable:

```bash
# local clone — edits land immediately
pip install -e /Users/EverettN/Christman-Sound

# git clone in a temp tree, still editable
pip install -e git+https://github.com/The-ChristmanAI-Project/Christman-Sound.git#egg=christman-sound
```

Frozen (no edit):

```bash
pip install git+https://github.com/The-ChristmanAI-Project/Christman-Sound.git
```

Then:

```python
from christman_voice_sdk.nonverbal.acoustic_live import deliver_acoustic_key
from christman_voice_sdk.audio.hearing import Hearing
```

## Logging

All modules use unified logging via `logger.py`. Enable debug output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Honesty Rule — Cardinal Rule 13

This system never fakes capabilities. When speech synthesis falls back to macOS `say`, the returned result explicitly states `engine = "macos_say_fallback"`. No pretending. No deception.

## License

Proprietary — Christman Family of Autonomous Beings

## Support

For integration issues or audio problems, check `christman_voice_sdk/utils/voice_diagnostics.py`.
