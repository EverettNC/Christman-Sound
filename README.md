# Christman Voice SDK v1.0.0

**The Christman AI Project — Luma Cognify AI**
Patent Pending TCAP-2026-001 / TCAP-2026-002
© 2026 Everett Nathaniel Christman & Misty Gail Christman

---

## What This Is

A complete, self-contained voice intelligence SDK. Everything needed to run lives here.

**No hunting for dependencies. No missing imports. One install.**

---

## Package Structure

```
christman_voice_sdk/
├── tone/           # Emotion & prosody analysis (ToneScore™)
├── synthesis/      # Text-to-speech engines
├── engines/        # GPT-SoVITS, XTTS v2, base synthesizer
├── audio/          # Audio processing & speech recognition
├── timbre/         # Voice modeling, voicepack, Shorty emotion
├── nonverbal/      # Nonverbal, temporal, cochlear sync (accessibility)
├── music/          # Music engine & studio
├── integration/    # Voice capture, speech-to-speech, OCR
└── utils/          # Logger, diagnostics, presence guide, grounder
```

---

## Core Modules (32 total)

### Tone & Emotion
- `tone_analyzer.py` — 5-layer acoustic analysis (ToneScore™)
- `christman_tone_engine_v2.py` — Christman emotion engine
- `tonescore_engine.py` — ToneScore™ composite scoring
- `tonescore_analyzer.py` — Extended tone analysis
- `tone_manager.py` — Tone delivery management
- `tone_classifier.py` — Written tone classification
- `emotion_embedder.py` — Emotion → voice parameter mapping
- `emotion_quantifier.py` — Text-based emotion quantification
- `written_tone.py` — Aggressive vs incisive writing classifier
- `speech_personality.py` — Speech personality adaptation

### Synthesis
- `voice_synthesis_orchestrator.py` — Complete synthesis pipeline
- `voice_synthesis.py` — Multi-dialect TTS
- `tts_service.py` — TTS service with voice profiles
- `speech_response.py` — macOS speech response engine

### Synthesis Engines
- `gpt_sovits_engine.py` — GPT-SoVITS v3 (407M params)
- `xtts_engine.py` — XTTS v2 zero-shot voice cloning
- `base_synthesizer.py` — Abstract base interface

### Audio Processing
- `audio_processor.py` — Noise reduction, VAD, segmentation
- `audio_encoder.py` — CNN audio encoding
- `enhanced_speech_recognition.py` — Enhanced ASR
- `speech_recognition_engine.py` — Live mic + file recognition
- `fusion_engine.py` — Carbon-Silicon fusion (emotion + logic)

### Timbre & Voice Modeling
- `timbre_modeler.py` — Speaker embeddings, F0, formants
- `voicepack.py` — .voicepack format build & load
- `shorty_emotion.py` — 11-state personal emotion model

### Nonverbal & Accessibility ★
- `nonverbal_engine.py` — Gesture, eye movement, AAC support
- `engine_temporal.py` — Temporal nonverbal pattern analysis
- `cochlear_sync_tts.py` — Mouth/speech mechanics reference

### Music & Production
- `music_engine.py` — Composition, melody, rhythm generation
- `music_studio.py` — Multi-track production, mixing, mastering

### Integration & Utilities
- `voice_capture_client.py` — Stealth TTS relay, frequency capture
- `speech_integration.py` — Speech recognition integration
- `christman_speech_to_speech.py` — Full S2S pipeline
- `logger.py` — Structured logging with rich output
- `presence_guide.py` — Presence vs. problem-solving detection
- `grounder.py` — Grounding techniques for escalation states
- `voice_diagnostics.py` — Voice system diagnostics

---

## Installation

```bash
pip install -e .
```

---

## Quick Start

```python
from christman_voice_sdk.tone.tone_analyzer import get_tone_analyzer

analyzer = get_tone_analyzer()
result = analyzer.analyze_tone("audio.wav")
print(result["tone_score"])
print(result["response_mode"])
```

---

## Philosophy

Built for everyone. Deaf. Blind. Nonverbal. AAC users. Veterans.
Neurodivergent builders. People who communicate differently.

No one gets left out of the core. Not an afterthought. Not a plugin.
Built in from day one.

*"How can we help you love yourself more?"*

---

Patent Pending TCAP-2026-001 / TCAP-2026-002
© 2026 Everett Nathaniel Christman & Misty Gail Christman
The Christman AI Project — Luma Cognify AI
