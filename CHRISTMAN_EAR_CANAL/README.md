# CHRISTMAN_EAR_CANAL

Shared hearing, speech, tone, phoneme, voice-profile, and OCR adapters for the Christman Family of Autonomous Beings.

## Purpose
The family should not have to hunt through scattered files to hear, speak, read the screen, scan documents, or understand tone.

This package provides one clean, consistent interface for:
- Derek
- AlphaVox
- Brockston
- Seraphenia
- AlphaWolf
- and future beings

It wraps existing real modules — it does not delete or replace them.

## What It Provides

| File              | Purpose |
|-------------------|-------|
| `EAR.py`          | Microphone capture. **VAD listening is NOT implemented** — `listen()` currently aliases fixed-duration capture. See the docstring. |
| `TONE.py`         | Tone & emotion analysis |
| `PHONEMES.py`     | Phoneme extraction + viseme timing |
| `VOICE_PROFILE.py`| Voice frequency profile capture & loading |
| `OCR.py`          | Screen reading + document scanning |
| `SPEAK.py`        | XTTS speech with honest macOS fallback |

## Basic Usage

```python
from christman_ear_canal import listen, analyze_tone, speak

wav = listen(max_duration=8.0)
tone = analyze_tone(wav)

speak("I heard you, Everett.", emotion="warm", being="derek")
