# Christman Sound — Audio, Voice, and Speech

Christman Sound is the sound package of The Christman AI Project. It holds:

- **the C engine**: measurement (RMS, zero-crossing rate, YIN pitch, LPC), playback, the synthesizer and the frame vocoder;
- **the voice synthesis path**: `christman_synth`, the vocoder, stitch and frame voices, driven from Python by `christman_sound`;
- **ToneScore and voice measurement**: pitch, jitter, shimmer, HNR and the features ToneScore reads from raw audio;
- **the hearing modules**: a measured utterance card in, affect and harm frame out, and local speech recognition;
- **the ear** (`christman_sound.ear`): the package's own speech-to-text, in C;
- **reading aloud** (`christman_ocr_shared.py`): printed text on a screen, an image or a PDF, read so it can be spoken in the caller's voice.

The package holds no being's identity. A being passes its own voice profile or frame voice on every speaking call; with no voice given, nothing is spoken and the call raises a typed failure with the reason. No outside speech engine is ever substituted.

## Overview

```
measured utterance card + tape            words (WAV)            text
          ↓                                     ↓                   ↓
hearing.py + corti_ingest.py       speech_recognition_engine    the mouth (voice_synthesis)
   ├─ tape_contour / prosody  (arousal only, relative)              ↓
   ├─ structural_affect       (clauses, not bag-of-words)    christman_sound: lexicon → synth / stitch
   └─ harm_frame              (who did what to whom)                ↓
          ↓                                                  christman_dsp.so (C) → sound
   fusion + face  (decide, or MODE_UNKNOWN)
```

`listen()` (`voice_capture_client.capture_audio`) is a timed microphone grab. It is not hearing.

## The hearing contract

Hearing consumes a measured utterance card and its tape. The ear that produces them measures; it does not classify. Measurements cross into this package; client labels do not.

**Per-utterance card**, ring of 8:
`kind` · `duration` · `attack` · `decay` · `peak RMS` · `median F0` (or null) ·
`last F1/F2` · `mean ZCR` · `voiced`

**`tape: TapeFrame[]`**, cap 360:
`{ t: ms from onset, rms, zcr, f0: number | null }`

Rules that hold whatever consumes it:

- **The cap drops the TAIL.** Past roughly 6 s the ending is gone. Every end-of-utterance feature must be `None` when `truncated`.
- **`t` is `performance.now()` — about 16 ms per animation frame, and NOT uniform.** Slopes must use real `t`. A gap in `t` is a dropped frame, not a pause.
- **A null `f0` is a hole, not a zero.** Never interpolate across it.
- **Jitter, shimmer, and HNR are per-frame only.** They die with the frame.
- **Cold start: no speaker, no baseline.**
- **`kind` does not enter this package.** It is the client's word. Dropped at ingest.

Do not wire the card source as a `SoundDetectorBackend`.

## Package Structure

### christman_voice_sdk

#### Audio (`audio/`)
- `hearing.py` — Composes ingest, affect and harm frame over a measured card. Opens no microphone
- `corti_ingest.py` — Card JSON → VocalEvent + tape. Null F0 stays None
- `corti_receiver.py` — Localhost receiver (127.0.0.1:5486) that seats a posted card under Hearing. Not running unless you start it
- `word_ear_receiver.py` — Localhost-only receiver for words: one WAV in, local recognition out. It has no default port; pass `--port` or set `CHRISTMAN_WORD_EAR_PORT` (5173, 5485, 5486, 5487 and 5497 are refused)
- `prosody.py` — Relative arousal from the card. No valence
- `tape_contour.py` — Tape contour (null F0 is a hole)
- `structural_affect.py` — Clause-level affect. Structure decides
- `harm_frame.py` — Who did what to whom. Grief is not crisis
- `fusion_engine.py` — Carbon and silicon signals fused: decide, or unknown
- `face_reading.py` — What Sound hands the face: structure and arousal, no emotion names
- `face_vector.py` — The 8-dim face vector Sound hands the Soul Mirror, with its measured mask and the clauses it came from
- `body_signals.py` — The body-signal input contract
- `speech_recognition_engine.py` — Local speech recognition through `christman_sound.ear`. Audio never leaves the machine
- `enhanced_speech_recognition.py` — Coordinates speech and nonverbal sound recognition; recognizes nothing itself
- `real_speech_recognition.py` — Voice activity detection and capture; detects that someone spoke
- `sound_recognition_service.py` — Sound pattern recognition for nonverbal communicators
- `recognition_result.py` — The recognition result contract every recognizer returns
- `audio_processor.py` — Raw audio intake: noise reduction, segmentation, quality analysis
- `audio_encoder.py` — Audio encoder: mel spectrogram to audio embedding
- `mouth.py` — The being's own mouth: its voice in, one mouth embedding per video frame out. Local checkpoint only
- `config.py` — SDK configuration. Its folders live under `$CHRISTMAN_AI_HOME/voice_sdk` (default `~/.christman_ai/voice_sdk`) and are made only when needed

#### Engines (`engines/`)
- `synthesis_result.py` — The result contract (`SynthesisResult`) and the typed failure (`SynthesisUnavailable`) a speaking path raises when no voice can be made
- `logger.py` — Logger bridge

#### Synthesis (`synthesis/`)
- `voice_synthesis.py` — The mouth. Speaks text in the voice its caller gives, through `christman_sound.synth`
- `tts_service.py` — Local Flask service (127.0.0.1) in front of the mouth
- `phoneme_labeler.py` — Phoneme and viseme timing
- `speech_response.py` — `speak()`; raises `SynthesisUnavailable`. It does not reach the mouth; use `voice_synthesis`
- `reply_guard.py` — Lines a being never says to a client ("you're not okay", "calm down"); runs on the being's own reply only, never on the client's words
- `shorty_voice_engine_v3.py` — Emotion quantification; its `synthesize()` raises `SynthesisUnavailable`
- `voice_synthesis_orchestrator.py` — Voicepack, timbre and emotion pipeline; its `synthesize()` raises `SynthesisUnavailable`

#### Tone (`tone/`)
- `voice_measure.py` — The one place the SDK measures a voice: pitch track (YIN in C), voice quality (jitter, shimmer, HNR by Praat), RMS contour
- `tonescore_engine.py` — ToneScore: raw audio → measured features → adaptive response mode
- `tonescore_analyzer.py` — Multi-layer tone analysis: pitch, jitter, shimmer, HNR, through `voice_measure`
- `christman_tone_engine_v2.py` — Emotion and ToneScore engine v2. Its classifier loads only from local files; nothing is downloaded
- `emotion_quantifier.py` — Stress, coherence, grounding and crisis signals from text
- `emotion_embedder.py` — Maps emotional states to voice parameters
- `emotion_labels.py` — Labels from model.config.id2label
- `analyze_emotion.py` — Emotional tagging for a memory file
- `tone_manager.py` — Communication adjustments after tone is read
- `tone_classifier.py` — Superseded; delegates to `written_tone.py`
- `written_tone.py` — Tone analysis for text

#### Timbre (`timbre/`)
- `timbre_modeler.py` — Voice timbre measurement, through `voice_measure` and `christman_sound.formants`
- `shorty_emotion.py` — Emotion scores from a local wav2vec2 model and a PCA projection. Loads only from local paths the caller gives; nothing is downloaded. No projection file is in this repository
- `voicepack.py` — Voicepack archives. Every member is checked before anything is extracted
- `logger.py` — Logger bridge

#### Music (`music/`)
- `music_engine.py` — Melodies and rhythms as composition data. Makes no sound
- `christman_studio.py` — Song arrangements as data. Makes no sound. Rendering, mixing, mastering and singing are not built (`NOT_BUILT`)
- `resonance_player.py` — Volume-capped player: peak −12 dBFS, RMS −30 dBFS, fades, checked before it plays

#### Nonverbal (`nonverbal/`)
- `nonverbal_engine.py` — Named AAC map engine
- `engine_temporal.py` — Temporal sequence classification
- `acoustic_live.py` — Delivers a live acoustic key from the card source; measures nothing itself

#### Integration (`integration/`)
- `voice_capture_client.py` — Microphone capture, frequency signature, voice profiles

#### Utils (`utils/`)
- `grounder.py` — Grounding scripts (breathing, 5-4-3-2-1, feet on the ground) chosen from stress metrics
- `presence_guide.py` — Reads text for states that need presence, not solutions, and checks a reply for toxic positivity
- `voice_diagnostics.py` — Runs a local synthesis backend over voice profiles and reports byte-identical outputs
- `logger.py` — Logging

### christman_sound — the voice

- `synth.py` — `synthesize()` and `speak()`: text or phones in, sound out. `voice=` is required: a `VoiceProfile`, a `FrameVoice`, or a path to either; with none, `NoVoice` is raised
- `stitch.py` — Frame voices: a voice speaking new sentences from its own recordings
- `lexicon.py` — Words to phones, from a Vosk alignment lexicon
- `formants.py` — F1, F2, F3 and their bandwidths, measured from a recording
- `voice_profile.py` — A measured voice profile (`load_voice_profile(path)`), applied to the synthesizer. Each profile carries its file's SHA-256 (`fingerprint()`)
- `dsp.py` — The one loader for the C library
- `logger.py` — Logging
- `ear/` — The ear: the package's own speech-to-text. Its C library is built once per machine with `python -m christman_sound.ear.build --check`
- `core.py` — Kept for the record. Nothing imports it, and it is not touched (Everett, 2026-10-11)

### C engine (repository root)

- `christman_dsp.c` — Measurement (RMS, ZCR, YIN pitch, LPC) and playback
- `christman_synth.c` — The synthesizer, ported from Codec War
- `christman_vocoder.c` — The frame vocoder, ported from Codec War's `vocoder.ts`

All three are compiled into `christman_sound/christman_dsp.so` by `scripts/build_dsp.py`. pip does not build them (see Installation).

### Reading aloud (repository root)

- `christman_ocr_shared.py` — OCR (PaddleOCR) for a screen, an image or a PDF. What it reads is spoken by `christman_sound.synth.speak` in the voice the caller passes (`voice=` is required); when it cannot be voiced it raises `SynthesisUnavailable` with the reason

### Scripts

- `scripts/build_dsp.py` — Builds the C library; `--check` runs its smoke test
- `scripts/build_voice_bank.py` and `scripts/voice_bank_driver.mjs` — Build a frame voice (below)
- `scripts/measure_voice.py` — Measures a being's own recordings into a voice profile (`--recordings`, `--out`, `--name`)
- `scripts/run_crisis_table.py`, `scripts/verify_corti_wire.py`, `scripts/verify_hearing_seat.py` — Checks run by hand

## Quick Start

### Basic Voice I/O

```python
from christman_voice_sdk import listen                      # timed mic grab; needs the capture extra
from christman_voice_sdk.integration.voice_capture_client import (
    CAPTURE_SAMPLE_RATE, extract_frequency_signature,
)
from christman_voice_sdk.audio.hearing import Hearing

audio = listen(8)                                           # 8 s at CAPTURE_SAMPLE_RATE (44.1 kHz), float32
print(extract_frequency_signature(audio, CAPTURE_SAMPLE_RATE))  # None for anything that was not measured


def page_a_human(safety_result, context) -> bool:
    ...                                                     # return True ONLY once a person was reached
    return True


hearing = Hearing(crisis_callback=page_a_human)
turn = hearing.hear(text="scared")                          # a measured card can be passed as corti_event=
print(turn.decision.mode, turn.decision.safety.status)      # gentle-lift clear
```

### Speech Recognition

Recognition is local. `christman_sound.ear` is the package's own speech-to-text, and the SDK's `speech_recognition_engine.py` hears through it (2026-10-29: the old recognizer is removed). Point it at a model folder with `VOSK_MODEL_PATH` and build the ear's C library once (`python -m christman_sound.ear.build --check`). Audio at a rate other than 16 kHz is resampled to 16 kHz first, and the result says so.

```python
from christman_voice_sdk.audio.speech_recognition_engine import get_speech_recognition_engine

engine = get_speech_recognition_engine()                    # unavailable, with the reason, if no model
result = engine.recognize_from_bytes(pcm16_mono, sample_rate=16000)
print(result.status, result.text)                           # text is "" on every non-OK status
```

### Voice Synthesis

The mouth speaks only in the voice its caller gives: `voice=` on `SpeechSynthesisEngine(...)` or on each `generate_speech_audio(...)` call, as a voice profile file, a frame voice folder, a `VoiceProfile` or a `FrameVoice`. There is no default voice. Controls are keywords (cadence, breath, jitter, shimmer, timbre, articulation, intonation, prosody, resonance, affect, rate, f0); each is applied and listed in `last_applied`, or refused with the reason.

```python
from christman_voice_sdk.synthesis.voice_synthesis import SpeechSynthesisEngine
from christman_voice_sdk.engines.synthesis_result import SynthesisUnavailable

mouth = SpeechSynthesisEngine(lexicon_path="<align_lexicon.txt>")   # or $CHRISTMAN_LEXICON_PATH
try:
    wav_path = mouth.generate_speech_audio(
        "I am here.",
        voice="~/.christman_ai/voices/<voice>",             # the being's own frame voice
        play_audio=False,                                   # True plays it, or raises PlaybackFailed
        cadence=0.5,
    )
    print(wav_path, mouth.last_applied)
except SynthesisUnavailable as err:
    # No outside voice is ever substituted. err.available is False and
    # err.reason says why (for example, no voice was given).
    print(err.reason)
```

A being that has a measured voice profile instead of a frame voice passes it straight to the synthesizer:

```python
from christman_sound.synth import synthesize
from christman_sound.voice_profile import load_voice_profile

voice = load_voice_profile("<the being's voice profile>.json")  # a missing field raises VoiceProfileError
result = synthesize("I am here.", voice=voice, lexicon_path="<align_lexicon.txt>")   # voice= is required
result.write_wav("here.wav")
```

A lexicon problem (none configured, a word not in it) raises `christman_sound.lexicon.LexiconError` and a control the voice cannot apply raises `christman_sound.stitch.StitchRefused`, each with the reason; nothing is dropped silently. `synthesis_readiness()` says whether a voice and lexicon are ready, and why not.

### A frame voice: a voice spoken from its own recordings

A frame voice is a folder built from one voice's own recordings and nothing else. It holds the recordings measured every 4.99 ms (pitch, a 96-point spectral envelope, aperiodicity in 8 bands), every word `christman_sound.ear` heard in them cut into phone units, and the voice's measured profile (pitch, phrase pitch range, syllable rate, breath, long-term spectrum, level, and jitter, shimmer, HNR and CPPS by Praat through the SDK). New sentences are stitched from stretches of those frames (`christman_sound.stitch`, the port of Codec War's `stitch.ts`) and rendered by `christman_vocoder.c` (the port of Codec War's `vocoder.ts`; the tests hold both ports to Codec War sample for sample).

The folder is built on the machine that speaks; it is not kept in git (a built voice runs to tens of megabytes). Build it with the repository's script:

```bash
# every recording 22050 Hz, 16-bit, mono WAV; on a Mac, convert with
#   afconvert -f WAVE -d LEI16@22050 -c 1 in.wav out.wav
python3 scripts/build_dsp.py
python3 scripts/build_voice_bank.py \
    --name <VoiceName> --out ~/.christman_ai/voices/<voice> \
    --recordings <the voice's recordings: folders or WAVs> \
    --profile-from <the recordings its profile is measured from> \
    --codecwar <a CODECWAR5TH checkout> \
    --vosk-model ~/models/vosk-model-en-us-0.22 \
    --lexicon ~/models/vosk-model-en-us-0.22/graph/phones/align_lexicon.txt
```

It needs node (Codec War's own analysis runs under it), the ear's C library, numpy and praat-parselmouth. Only the voice's own recordings go in: never another being's, never a person's. `--measure-only` keeps a built bank and measures the profile again.

### ToneScore and voice measurement

```python
from christman_voice_sdk.tone.tonescore_engine import ToneScoreEngine
from christman_voice_sdk.tone.voice_measure import read_wav, pitch_contour, voice_quality

y, sr = read_wav("utterance.wav")
track = pitch_contour(y, sr)                                # YIN in the C engine; unvoiced frames stay unvoiced
print(track.summary())
print(voice_quality(y, sr).to_dict())                       # jitter, shimmer, HNR by Praat; PraatUnavailable if absent

result = ToneScoreEngine().analyze_tone("utterance.wav")
print(result.to_dict())
```

### Reading printed text aloud

```python
import asyncio
from christman_ocr_shared import ChristmanOCR               # needs paddleocr (+ pymupdf for PDFs)

ocr = ChristmanOCR(voice="~/.christman_ai/voices/<voice>",  # the caller's own voice: required
                   lexicon_path="<align_lexicon.txt>")
result = asyncio.run(ocr.read_document("/path/to/document.pdf"))
print(result["pages"], result["confidence"], result["text"])
```

PaddleOCR fetches its model files the first time it is constructed; that is an open item against the offline rule (`tests/test_offline.py`).

## Configuration

### Environment Variables

```bash
# The voice
export CHRISTMAN_LEXICON_PATH=<vosk model>/graph/phones/align_lexicon.txt
export CHRISTMAN_VOICE_CACHE=~/.christman_ai/voice_cache     # where spoken WAVs are written
export CHRISTMAN_DSP_LIB=/path/to/christman_dsp.so           # only to override where the library is found

# Hearing
export VOSK_MODEL_PATH=~/models/vosk-model-en-us-0.22        # the model folder christman_sound.ear reads
export CHRISTMAN_SOUND_PORT=5486                             # corti_receiver
export CHRISTMAN_WORD_EAR_PORT=<port>                        # word_ear_receiver; no default

# The SDK's own folders: config folders, voicepacks, music memory
export CHRISTMAN_AI_HOME=~/.christman_ai
export CHRISTMAN_LOG_DIR=~/.christman_ai/logs                # logs
```

### Voice Profiles

```bash
python3 -m christman_voice_sdk.integration.voice_capture_client --capture <profile-name>   # 8 s from the mic
python3 -m christman_voice_sdk.integration.voice_capture_client --list
python3 -m christman_voice_sdk.integration.voice_capture_client --speak "I'm here." \
    --voice <voice profile file or frame voice folder> --lexicon <align_lexicon.txt>
```

A profile stores only what was measured. Silence yields `fundamental_hz: null`, never 150. A being's profile is made from that being's own voice. `--speak` goes through the package's own mouth in the voice given with `--voice`.

## Architecture

```
Client input (measured card, words, text)
    ↓
[christman_voice_sdk]
    ├── Card + tape → hearing → affect, harm frame → decision
    ├── Audio → local speech recognition → text
    ├── Audio → tone and voice measurement → measured features
    └── Text → the mouth (voice_synthesis) ─┐
                                            ↓
[christman_sound]  lexicon → synth / stitch → christman_dsp.so (C) → sound
    ↓
Client output (the being's own voice, from the profile the being passed)
```

## Key Features

- **Own Voice Only** — No outside speech engine (no XTTS, GPT-SoVITS, `say`, espeak, pyttsx3, gTTS or pygame). A being speaks in its own voice or raises `SynthesisUnavailable` with the reason
- **Offline** — `tests/test_offline.py` fails on any new download
- **Honesty Rule** — Truthful reporting of engine capabilities
- **No Fakes** — A path that cannot do its work says so, never pretending
- **Voice Profiles** — Measured values only, stored as JSON

## Requirements

- Python 3.10+
- A C compiler (Xcode Command Line Tools on a Mac) for `scripts/build_dsp.py`

## Installation

```bash
# local clone — edits land immediately
pip install -e .
python3 scripts/build_dsp.py --check        # builds the C library; pip does not
python -m christman_sound.ear.build --check # builds the ear's C library, once per machine

# extras
pip install -e '.[crypto]'                  # voicepack encryption
pip install -e '.[capture]'                 # microphone capture (sounddevice)
```

From git, still editable:

```bash
pip install -e git+https://github.com/The-ChristmanAI-Project/Christman-Sound.git#egg=christman-sound
```

Frozen (no edit):

```bash
pip install git+https://github.com/The-ChristmanAI-Project/Christman-Sound.git
```

Then run `scripts/build_dsp.py` from a clone, so `christman_sound/christman_dsp.so` exists. Without it, measurement and the voice raise `DSPUnavailable` with the reason.

librosa is not a dependency: nothing imports it, and its numba and llvmlite chain has no Intel Mac wheel.

## Logging

Modules log through Python's `logging`; most add a `NullHandler` and print nothing until the application configures logging. `christman_voice_sdk.utils.logger` adds rich console output and rotating files under `~/.christman_ai/logs` (`CHRISTMAN_LOG_DIR`). Enable debug output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Honesty Rule — Cardinal Rule 13

This system never fakes capabilities. When no voice can be made, every speaking path raises `SynthesisUnavailable` (`available` is False, `reason` says why). There is no fallback voice. No pretending. No deception.

## License

Proprietary — Christman Family of Autonomous Beings

## Tests

```bash
python3 -m pytest tests -q
```

A test that cannot run here reports **NOT RUN** with the reason; it never passes silently. Nothing in the suite plays out loud unless asked: a test that plays through the speakers carries `@pytest.mark.plays_aloud` and runs only with `CHRISTMAN_SOUND_PLAY_ALOUD=1` set. Any other test that reaches the speakers fails before a sample is played (`tests/conftest.py`). Render to a file (`play_audio=False`) instead.

```bash
CHRISTMAN_SOUND_PLAY_ALOUD=1 python3 -m pytest tests -q -m plays_aloud   # only when you mean to hear it
```

## Support

For integration issues or audio problems, run the tests (`python3 -m pytest tests -q`) and `python3 scripts/build_dsp.py --check`. `synthesis_readiness()` in `voice_synthesis` says whether the voice is ready and why not.
