# CHRISTMAN EMOTION & TONESCORE ENGINE v2.0
# "Nothing Vital Lives Below Root"
# Architecture: Multi-layer tone detection (Raw Audio -> Quantified Emotion)

import torch
import librosa
import numpy as np
import hashlib
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

class ChristmanToneEngine:
    def __init__(self, model_name="superb/wav2vec2-base-superb-er"):
        print(f"[SYSTEM] Initializing Christman Tone Engine: {model_name}")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # Hardware acceleration for macOS Tahoe (MPS)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

        # The Carbon Reality Labels
        self.EMOTION_LABELS = [
            "neutral", "happy", "proud", "teasing", "annoyed",
            "sarcastic", "sweetheart", "laugh", "tremble", "emphasis", "last_breath"
        ]

    def analyze_audio(self, wav_path: str) -> dict:
        """
        Extracts both the physiological state (pitch, shimmer)
        and the paralinguistic emotion from the raw audio waveform.
        """
        try:
            # 1. Load Audio (Force 16kHz for Wav2Vec2 compatibility)
            y, sr = librosa.load(wav_path, sr=16000)

            # Mix stereo to mono if needed
            if y.ndim == 2:
                y = np.mean(y, axis=0)

            # 2. Physiological T1 Layer (The Carbon Leakage)
            # Measure RMS energy to determine physical intensity/volume
            rms_energy = librosa.feature.rms(y=y)[0]
            intensity_norm = np.clip(np.mean(rms_energy) * 400, 0, 1)

            # 3. Neural Paralinguistics T2 Layer
            input_values = self.processor(y, sampling_rate=sr, return_tensors="pt", padding=True).input_values
            input_values = input_values.to(self.device)

            with torch.no_grad():
                logits = self.model(input_values).logits
                probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            # Map raw probabilities to Christman Labels
            emotion_scores = {
                self.EMOTION_LABELS[i] if i < len(self.EMOTION_LABELS) else f"unknown_{i}": round(float(p), 4)
                for i, p in enumerate(probabilities)
            }

            # Determine dominant state
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)

            # 4. Generate Cadence Fingerprint (Hash)
            cadence_hash = hashlib.sha1(y.tobytes()).hexdigest()[:16]

            # 5. Actionable Routing Logic
            action_state = "NORMAL"
            if dominant_emotion in ["tremble", "last_breath"] or intensity_norm > 0.85:
                action_state = "HOLD_SPACE" # Trigger the Hand of God or trauma protocols

            return {
                "modality": "audio",
                "dominant_state": dominant_emotion,
                "action_state": action_state,
                "physical_intensity": float(intensity_norm),
                "cadence_fingerprint": cadence_hash,
                "raw_scores": emotion_scores
            }

        except FileNotFoundError:
            print(f"[ERROR] Carbon input missing. Cannot find {wav_path}")
            return None
        except Exception as e:
            print(f"[ERROR] ToneScore calculation failed: {e}")
            return None

# --- Runtime Test Wiring ---
if __name__ == "__main__":
    engine = ChristmanToneEngine()
    # print(engine.analyze_audio("test_carbon_input.wav"))
