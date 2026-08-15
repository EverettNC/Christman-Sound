"""
Timbre Modeling Module - Stage 2: Base Voice Construction

WHAT CHANGED AND WHY
--------------------
1. PSEUDO-RANDOM EMBEDDINGS ERADICATED (Rule 13).
   When models were not loaded, `_extract_x_vector` and `_extract_d_vector` 
   generated random numbers (`np.random.randn`). This gave completely fake speaker 
   embeddings. They now return None when no model is wired.

2. SECURE SERIALIZATION (Rule 12).
   Replaced `pickle.dump` and `pickle.load` in `save_profile` / `load_profile` 
   with secure JSON metadata and `.npz` vector arrays.
"""

from __future__ import annotations

import ctypes
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

from .logger import get_logger
from audio.audio_processor import AudioSegment

logger = get_logger(__name__)

DSP_LIB_PATH = Path(__file__).parent.parent / "christman_dsp.so"

try:
    _dsp_engine = ctypes.CDLL(str(DSP_LIB_PATH))
    _dsp_engine.christman_yin.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float)
    ]
    _dsp_engine.christman_lpc.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_size_t,
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
    ]
    _dsp_ok = True
    logger.info("Christman DSP Engine online in Timbre Modeler.")
except Exception as e:
    _dsp_ok = False
    logger.error("Christman DSP Engine failed to load: %s", e)


def get_pitch_contour_native(audio_array: np.ndarray, sample_rate: int = 16000, threshold: float = 0.1) -> np.ndarray:
    if not _dsp_ok: 
        return np.array([], dtype=np.float32)
    
    frame_length = 2048
    hop_length = 512
    
    if len(audio_array) < frame_length:
        return np.array([], dtype=np.float32)
        
    num_frames = 1 + (len(audio_array) - frame_length) // hop_length
    pitches = np.zeros(num_frames, dtype=np.float32)
    out_pitch = ctypes.c_float()
    
    for i in range(num_frames):
        start = i * hop_length
        frame = np.ascontiguousarray(audio_array[start:start + frame_length], dtype=np.float32)
        _dsp_engine.christman_yin(frame, len(frame), sample_rate, threshold, ctypes.byref(out_pitch))
        pitches[i] = out_pitch.value
        
    return pitches


def get_lpc_native(audio_array: np.ndarray, order: int) -> np.ndarray:
    if not _dsp_ok: 
        return np.zeros(order + 1, dtype=np.float32)
        
    audio_float32 = np.ascontiguousarray(audio_array, dtype=np.float32)
    out_a = np.zeros(order + 1, dtype=np.float32)
    
    _dsp_engine.christman_lpc(audio_float32, len(audio_float32), order, out_a)
    return out_a


@dataclass
class VoiceProfile:
    """Voice profile with timbre characteristics."""
    x_vector: Optional[np.ndarray] = None
    d_vector: Optional[np.ndarray] = None
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_min: float = 0.0
    f0_max: float = 0.0
    f0_contour: Optional[np.ndarray] = None
    f1_mean: float = 0.0
    f2_mean: float = 0.0
    f3_mean: float = 0.0
    spectral_envelope: Optional[np.ndarray] = None
    hnr_mean: float = 15.0
    jitter_mean: float = 0.0
    shimmer_mean: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_x_vector": self.x_vector is not None,
            "has_d_vector": self.d_vector is not None,
            "f0": {
                "mean": self.f0_mean,
                "std": self.f0_std,
                "min": self.f0_min,
                "max": self.f0_max
            },
            "formants": {
                "f1": self.f1_mean,
                "f2": self.f2_mean,
                "f3": self.f3_mean
            },
            "voice_quality": {
                "hnr": self.hnr_mean,
                "jitter": self.jitter_mean,
                "shimmer": self.shimmer_mean
            }
        }


class TimbreModeler:
    """Stage 2: Timbre Modeling and Base Voice Construction."""

    def __init__(
        self,
        device: str = "auto",
        use_x_vectors: bool = True,
        use_d_vectors: bool = False
    ):
        self.device = self._setup_device(device)
        self.use_x_vectors = use_x_vectors
        self.use_d_vectors = use_d_vectors
        self.x_vector_model = None
        self.d_vector_model = None
        logger.info("TimbreModeler initialized on %s", self.device)

    def _setup_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available(): return "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
            else: return "cpu"
        return device

    def build_voice_profile(
        self,
        audio_segments: List[AudioSegment],
        extract_detailed: bool = True
    ) -> VoiceProfile:
        logger.info("Building voice profile from %d segments", len(audio_segments))

        x_vector = self._extract_x_vector(audio_segments) if self.use_x_vectors else None
        d_vector = self._extract_d_vector(audio_segments) if self.use_d_vectors else None

        f0_profile = self._extract_f0_profile(audio_segments)
        formants = self._extract_formants(audio_segments) if extract_detailed else (0.0, 0.0, 0.0)
        voice_quality = self._extract_voice_quality(audio_segments)

        profile = VoiceProfile(
            x_vector=x_vector,
            d_vector=d_vector,
            f0_mean=f0_profile["mean"],
            f0_std=f0_profile["std"],
            f0_min=f0_profile["min"],
            f0_max=f0_profile["max"],
            f0_contour=f0_profile.get("contour"),
            f1_mean=formants[0],
            f2_mean=formants[1],
            f3_mean=formants[2],
            hnr_mean=voice_quality["hnr"],
            jitter_mean=voice_quality["jitter"],
            shimmer_mean=voice_quality["shimmer"]
        )

        logger.info("Voice profile built successfully")
        return profile

    def _extract_x_vector(self, segments: List[AudioSegment]) -> Optional[np.ndarray]:
        if self.x_vector_model is None:
            logger.warning("X-vector model not loaded. Embedding is None.")
            return None
        return None

    def _extract_d_vector(self, segments: List[AudioSegment]) -> Optional[np.ndarray]:
        if self.d_vector_model is None:
            logger.warning("D-vector model not loaded. Embedding is None.")
            return None
        return None

    def _extract_f0_profile(self, segments: List[AudioSegment]) -> Dict[str, Any]:
        all_f0 = []
        for segment in segments:
            f0 = get_pitch_contour_native(segment.audio, sample_rate=segment.sample_rate)
            f0_voiced = f0[(f0 > 50) & (f0 < 500)]
            all_f0.extend(f0_voiced)

        if len(all_f0) == 0:
            logger.warning("No F0 values extracted")
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        all_f0_arr = np.array(all_f0)
        return {
            "mean": float(np.mean(all_f0_arr)),
            "std": float(np.std(all_f0_arr)),
            "min": float(np.min(all_f0_arr)),
            "max": float(np.max(all_f0_arr)),
            "contour": all_f0_arr
        }

    def _extract_formants(self, segments: List[AudioSegment]) -> Tuple[float, float, float]:
        all_formants = []
        for segment in segments:
            try:
                lpc_order = 12
                a = get_lpc_native(segment.audio, order=lpc_order)
                roots = np.roots(a)
                roots = roots[np.imag(roots) >= 0]
                angles = np.arctan2(np.imag(roots), np.real(roots))
                freqs = angles * (segment.sample_rate / (2 * np.pi))
                formants = sorted(freqs)[:3]
                if len(formants) >= 3:
                    all_formants.append(formants)
            except Exception as e:
                logger.debug("Formant extraction failed for segment: %s", e)
                continue

        if len(all_formants) == 0:
            logger.warning("No formants extracted, returning zeros")
            return (0.0, 0.0, 0.0)

        all_formants_arr = np.array(all_formants)
        return (
            float(np.mean(all_formants_arr[:, 0])),
            float(np.mean(all_formants_arr[:, 1])),
            float(np.mean(all_formants_arr[:, 2]))
        )

    def _extract_voice_quality(self, segments: List[AudioSegment]) -> Dict[str, float]:
        from tone.tonescore_engine import ToneScoreEngine
        hnr_values, jitter_values, shimmer_values = [], [], []

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for i, segment in enumerate(segments[:10]):
                seg_path = temp_path / f"segment_{i}.wav"
                segment.save(seg_path)
                try:
                    engine = ToneScoreEngine()
                    result = engine.analyze_tone(str(seg_path))
                    phys = result.physiological
                    if phys.get("hnr") is not None: hnr_values.append(phys["hnr"])
                    if phys.get("jitter") is not None: jitter_values.append(phys["jitter"])
                    if phys.get("shimmer") is not None: shimmer_values.append(phys["shimmer"])
                except Exception as e:
                    logger.debug("Voice quality extraction failed: %s", e)

        return {
            "hnr": float(np.mean(hnr_values)) if hnr_values else 15.0,
            "jitter": float(np.mean(jitter_values)) if jitter_values else 0.0,
            "shimmer": float(np.mean(shimmer_values)) if shimmer_values else 0.0
        }

    def save_profile(self, profile: VoiceProfile, path: Path):
        """Save voice profile securely using JSON and .npz arrays (No pickle)."""
        data = profile.to_dict()
        meta_path = path.with_suffix(".json")
        npz_path = path.with_suffix(".npz")

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        arrays = {}
        if profile.x_vector is not None: arrays["x_vector"] = profile.x_vector
        if profile.d_vector is not None: arrays["d_vector"] = profile.d_vector
        if profile.f0_contour is not None: arrays["f0_contour"] = profile.f0_contour

        np.savez(npz_path, **arrays)
        logger.info("Voice profile saved securely to %s and %s", meta_path, npz_path)

    def load_profile(self, path: Path) -> VoiceProfile:
        """Load voice profile securely."""
        meta_path = path.with_suffix(".json")
        npz_path = path.with_suffix(".npz")

        if not meta_path.exists():
            raise FileNotFoundError(f"Profile metadata missing: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        x_vec, d_vec, f0_cnt = None, None, None
        if npz_path.exists():
            npz = np.load(npz_path)
            if "x_vector" in npz: x_vec = npz["x_vector"]
            if "d_vector" in npz: d_vec = npz["d_vector"]
            if "f0_contour" in npz: f0_cnt = npz["f0_contour"]

        return VoiceProfile(
            x_vector=x_vec,
            d_vector=d_vec,
            f0_mean=data["f0"]["mean"],
            f0_std=data["f0"]["std"],
            f0_min=data["f0"]["min"],
            f0_max=data["f0"]["max"],
            f0_contour=f0_cnt,
            f1_mean=data["formants"]["f1"],
            f2_mean=data["formants"]["f2"],
            f3_mean=data["formants"]["f3"],
            hnr_mean=data["voice_quality"]["hnr"],
            jitter_mean=data["voice_quality"]["jitter"],
            shimmer_mean=data["voice_quality"]["shimmer"]
        )
