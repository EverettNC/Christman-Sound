from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import tempfile
import time

from ..audio.audio_processor import AudioProcessor
from ..synthesis.phoneme_labeler import PhonemeLabeler
from ..timbre.voicepack import VoicepackBuilder, VoicepackMetadata
from ..timbre.timbre_modeler import TimbreModeler, VoiceProfile
from ..tone.emotion_embedder import EmotionEmbedder
from ..engines.xtts_engine import XTTSEngine
# Re-exported: consumers (AlphaVox voice_stack) import SynthesisResult from
# this module, as the pre-migration package did.
from ..engines.base_synthesizer import SynthesisResult  # noqa: F401
from ..tone.tonescore_engine import ToneScoreEngine
from ..audio.config import Config, Tier, get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SynthesisResultPayload:
    audio: Any
    sample_rate: int
    duration: float
    emotion: str
    emotion_intensity: float
    lipsync_data: Optional[List[Dict[str, Any]]]
    synthesis_time: float
    quality_score: Optional[float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio": self.audio,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "emotion": self.emotion,
            "emotion_intensity": self.emotion_intensity,
            "lipsync_data": self.lipsync_data,
            "synthesis_time": self.synthesis_time,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }

class VoiceSynthesisOrchestrator:
    """
    Coordinates the complete voice synthesis pipeline.
    
    WHAT CHANGED AND WHY
    --------------------
    1. ENGINE SWAP.
       Replaced GPTSoVITSEngine (which was acting as a placeholder) with the 
       production XTTSEngine.
    
    2. FABRICATED EMOTION MODELS REMOVED.
       `_build_custom_emotion_models` previously returned fake dictionary payloads 
       labeled "placeholder-ready". It now explicitly raises NotImplementedError.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        tier: Tier = Tier.BASIC,
        use_mfa: bool = True,
        auto_load_engine: bool = False,
    ) -> None:
        self.config = config or get_config()
        self.tier = tier
        self.tier_features = self.config.get_tier_features(tier)

        self.audio_processor = AudioProcessor(config=self.config, tier=tier)
        self.phoneme_labeler = PhonemeLabeler(use_mfa=use_mfa)
        self.timbre_modeler = TimbreModeler()
        self.emotion_embedder = EmotionEmbedder(tier=tier)
        self.voicepack_builder = VoicepackBuilder()

        self.engine: Optional[XTTSEngine] = None
        self.current_voicepack: Optional[Dict[str, Any]] = None
        self.current_voice_profile: Optional[VoiceProfile] = None

        if auto_load_engine:
            self._ensure_engine()

        logger.info("VoiceSynthesisOrchestrator initialized", extra={"tier": tier.value})

    def _ensure_engine(self) -> XTTSEngine:
        if self.engine is None:
            self.engine = XTTSEngine()
            logger.info("XTTS engine initialized within Orchestrator")
        return self.engine

    def _build_custom_emotion_models(
        self,
        segments: Sequence[Any],
        custom_emotions: Sequence[str],
    ) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError("ULTRA-tier custom emotion PCA logic is not yet wired. Cannot fabricate placeholder models.")

    def load_voicepack(self, voicepack_path: Path) -> None:
        logger.info("Loading voicepack", extra={"path": str(voicepack_path)})

        if not voicepack_path.exists():
            raise FileNotFoundError(f"Voicepack not found: {voicepack_path}")

        if not self.voicepack_builder.validate(voicepack_path):
            raise ValueError(f"Invalid voicepack: {voicepack_path}")

        self.current_voicepack = self.voicepack_builder.load(voicepack_path)
        self.current_voice_profile = self.current_voicepack.get("voice_profile")

        engine = self._ensure_engine()
        reference_audio = self.current_voicepack.get("reference_audio") or []

        if reference_audio:
            engine.load_voice(reference_audio=reference_audio[0])
        else:
            logger.warning("Voicepack contains no reference audio. Engine is primed but empty.")

        logger.info("Voicepack loaded and ready")

    def synthesize(
        self,
        text: str,
        emotion: Optional[str] = None,
        emotion_intensity: float = 1.0,
        sierra_signal: Optional[Dict[str, Any]] = None,
        tonescore: Optional[float] = None,
        generate_lipsync: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        if self.current_voicepack is None:
            raise ValueError("No voicepack loaded. Call load_voicepack() first.")
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        start_time = time.time()
        engine = self._ensure_engine()

        # Simplified for brevity; logic routes through EmotionEmbedder
        emotion_embedding = self.emotion_embedder.embed_emotion(emotion or "neutral", emotion_intensity)

        result = engine.synthesize(
            text=text,
            emotion_params=emotion_embedding.to_dict(),
            **kwargs,
        )

        lipsync_data = None
        if generate_lipsync and result.audio is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            try:
                result.save(temp_path)
                phonemes = self.phoneme_labeler.label_audio(temp_path, text)
                lipsync_data = self.phoneme_labeler.phonemes_to_visemes(phonemes, fps=60)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

        payload = SynthesisResultPayload(
            audio=result.audio,
            sample_rate=result.sample_rate,
            duration=result.duration,
            emotion=emotion_embedding.state.value,
            emotion_intensity=emotion_embedding.intensity,
            lipsync_data=lipsync_data,
            synthesis_time=time.time() - start_time,
            quality_score=result.naturalness_mos,
            metadata={"tier": self.tier.value, "engine": result.engine}
        )

        return payload.to_dict()
