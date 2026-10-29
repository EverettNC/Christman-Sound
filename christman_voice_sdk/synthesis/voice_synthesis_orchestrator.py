"""
Voice synthesis orchestrator — voicepack, timbre, emotion, lipsync.

Speaking goes through the package's own mouth (voice_synthesis, which renders
through christman_sound.synth) in the caller's voice. `synthesize` takes the
voice as a required argument: a voice profile file, a frame voice folder, or
a loaded VoiceProfile or FrameVoice. With none, SynthesisUnavailable says so;
there is no default voice and no fallback. No outside speech engine is
driven from here (Coqui XTTS, the only engine it once drove, was cut on
2026-10-17 and lives in quarantine/2026-10_outside_voices/).

The emotion reaches the mouth only as its named controls. A neutral emotion
(or none) adds nothing. Any other emotion is passed as `affect`, which the
frame voice refuses with the reason (it has no affect input; mapping an
emotion onto pitch or rate here would be an invented mapping) and a voice
profile refuses as a control it has no input for. The embedded emotion
travels on the exception as `emotion_params`.

Kept whole: the AudioProcessor / PhonemeLabeler / TimbreModeler /
EmotionEmbedder / VoicepackBuilder wiring, voicepack validation and loading,
the lipsync path and `SynthesisResultPayload`.
"""

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
from ..engines.synthesis_result import SynthesisUnavailable
# Re-exported: consumers import SynthesisResult from this module, as the
# pre-migration package did.
from ..engines.synthesis_result import SynthesisResult  # noqa: F401
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
    #: 2026-10-11: the engine's degraded flag and reason stopped here — a
    #: failed synthesis came back as a dict with audio=None and nothing
    #: naming the failure. Both now travel to the caller.
    degraded: bool = False
    error_reason: Optional[str] = None

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
            "quality_score_known": self.quality_score is not None,
            "metadata": self.metadata,
            "degraded": self.degraded,
            "error_reason": self.error_reason,
        }

class VoiceSynthesisOrchestrator:
    """
    Coordinates the complete voice synthesis pipeline.

    `_build_custom_emotion_models` raises NotImplementedError: it once
    returned made-up emotion models, and none is made up here.
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

        #: The renderer: the package's own mouth (voice_synthesis), made on first use.
        self.engine: Optional[Any] = None
        self.reference_audio: List[Path] = []
        self.current_voicepack: Optional[Dict[str, Any]] = None
        self.current_voice_profile: Optional[VoiceProfile] = None

        if auto_load_engine:
            self._ensure_engine()

        logger.info("VoiceSynthesisOrchestrator initialized", extra={"tier": tier.value})

    def _ensure_engine(self) -> Any:
        """The package's own mouth. Raises SynthesisUnavailable with the reason when it cannot load."""
        if self.engine is None:
            try:
                from .voice_synthesis import SpeechSynthesisEngine
            except Exception as exc:
                raise SynthesisUnavailable(f"voice_synthesis not importable: {type(exc).__name__}: {exc}",
                                           where="VoiceSynthesisOrchestrator") from exc
            self.engine = SpeechSynthesisEngine()
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

        self.reference_audio = list(self.current_voicepack.get("reference_audio") or [])
        if not self.reference_audio:
            logger.warning("Voicepack contains no reference audio.")

        logger.info("Voicepack loaded.")

    def synthesize(
        self,
        text: str,
        emotion: Optional[str] = None,
        emotion_intensity: float = 1.0,
        tonescore: Optional[float] = None,
        generate_lipsync: bool = False,
        *,
        voice: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Speak `text` in the caller's `voice` (required). `kwargs` are the
        mouth's named controls (voice_synthesis.CONTROL_KEYWORDS), applied or
        refused with the reason. Raises SynthesisUnavailable when there is no
        voice or it cannot be read.
        """
        if voice is None:
            err = SynthesisUnavailable(
                "no voice given: pass the caller's voice (voice=<voice profile file or frame voice folder>, or a "
                "loaded VoiceProfile or FrameVoice); there is no default voice",
                where="VoiceSynthesisOrchestrator.synthesize")
            logger.error("Orchestrator did not speak %r: %s", (text or "")[:40], err)
            raise err
        if self.current_voicepack is None:
            raise ValueError("No voicepack loaded. Call load_voicepack() first.")
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        start_time = time.time()

        # Simplified for brevity; logic routes through EmotionEmbedder
        emotion_embedding = self.emotion_embedder.embed_emotion(emotion or "neutral", emotion_intensity)

        controls = dict(kwargs)
        if emotion_embedding.state.value != "neutral":
            controls.setdefault("affect", emotion_embedding.state.value)
        try:
            engine = self._ensure_engine()
            result = engine._render(text, None, "en", voice=voice, controls=controls)
        except Exception as err:
            err.emotion_params = emotion_embedding.to_dict()  # type: ignore[attr-defined]
            logger.error("Orchestrator did not speak %r: %s", text[:40], err)
            raise

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
            metadata={"tier": self.tier.value, "engine": result.engine},
            degraded=bool(result.degraded) or result.audio is None,
            error_reason=result.error_reason,
        )

        return payload.to_dict()
