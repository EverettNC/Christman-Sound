"""
Configuration for Christman Voice SDK (ICanHearYou voice cloning system).

Tier structure kept for compatibility with existing beings.
EVERY feature is turned ON for EVERYONE. No paywalls. No restrictions.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
import yaml


class Tier(Enum):
    """Capability levels. Kept for compatibility across beings."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ELITE = "elite"
    ULTRA = "ultra"


@dataclass(frozen=True)
class TierFeatures:
    """All features enabled at every level."""
    # Audio processing
    noise_reduction_quality: str = "studio"
    max_audio_duration_hours: float = float("inf")

    # Voice synthesis
    synthesis_engines: list[str] = None  # type: ignore
    emotional_range: int = 11
    prosody_control: bool = True
    cadence_fingerprinting: bool = True

    # Advanced features
    custom_emotion_model: bool = True
    realtime_synthesis: bool = True
    avatar_integration: bool = True
    batch_processing: bool = True

    # Performance
    max_concurrent_requests: int | float = float("inf")
    priority_queue: str = "ultra"


# All tiers get full access
TIER_FEATURES: Dict[Tier, TierFeatures] = {
    tier: TierFeatures(
        synthesis_engines=["gpt_sovits", "f5_tts", "style_tts2"]
    )
    for tier in Tier
}


class Config:
    """Main configuration manager."""

    def __init__(self, config_path: Optional[Path] = None):
        self.root_dir = Path(__file__).parent.parent

        self.config_path = config_path or self.root_dir / "config" / "default_config.yaml"

        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config: Dict[str, Any] = yaml.safe_load(f) or {}
        else:
            self._config = self._get_default_config()

        self.models_dir = self.root_dir / "models"
        self.data_dir = self.root_dir / "data"
        self.logs_dir = self.root_dir / "logs"

        for directory in (self.models_dir, self.data_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "system": {
                "device": "auto",
                "log_level": "INFO",
                "num_workers": 4,
            },
            "audio": {
                "sample_rate": 16000,
                "target_db": -20.0,
                "silence_threshold_db": -40.0,
                "segment_length_seconds": 10.0,
                "overlap_seconds": 2.0,
            },
            "models": {
                "wav2vec2_model": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
                "gpt_sovits_checkpoint": "models/gpt_sovits_v3.pth",
                "f5_tts_checkpoint": "models/f5_tts.pth",
                "style_tts2_checkpoint": "models/style_tts2.pth",
            },
            "synthesis": {
                "max_length": 1000,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.9,
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get_tier_features(self, tier: Tier) -> TierFeatures:
        """Returns full features for any tier."""
        return TIER_FEATURES[tier]

    def save(self, path: Optional[Path] = None) -> None:
        save_path = path or self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)


def get_config(config_path: Optional[Path] = None) -> Config:
    return Config(config_path)
