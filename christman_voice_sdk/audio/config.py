"""
Configuration for Christman Voice SDK (ICanHearYou voice cloning system).

Tier structure kept for compatibility with existing callers.
EVERY feature is turned ON for EVERYONE. No paywalls. No restrictions.

NO OUTSIDE VOICES
-----------------
Every outside speech engine is cut. `synthesis_engines` is an empty list:
the config names no outside synthesis model and claims no engine it does not
have. The wav2vec2 model setting stays; that is the ear side, not a voice.

NO FOLDERS INSIDE THE PACKAGE
-----------------------------
voice_capture_client.py and christman_ocr_shared.py build a Config at import,
so a Config that made folders would write into site-packages (or the clone).
The three folders live under the SDK's data folder,
`$CHRISTMAN_AI_HOME/voice_sdk` (default `~/.christman_ai/voice_sdk`, beside the
logs, voice profiles and voice cache the SDK already keeps there). Building a
Config creates nothing. A caller that needs one calls `config.ensure_dir(...)`,
which makes it then, owner-only (0700).

The default config also named a Hugging Face model id
("jonatasgrosman/wav2vec2-large-xlsr-53-english"). Nothing read it, and a model
id is what from_pretrained downloads. Everett's rule is offline, no downloads,
so it is gone; a model is loaded from a local directory its caller gives.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
import yaml


def sdk_data_root() -> Path:
    """The SDK's data folder: $CHRISTMAN_AI_HOME, else ~/.christman_ai."""
    raw = os.environ.get("CHRISTMAN_AI_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".christman_ai"


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
    # Ruling AC (2026-10-29 review): the leftover avatar_integration = True
    # flag is removed. No avatar work is done in this package (see
    # face_vector.py and face_reading.py), so a flag saying it was on was
    # not true.
    batch_processing: bool = True

    # Performance
    max_concurrent_requests: int | float = float("inf")
    priority_queue: str = "ultra"


# All tiers get full access
TIER_FEATURES: Dict[Tier, TierFeatures] = {
    tier: TierFeatures(
        synthesis_engines=[]
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

        # Paths only. Nothing is created until a caller asks (ensure_dir).
        self.data_root = sdk_data_root() / "voice_sdk"
        self.models_dir = self.data_root / "models"
        self.data_dir = self.data_root / "data"
        self.logs_dir = self.data_root / "logs"

    def ensure_dir(self, directory: Path) -> Path:
        """Create `directory` now (owner-only), because a caller needs it."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        return directory

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


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


def require_local_or_tls(uri: str, what: str = "endpoint") -> str:
    """
    Accept a WebSocket URI only if it cannot carry a client's voice off the
    machine in the clear: `ws://` to a loopback host, or `wss://` anywhere.

    Rule 12. Voice biometrics and OCR'd text go over these sockets. The
    previous defaults were plain `ws://` with a config-overridable host, so
    one edited YAML line sent them in plaintext to any address.

    Raises:
        ValueError: on any other scheme, or on plaintext to a non-loopback host.
    """
    from urllib.parse import urlsplit

    parts = urlsplit(uri)
    scheme = (parts.scheme or "").lower()
    host = (parts.hostname or "").lower()
    if scheme == "wss":
        return uri
    if scheme == "ws" and host in _LOOPBACK_HOSTS:
        return uri
    raise ValueError(
        f"{what} {uri!r} is not allowed: plaintext ws:// is permitted only to "
        f"loopback ({sorted(_LOOPBACK_HOSTS)}); anything else must be wss://."
    )
