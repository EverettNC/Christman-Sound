"""
Voicepack File Format and Export System

WHAT CHANGED AND WHY
--------------------
1. FAKE ENCRYPTION REMOVED (Rule 12).
   The predecessor used `shutil.copy` to append `.encrypted` and logged 
   'Voicepack encrypted' without encrypting anything. Real symmetric encryption 
   via `cryptography.fernet.Fernet` is now wired when requested.

2. ELIMINATED PICKLE SERIALIZATION.
   Voicepack internal archives now store profiles as secure JSON metadata 
   and `.npz` array payloads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from timbre.timbre_modeler import TimbreModeler, VoiceProfile
from .logger import get_logger

logger = get_logger(__name__)

try:
    from cryptography.fernet import Fernet
    _crypto_ok = True
except ImportError:
    _crypto_ok = False


@dataclass
class VoicepackMetadata:
    name: str
    version: str = "1.0.0"
    created: str = ""
    tier: str = "basic"
    gender: Optional[str] = None
    age_range: Optional[str] = None
    accent: Optional[str] = None
    training_hours: float = 0.0
    sample_count: int = 0
    quality_score: float = 0.0
    emotions: Optional[List[str]] = None
    checksum: str = ""
    encrypted: bool = False

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now(timezone.utc).isoformat()
        if self.emotions is None:
            self.emotions = ["neutral"]


class VoicepackBuilder:
    """Builds, validates, and unpacks .voicepack archives securely."""

    def __init__(self, output_dir: Path = Path("data/voicepacks")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("VoicepackBuilder initialized: %s", output_dir)

    def build(
        self,
        name: str,
        voice_profile: VoiceProfile,
        reference_audio: List[Path],
        metadata: VoicepackMetadata,
        emotion_models: Optional[Dict[str, Path]] = None,
        compress: bool = True,
        encrypt: bool = False,
        encryption_key: Optional[bytes] = None,
        extras: Optional[Dict[str, Any]] = None
    ) -> Path:
        logger.info("Building voicepack: %s", name)

        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / "voicepack_build"
            build_dir.mkdir()

            metadata_path = build_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(asdict(metadata), f, indent=2)

            modeler = TimbreModeler()
            profile_base = build_dir / "voice_profile"
            modeler.save_profile(voice_profile, profile_base)

            ref_dir = build_dir / "reference_audio"
            ref_dir.mkdir()

            for i, audio_file in enumerate(reference_audio):
                if audio_file.exists():
                    dest = ref_dir / f"sample_{i:03d}{audio_file.suffix}"
                    shutil.copy(audio_file, dest)

            if emotion_models:
                emotion_dir = build_dir / "emotion_models"
                emotion_dir.mkdir()
                for model_name, model_path in emotion_models.items():
                    if model_path.exists():
                        dest = emotion_dir / model_path.name
                        shutil.copy(model_path, dest)

            validation = self._generate_validation(voice_profile, reference_audio, metadata)
            validation_path = build_dir / "validation.json"
            with open(validation_path, "w", encoding="utf-8") as f:
                json.dump(validation, f, indent=2)

            output_path = self.output_dir / f"{name}.voicepack"

            if compress:
                self._create_zip(build_dir, output_path)
            else:
                shutil.copytree(build_dir, output_path.with_suffix(".voicepack_dir"))

            if encrypt:
                output_path = self._encrypt_voicepack(output_path, encryption_key)

        logger.info("Voicepack created successfully: %s", output_path)
        return output_path

    def _create_zip(self, source_dir: Path, output_path: Path):
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir)
                    zipf.write(file_path, arcname)

    def _generate_validation(
        self,
        voice_profile: VoiceProfile,
        reference_audio: List[Path],
        metadata: VoicepackMetadata
    ) -> Dict[str, Any]:
        profile_dict = voice_profile.to_dict()
        profile_str = json.dumps(profile_dict, sort_keys=True)
        profile_hash = hashlib.sha256(profile_str.encode()).hexdigest()

        audio_hashes = []
        for audio_file in reference_audio:
            if audio_file.exists():
                with open(audio_file, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    audio_hashes.append({"file": audio_file.name, "sha256": file_hash})

        return {
            "version": "1.0",
            "validated": datetime.now(timezone.utc).isoformat(),
            "profile_checksum": profile_hash,
            "audio_checksums": audio_hashes,
            "tier": metadata.tier,
            "emotions": metadata.emotions,
        }

    def _encrypt_voicepack(self, voicepack_path: Path, encryption_key: Optional[bytes]) -> Path:
        if not _crypto_ok:
            raise RuntimeError("cryptography library not installed. Cannot encrypt voicepack.")
        if not encryption_key:
            raise ValueError("Encryption requested but no encryption_key provided.")

        fernet = Fernet(encryption_key)
        with open(voicepack_path, "rb") as f:
            encrypted_data = fernet.encrypt(f.read())

        encrypted_path = voicepack_path.with_suffix(".voicepack.enc")
        with open(encrypted_path, "wb") as f:
            f.write(encrypted_data)

        voicepack_path.unlink()
        logger.info("Voicepack encrypted securely: %s", encrypted_path)
        return encrypted_path

    def load(self, voicepack_path: Path, decryption_key: Optional[bytes] = None) -> Dict[str, Any]:
        if not voicepack_path.exists():
            raise FileNotFoundError(f"Voicepack not found: {voicepack_path}")

        logger.info("Loading voicepack: %s", voicepack_path.name)

        cache_dir = self.output_dir.parent / "cache" / voicepack_path.stem
        cache_dir.mkdir(parents=True, exist_ok=True)

        archive_to_open = voicepack_path

        if voicepack_path.suffix == ".enc":
            if not _crypto_ok:
                raise RuntimeError("cryptography not installed. Cannot decrypt voicepack.")
            if not decryption_key:
                raise ValueError("Voicepack is encrypted but no decryption_key was provided.")
            fernet = Fernet(decryption_key)
            with open(voicepack_path, "rb") as f:
                decrypted_bytes = fernet.decrypt(f.read())
            
            temp_zip = cache_dir / "temp_decrypted.zip"
            with open(temp_zip, "wb") as f:
                f.write(decrypted_bytes)
            archive_to_open = temp_zip

        with zipfile.ZipFile(archive_to_open, "r") as zipf:
            zipf.extractall(cache_dir)

        metadata_path = cache_dir / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        modeler = TimbreModeler()
        profile_base = cache_dir / "voice_profile"
        voice_profile = modeler.load_profile(profile_base)

        ref_dir = cache_dir / "reference_audio"
        reference_audio = list(ref_dir.glob("*.wav")) if ref_dir.exists() else []

        validation_path = cache_dir / "validation.json"
        with open(validation_path, "r", encoding="utf-8") as f:
            validation = json.load(f)

        return {
            "metadata": metadata,
            "voice_profile": voice_profile,
            "reference_audio": reference_audio,
            "validation": validation,
        }

    def validate(self, voicepack_path: Path) -> bool:
        try:
            data = self.load(voicepack_path)
            return bool(data.get("metadata") and data.get("voice_profile") and data.get("validation"))
        except Exception as e:
            logger.error("Validation failed: %s", e)
            return False
