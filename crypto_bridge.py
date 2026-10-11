"""
Crypto bridge — Christman-Sound voicepacks on Harvest Now, Decrypt Later.

Source (law): https://github.com/The-ChristmanAI-Project/Harvest-Now-Decrypt-Later.git
Package: christman-crypto  Apache 2.0

WHAT CHANGED AND WHY
--------------------
The old file did not parse (class body at column 0) and imported names that
did not exist on the stale AlphaVox copy of christman_crypto. It logged
"encrypted with ML-KEM-768" without running a cipher.

This module:
- Loads the official HNDL tree first (PYTHONPATH often points at AlphaVox).
- Encrypts with HybridPQCipher(768) — ML-KEM-768 + XChaCha20-Poly1305.
- Persists ek/dk under ~/.christman_ai/hndl so decrypt works across process.
- Signs with HybridSigner. If oqs is missing, signer.mode is classical RSA-PSS
  and that fact is returned — never called hybrid.

Prove: encrypt → sign → verify → decrypt roundtrip on real bytes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HNDL_CANDIDATES = (
    Path.home() / "Harvest-Now-Decrypt-Later",
    Path("/Users/EverettN/Harvest-Now-Decrypt-Later"),
)


def _bind_hndl() -> Path:
    for root in _HNDL_CANDIDATES:
        if (root / "christman_crypto" / "__init__.py").is_file():
            resolved = str(root.resolve())
            if resolved in sys.path:
                sys.path.remove(resolved)
            sys.path.insert(0, resolved)
            return root.resolve()
    raise ImportError(
        "Harvest-Now-Decrypt-Later not found. Clone "
        "https://github.com/The-ChristmanAI-Project/Harvest-Now-Decrypt-Later.git "
        "to ~/Harvest-Now-Decrypt-Later"
    )


_HNDL_ROOT = _bind_hndl()

# Evict a stale christman_crypto (AlphaVox copy has no HybridSigner).
for _k in [k for k in list(sys.modules) if k == "christman_crypto" or k.startswith("christman_crypto.")]:
    loc = getattr(sys.modules[_k], "__file__", "") or ""
    if "Harvest-Now-Decrypt-Later" not in loc.replace("\\", "/"):
        sys.modules.pop(_k, None)

from christman_crypto import HybridPQCipher, HybridSigner  # noqa: E402


class VoicepackCryptoEngine:
    """Voicepack seal via official HNDL HybridPQCipher + HybridSigner."""

    KEY_DIR = Path.home() / ".christman_ai" / "hndl"

    def __init__(self, key_dir: Optional[Path] = None) -> None:
        self.source = "https://github.com/The-ChristmanAI-Project/Harvest-Now-Decrypt-Later.git"
        self.hndl_root = _HNDL_ROOT
        self.key_dir = Path(key_dir) if key_dir else self.KEY_DIR
        self.key_dir.mkdir(parents=True, exist_ok=True)
        self.pq = HybridPQCipher(768)
        self.ek, self.dk = self._load_or_create_kem()
        self.signer = HybridSigner(use_pq=True)
        logger.info(
            "VoicepackCryptoEngine online · HNDL %s · signer %s · pq_sig %s",
            self.hndl_root,
            self.signer.mode,
            self.signer.pq_available,
        )

    def _kem_paths(self) -> tuple[Path, Path]:
        return self.key_dir / "voicepack.ek", self.key_dir / "voicepack.dk"

    def _load_or_create_kem(self) -> tuple[bytes, bytes]:
        ek_path, dk_path = self._kem_paths()
        if ek_path.is_file() and dk_path.is_file():
            return ek_path.read_bytes(), dk_path.read_bytes()
        ek, dk = self.pq.keygen()
        ek_path.write_bytes(ek)
        dk_path.write_bytes(dk)
        ek_path.chmod(0o600)
        dk_path.chmod(0o600)
        return ek, dk

    def encrypt_voicepack(self, voicepack_data: bytes, voicepack_id: str) -> bytes:
        if not isinstance(voicepack_data, (bytes, bytearray)):
            raise TypeError("voicepack_data must be bytes")
        bundle = self.pq.encrypt(self.ek, bytes(voicepack_data))
        logger.info(
            "Voicepack %s encrypted · HybridPQCipher ML-KEM-768 + XChaCha20 · %d bytes",
            voicepack_id,
            len(bundle),
        )
        return bundle

    def sign_voicepack(self, encrypted_data: bytes, voicepack_id: str) -> bytes:
        if not isinstance(encrypted_data, (bytes, bytearray)):
            raise TypeError("encrypted_data must be bytes")
        signature = self.signer.sign(bytes(encrypted_data))
        logger.info(
            "Voicepack %s signed · %s · %d bytes",
            voicepack_id,
            self.signer.mode,
            len(signature),
        )
        return signature

    def decrypt_voicepack(self, encrypted_data: bytes, signature: bytes) -> bytes:
        if not self.signer.verify(bytes(encrypted_data), bytes(signature)):
            raise ValueError("Voicepack signature verification failed")
        plain = self.pq.decrypt(self.dk, bytes(encrypted_data))
        logger.info("Voicepack decrypted and verified · HNDL HybridPQCipher")
        return plain


__all__ = ["VoicepackCryptoEngine"]
