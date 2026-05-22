"""
Crypto bridge for HNDL integration into voice synthesis.
Wires ML-KEM-768 post-quantum encryption into voicepack protection.

Part of the Christman AI Project — Luma Cognify AI
"""

from christman_crypto import HybridSigner, encrypt_payload, decrypt_payload
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VoicepackCryptoEngine:
"""Encrypts and signs voicepacks using HNDL stack."""

def __init__(self):
self.signer = HybridSigner()
logger.info("VoicepackCryptoEngine initialized with ML-KEM-768")

def encrypt_voicepack(self, voicepack_data: bytes, voicepack_id: str) -> bytes:
"""Encrypt voicepack with full HNDL stack."""
encrypted = encrypt_payload(voicepack_data)
logger.info(f"Voicepack {voicepack_id} encrypted with ML-KEM-768")
return encrypted

def sign_voicepack(self, encrypted_data: bytes, voicepack_id: str) -> bytes:
"""Sign encrypted voicepack."""
signature = self.signer.sign(encrypted_data)
logger.info(f"Voicepack {voicepack_id} signed and sealed")
return signature

def decrypt_voicepack(self, encrypted_data: bytes, signature: bytes) -> bytes:
"""Decrypt and verify voicepack integrity."""
verified = self.signer.verify(encrypted_data, signature)
if not verified:
raise ValueError("Voicepack signature verification failed")
decrypted = decrypt_payload(encrypted_data)
logger.info("Voicepack decrypted and verified")
return decrypted

__all__ = ["VoicepackCryptoEngine"]
