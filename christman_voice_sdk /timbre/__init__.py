from integration.voice_capture_client import VoiceCaptureClient

def __init__(self, ...):
    # existing code...
    self.voice_capture = VoiceCaptureClient()
    self.captured_voice_profile = None

def _extract_x_vector(self, segments: List[AudioSegment]) -> np.ndarray:
    """Extract X-vector with voice capture profile integration."""
    
    # Load captured voice profile if available
    if self.captured_voice_profile is None:
        self.captured_voice_profile = self.voice_capture.load_latest_profile()
    
    # Rest of x-vector extraction code...


"""
Christman Voice SDK — timbre module
© 2026 Everett Nathaniel Christman & Misty Gail Christman
The Christman AI Project — Luma Cognify AI
Patent Pending TCAP-2026-001 / TCAP-2026-002
"""
