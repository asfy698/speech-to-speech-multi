import numpy as np
from typing import Optional

class VADHandler:
    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold

    def process(self, audio_chunk: bytes = b"") -> bool:
        if not audio_chunk:
            return False
        try:
            # Ensure length is multiple of 2 for 16-bit PCM
            if len(audio_chunk) % 2 != 0:
                audio_chunk = audio_chunk[:len(audio_chunk) - (len(audio_chunk) % 2)]
            samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
            if len(samples) == 0:
                return False
            # Normalize to [-1.0, 1.0] range
            samples = samples / 32768.0
            rms = float(np.sqrt(np.mean(samples**2)))
            return rms > self.threshold
        except Exception:
            # Fallback simple energy calculation if numpy or formatting fails
            if len(audio_chunk) == 0:
                return False
            val = sum(abs(b - 128) for b in audio_chunk) / len(audio_chunk)
            norm_val = val / 128.0
            return norm_val > self.threshold

    def run_unit_test(self) -> bool:
        active_chunk = b'\x7f\xff' * 100
        silent_chunk = b'\x00\x00' * 100
        return self.process(active_chunk) is True and self.process(silent_chunk) is False
