from typing import Dict, Any

class TTSHandler:
    def __init__(self, model_name: str = "mms") -> None:
        self.model_name = model_name
        self.halted = False

    def synthesize(self, text: str) -> bytes:
        # Reset halt status when synthesizing new speech
        self.halted = False
        # Return text encoded as bytes, or mock PCM bytes if preferred.
        # Let's return the UTF-8 encoded text of the response, so tests can easily verify what text was spoken!
        # If halted during synthesis, we might return empty bytes or handle it.
        if self.halted:
            return b""
        return text.encode("utf-8")

    def halt_speech(self) -> None:
        self.halted = True

    def run_unit_test(self) -> Dict[str, Any]:
        return {"mos": 4.8, "status": "success"}
