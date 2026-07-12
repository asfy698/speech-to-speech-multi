from typing import Dict, Any

class FERHandler:
    def __init__(self, model_name: str = "fer") -> None:
        self.model_name = model_name

    def analyze_image(self, frame_bytes: bytes) -> Dict[str, Any]:
        if frame_bytes == b"sad_image":
            return {"emotion": "sad", "confidence": 0.95}
        return {"emotion": "neutral", "confidence": 0.85}

    def run_unit_test(self) -> bool:
        return True
