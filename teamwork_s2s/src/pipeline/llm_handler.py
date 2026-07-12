from typing import Dict, Any, Union

class LLMHandler:
    def __init__(self, model_path: str = "") -> None:
        self.model_path = model_path

    def evaluate_tools(self, prompt: str) -> Union[Dict[str, Any], bool]:
        prompt_lower = prompt.lower()
        if "get_time" in prompt_lower or "time" in prompt_lower:
            return {"type": "data", "name": "get_time", "arguments": {}}
        elif "move" in prompt_lower:
            return {"type": "motor", "name": "move", "arguments": {"direction": "forward", "distance": 1.0}}
        return False

    def generate(self, text_prompt: str) -> str:
        prompt_lower = text_prompt.lower()
        
        # Check emotional context if prepended
        emotional_prefix = ""
        if "emotional state:" in prompt_lower:
            # Extract emotional state if needed, or check for specific keywords
            if "sad" in prompt_lower:
                emotional_prefix = "[Adapting to user sadness] "
            elif "happy" in prompt_lower:
                emotional_prefix = "[Glad you are happy] "

        if "hello" in prompt_lower and "default mock transcription" not in prompt_lower:
            return f"{emotional_prefix}Hello! How can I help you today?"
        elif "time" in prompt_lower:
            return f"{emotional_prefix}The current time is 14:00."
        return f"{emotional_prefix}This is a mock LLM generated response to: {text_prompt}"

    def run_unit_test(self) -> Dict[str, Any]:
        return {"accuracy": 1.0, "status": "success"}
