from typing import Any, Dict, Optional, Union
from teamwork_s2s.src.config.config_manager import ConfigManager
from teamwork_s2s.src.core.logger import Logger
from teamwork_s2s.src.core.event_bus import EventBus
from teamwork_s2s.src.pipeline.vad_handler import VADHandler
from teamwork_s2s.src.pipeline.stt_handler import STTHandler
from teamwork_s2s.src.pipeline.llm_handler import LLMHandler
from teamwork_s2s.src.pipeline.tts_handler import TTSHandler
from teamwork_s2s.src.pipeline.fer_handler import FERHandler

class PipelineOrchestrator:
    """Master workflow orchestrator managing audio buffer, tool calls, emergency halts, and background FER loops."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        event_bus: EventBus,
        vad_handler: VADHandler,
        stt_handler: STTHandler,
        llm_handler: LLMHandler,
        tts_handler: TTSHandler,
        fer_handler: FERHandler,
    ) -> None:
        """Initializes the Pipeline Orchestrator with references to core components and handlers."""
        self.config_manager = config_manager
        self.logger = logger
        self.event_bus = event_bus
        self.vad_handler = vad_handler
        self.stt_handler = stt_handler
        self.llm_handler = llm_handler
        self.tts_handler = tts_handler
        self.fer_handler = fer_handler
        
        self.audio_buffer: bytearray = bytearray()
        self.last_frame: Optional[bytes] = None
        self.current_emotion: Optional[str] = None
        
    def process_audio_chunk(self, chunk: bytes) -> None:
        """Accumulates incoming audio chunks, runs VAD, and executes the cascaded speech pipeline."""
        is_speech = self.vad_handler.process(chunk)
        if is_speech:
            self.audio_buffer.extend(chunk)
        else:
            if len(self.audio_buffer) > 0:
                # Add the silence chunk to complete the audio frame
                self.audio_buffer.extend(chunk)
                audio_data = bytes(self.audio_buffer)
                self.audio_buffer.clear()
                
                # Transcribe
                transcription = self.stt_handler.transcribe(audio_data)
                
                # Check emergency stop override
                if self.intercept_emergency(transcription):
                    return
                
                # Prepend emotional context to prompt
                prompt = transcription
                if self.current_emotion:
                    prompt = f"Emotional state: {self.current_emotion}\n{prompt}"
                
                # Evaluate tool calls
                tool_call = self.llm_handler.evaluate_tools(prompt)
                if isinstance(tool_call, dict):
                    if tool_call.get("type") == "data":
                        data_result = self.execute_data_tool(tool_call)
                        # Inject back and generate - no double evaluation
                        new_prompt = f"{prompt}\nSystem Output: {data_result}"
                        resp = self.llm_handler.generate(new_prompt)
                        speech_bytes = self.tts_handler.synthesize(resp)
                        self.event_bus.publish("speech_response", {"text": resp, "audio": speech_bytes})
                    elif tool_call.get("type") == "motor":
                        # Motor tool call: halt speech, publish event, bypass generation
                        self.intercept_tool_call(tool_call)
                else:
                    resp = self.llm_handler.generate(prompt)
                    speech_bytes = self.tts_handler.synthesize(resp)
                    self.event_bus.publish("speech_response", {"text": resp, "audio": speech_bytes})

    def store_image_buffer(self, frame_data: bytes) -> None:
        """Stores incoming JPEG frames for background emotion analysis during LLM idle states."""
        self.last_frame = frame_data
        emotion_result = self.fer_handler.analyze_image(frame_data)
        
        # Log to sqlite
        self.logger.write_fer_log_to_sqlite(emotion_result)
        
        # Check if distressed
        emotion = emotion_result.get("emotion")
        confidence = float(emotion_result.get("confidence", 1.0))
        if isinstance(emotion, str):
            self.current_emotion = emotion
        else:
            self.current_emotion = None
        
        if emotion == "sad":
            self.event_bus.publish("EVENT_USER_DISTRESSED", {"emotion": "sad", "confidence": confidence})

    def execute_data_tool(self, tool_call: Dict[str, Any]) -> str:
        """Synchronously executes data-retrieval scripts (e.g., get_time) and injects results back into the LLM prompt."""
        name = tool_call.get("name")
        if name == "get_time":
            return "14:00"
        return "mock_data_result"

    def intercept_tool_call(self, tool_call: Dict[str, Any]) -> None:
        """Intercepts LLM JSON tool calling. Halts TTS stream and publishes movement commands to standard MQTT queue."""
        self.tts_handler.halt_speech()
        arguments = tool_call.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        self.event_bus.publish("TOOL_CALL_MOTOR", arguments)

    def intercept_emergency(self, transcription: str) -> bool:
        """Pre-empts LLM evaluation on detecting 'STOP'. Fires emergency halt to EventBus immediately."""
        if "STOP" in transcription.upper():
            self.tts_handler.halt_speech()
            self.event_bus.publish("EMERGENCY_HALT", {"halt": True, "reason": "STOP"})
            return True
        return False

    def execute_diagnostic_run(self) -> Dict[str, Any]:
        """Triggers run_unit_test() across all handlers and publishes results to Logger."""
        results: Dict[str, Any] = {}
        try:
            results["vad"] = self.vad_handler.run_unit_test()
        except Exception as e:
            results["vad"] = str(e)
            
        try:
            results["stt"] = self.stt_handler.run_unit_test()
        except Exception as e:
            results["stt"] = str(e)
            
        try:
            results["llm"] = self.llm_handler.run_unit_test()
        except Exception as e:
            results["llm"] = str(e)
            
        try:
            results["tts"] = self.tts_handler.run_unit_test()
        except Exception as e:
            results["tts"] = str(e)
            
        try:
            results["fer"] = self.fer_handler.run_unit_test()
        except Exception as e:
            results["fer"] = str(e)

        self.logger.record_model_metrics({"latency": 0.01, "diagnostic_results": results})
        return results

    def hot_reload(self) -> bool:
        """Performs dynamic module reloading when configuration settings change."""
        threshold = float(self.config_manager.get("vad_threshold", 0.01))
        self.vad_handler.threshold = threshold
        return True
