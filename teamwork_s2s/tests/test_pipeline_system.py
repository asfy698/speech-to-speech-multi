import sys
import os
import unittest
import sqlite3
from typing import Dict, Any, List

# Setup sys.path to resolve teamwork_s2s package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from teamwork_s2s.src.config.config_manager import ConfigManager
from teamwork_s2s.src.core.logger import Logger
from teamwork_s2s.src.core.event_bus import EventBus
from teamwork_s2s.src.pipeline.vad_handler import VADHandler
from teamwork_s2s.src.pipeline.stt_handler import STTHandler
from teamwork_s2s.src.pipeline.llm_handler import LLMHandler
from teamwork_s2s.src.pipeline.tts_handler import TTSHandler
from teamwork_s2s.src.pipeline.fer_handler import FERHandler
from teamwork_s2s.src.pipeline.orchestrator import PipelineOrchestrator

class TestPipelineSystem(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "test_metrics.db"
        # Ensure clean database before each test
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
        self.config = ConfigManager()
        self.logger = Logger(self.db_path)
        self.event_bus = EventBus()
        self.vad = VADHandler()
        self.stt = STTHandler()
        self.llm = LLMHandler()
        self.tts = TTSHandler()
        self.fer = FERHandler()
        
        self.orchestrator = PipelineOrchestrator(
            config_manager=self.config,
            logger=self.logger,
            event_bus=self.event_bus,
            vad_handler=self.vad,
            stt_handler=self.stt,
            llm_handler=self.llm,
            tts_handler=self.tts,
            fer_handler=self.fer
        )

    def tearDown(self) -> None:
        # Clean up database after each test
        if os.path.exists(self.db_path):
            # Attempt to delete, ignore if locked
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_handler_units(self) -> None:
        """Unit tests for individual handler components."""
        # VAD
        self.assertTrue(self.vad.run_unit_test())
        # STT
        stt_test = self.stt.run_unit_test()
        self.assertEqual(stt_test["status"], "success")
        # LLM
        llm_test = self.llm.run_unit_test()
        self.assertEqual(llm_test["status"], "success")
        # TTS
        tts_test = self.tts.run_unit_test()
        self.assertEqual(tts_test["status"], "success")
        # FER
        self.assertTrue(self.fer.run_unit_test())

    def test_pure_conversation_flow(self) -> None:
        """Pure Conversation: VAD silence boundary triggers STT, LLM generation, and TTS synthesis."""
        received_events: List[Dict[str, Any]] = []
        
        def on_speech_response(payload: Dict[str, Any]) -> None:
            received_events.append(payload)
            
        self.event_bus.subscribe("speech_response", on_speech_response)
        
        # 1. Feed active chunk (energy high)
        active_chunk = b'\x7f\xff' * 100
        self.orchestrator.process_audio_chunk(active_chunk)
        self.assertTrue(len(self.orchestrator.audio_buffer) > 0)
        self.assertEqual(len(received_events), 0)
        
        # 2. Feed silent chunk (energy low) to trigger boundary
        silent_chunk = b'\x00\x00' * 100
        self.orchestrator.process_audio_chunk(silent_chunk)
        
        # Buffer should be cleared and event published
        self.assertEqual(len(self.orchestrator.audio_buffer), 0)
        self.assertEqual(len(received_events), 1)
        self.assertIn("Hello, this is a default mock transcription.", received_events[0]["text"])
        self.assertIsNotNone(received_events[0]["audio"])

    def test_data_retrieval_tool_call_flow(self) -> None:
        """Data Retrieval Tool Call: query contains 'get_time', executes tool, injects result, and LLM generates response."""
        received_events: List[Dict[str, Any]] = []
        
        def on_speech_response(payload: Dict[str, Any]) -> None:
            received_events.append(payload)
            
        self.event_bus.subscribe("speech_response", on_speech_response)
        
        # Feed "get_time" query as bytes to trigger VAD and decode to "get_time"
        self.orchestrator.process_audio_chunk(b"get_time")
        self.orchestrator.process_audio_chunk(b"\x00" * 100) # silence to trigger processing
        
        self.assertEqual(len(received_events), 1)
        response_text = received_events[0]["text"]
        # Response should contain the injected tool result ("14:00")
        self.assertIn("14:00", response_text)

    def test_motor_tool_actuation_flow(self) -> None:
        """Motor Tool Actuation: query contains 'move', intercepts tool call, halts speech, publishes standard motor command, and bypasses text generation."""
        motor_events: List[Dict[str, Any]] = []
        speech_events: List[Dict[str, Any]] = []
        
        def on_motor_call(payload: Dict[str, Any]) -> None:
            motor_events.append(payload)
            
        def on_speech_response(payload: Dict[str, Any]) -> None:
            speech_events.append(payload)
            
        self.event_bus.subscribe("TOOL_CALL_MOTOR", on_motor_call)
        self.event_bus.subscribe("speech_response", on_speech_response)
        
        # Feed "move" query to trigger motor tool call
        self.orchestrator.process_audio_chunk(b"move forward")
        self.orchestrator.process_audio_chunk(b"\x00" * 100) # silence
        
        # Verify speech is halted, TOOL_CALL_MOTOR is published, and speech_response is bypassed
        self.assertTrue(self.tts.halted)
        self.assertEqual(len(motor_events), 1)
        self.assertEqual(motor_events[0].get("direction"), "forward")
        self.assertEqual(len(speech_events), 0)

    def test_emergency_stop_override_flow(self) -> None:
        """Emergency STOP override: transcription is 'STOP', publishes EMERGENCY_HALT, halts speech, and bypasses LLM completely."""
        halt_events: List[Dict[str, Any]] = []
        speech_events: List[Dict[str, Any]] = []
        
        def on_emergency_halt(payload: Dict[str, Any]) -> None:
            halt_events.append(payload)
            
        def on_speech_response(payload: Dict[str, Any]) -> None:
            speech_events.append(payload)
            
        self.event_bus.subscribe("EMERGENCY_HALT", on_emergency_halt)
        self.event_bus.subscribe("speech_response", on_speech_response)
        
        # Feed "STOP" query
        self.orchestrator.process_audio_chunk(b"STOP")
        self.orchestrator.process_audio_chunk(b"\x00" * 100) # silence
        
        # Verify emergency halt published, speech halted, and speech generation bypassed
        self.assertTrue(self.tts.halted)
        self.assertEqual(len(halt_events), 1)
        self.assertTrue(halt_events[0].get("halt"))
        self.assertEqual(len(speech_events), 0)

    def test_emotion_adaptation_flow(self) -> None:
        """Emotion Adaptation: detects 'sad' emotion, logs to SQLite, publishes EVENT_USER_DISTRESSED, and adapts subsequent LLM prompts."""
        distress_events: List[Dict[str, Any]] = []
        speech_events: List[Dict[str, Any]] = []
        
        def on_distress(payload: Dict[str, Any]) -> None:
            distress_events.append(payload)
            
        def on_speech_response(payload: Dict[str, Any]) -> None:
            speech_events.append(payload)
            
        self.event_bus.subscribe("EVENT_USER_DISTRESSED", on_distress)
        self.event_bus.subscribe("speech_response", on_speech_response)
        
        # Store sad image
        self.orchestrator.store_image_buffer(b"sad_image")
        
        # 1. Verify distressed event was published
        self.assertEqual(len(distress_events), 1)
        self.assertEqual(distress_events[0].get("emotion"), "sad")
        
        # 2. Verify emotion logged in SQLite metrics.db
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT emotion, confidence FROM fer_logs")
        rows = cursor.fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "sad")
        self.assertAlmostEqual(rows[0][1], 0.95)
        
        # 3. Verify emotion is prepended to subsequent LLM generation
        self.orchestrator.process_audio_chunk(b"hello")
        self.orchestrator.process_audio_chunk(b"\x00" * 100) # silence
        
        self.assertEqual(len(speech_events), 1)
        response_text = speech_events[0]["text"]
        self.assertIn("[Adapting to user sadness]", response_text)

    def test_diagnostic_run(self) -> None:
        """Diagnostic run: triggers all component unit tests and logs results."""
        results = self.orchestrator.execute_diagnostic_run()
        self.assertEqual(results.get("vad"), True)
        self.assertEqual(results.get("stt", {}).get("status"), "success")
        self.assertEqual(results.get("llm", {}).get("status"), "success")
        self.assertEqual(results.get("tts", {}).get("status"), "success")
        self.assertEqual(results.get("fer"), True)
        
        # Check that metrics were logged to SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT latency FROM kpi_benchmarks")
        rows = cursor.fetchall()
        conn.close()
        self.assertTrue(len(rows) >= 1)

    def test_hot_reload(self) -> None:
        """Hot reload: updates the VAD threshold from configuration."""
        self.config.update_config({"vad_threshold": 0.05})
        self.assertTrue(self.orchestrator.hot_reload())
        self.assertEqual(self.vad.threshold, 0.05)

if __name__ == "__main__":
    unittest.main()
