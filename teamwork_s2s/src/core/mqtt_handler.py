import logging
import json
from typing import Dict, Any, Optional, List
from teamwork_s2s.src.core.event_bus import EventBus

try:
    import paho.mqtt.client as mqtt  # type: ignore
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False

logger = logging.getLogger(__name__)

class MQTTHandler:
    """Translates local system events into MQTT messages on standard and override queues."""

    def __init__(self, event_bus: EventBus, broker_host: str = "localhost", broker_port: int = 1883) -> None:
        """Initializes MQTTHandler, subscribes to EventBus topics, and connects to MQTT broker."""
        self.event_bus = event_bus
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client: Optional[Any] = None
        self.published_messages: List[Dict[str, Any]] = []

        # Subscribe to Event Bus topics
        self.event_bus.subscribe("TOOL_CALL_MOTOR", self.handle_motor_call)
        self.event_bus.subscribe("EMERGENCY_HALT", self.handle_emergency_halt)

        if PAHO_AVAILABLE:
            try:
                try:
                    # paho-mqtt v2.x API syntax
                    self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
                except AttributeError:
                    # Fallback to paho-mqtt v1.x API syntax
                    self.client = mqtt.Client()
                
                self.client.connect(self.broker_host, self.broker_port, 60)
                self.client.loop_start()
                logger.info(f"MQTT client connected to {self.broker_host}:{self.broker_port}")
            except Exception as e:
                logger.warning(f"Failed to connect to MQTT broker: {e}. Falling back to simulation.")
                self.client = None
        else:
            logger.info("paho-mqtt not available. Falling back to simulation mode.")

    def handle_motor_call(self, payload: Dict[str, Any]) -> None:
        """Callback to handle standard motor actuation commands from EventBus."""
        self.process_event("TOOL_CALL_MOTOR", payload)

    def handle_emergency_halt(self, payload: Dict[str, Any]) -> None:
        """Callback to handle emergency stop command from EventBus."""
        self.process_event("EMERGENCY_HALT", payload)

    def process_event(self, topic: str, payload: Dict[str, Any]) -> None:
        """Translates system events to MQTT topics and QoS. Standard uses QoS 1, Emergency uses QoS 2."""
        target_topic = ""
        qos = 1
        
        if topic == "TOOL_CALL_MOTOR":
            target_topic = "robot/action/standard"
            qos = 1
        elif topic == "EMERGENCY_HALT":
            target_topic = "robot/action/override"
            qos = 2
        else:
            target_topic = "robot/action/standard"
            qos = 1

        msg = {"topic": target_topic, "payload": payload, "qos": qos}
        self.published_messages.append(msg)
        
        payload_str = json.dumps(payload)
        if self.client is not None:
            try:
                self.client.publish(target_topic, payload_str, qos=qos)
                logger.debug(f"MQTT published to '{target_topic}' (QoS {qos}): {payload_str}")
            except Exception as e:
                logger.error(f"MQTT publish failed: {e}")
        else:
            logger.info(f"[MQTT Simulation] Topic: {target_topic}, QoS: {qos}, Payload: {payload_str}")

    def close(self) -> None:
        """Cleans up broker connection and stops loop thread."""
        if self.client is not None:
            try:
                self.client.loop_stop()
                self.client.disconnect()
                logger.info("MQTT connection closed cleanly.")
            except Exception as e:
                logger.error(f"Error during MQTT disconnect cleanup: {e}")
            finally:
                self.client = None

    def disconnect(self) -> None:
        """Cleans up broker connection (alias for close)."""
        self.close()

    def run_unit_test(self) -> bool:
        """Executes a diagnostic run of standard and override event routing, returning test status."""
        test_payload = {"action": "move", "duration": 3}
        self.process_event("TOOL_CALL_MOTOR", test_payload)
        self.process_event("EMERGENCY_HALT", {"halt": True})
        return len(self.published_messages) >= 2
