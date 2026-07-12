import json
from unittest.mock import patch, MagicMock
from teamwork_s2s.src.core.event_bus import EventBus
from teamwork_s2s.src.core.mqtt_handler import MQTTHandler

def test_mqtt_event_routing():
    eb = EventBus()
    # Mock PAHO_AVAILABLE as False to test simulation mode
    with patch("teamwork_s2s.src.core.mqtt_handler.PAHO_AVAILABLE", False):
        handler = MQTTHandler(event_bus=eb)
        
        eb.publish("TOOL_CALL_MOTOR", {"motor": "pan", "value": 90})
        eb.publish("EMERGENCY_HALT", {"stop": True})
        
        # Check internal storage tracker
        assert len(handler.published_messages) == 2
        
        # standard motor call -> robot/action/standard, QoS 1
        m1 = handler.published_messages[0]
        assert m1["topic"] == "robot/action/standard"
        assert m1["qos"] == 1
        assert m1["payload"] == {"motor": "pan", "value": 90}
        
        # emergency halt -> robot/action/override, QoS 2
        m2 = handler.published_messages[1]
        assert m2["topic"] == "robot/action/override"
        assert m2["qos"] == 2
        assert m2["payload"] == {"stop": True}
        
        handler.disconnect()

def test_mqtt_client_publish():
    eb = EventBus()
    mock_client = MagicMock()
    # Mock PAHO_AVAILABLE as True and Client constructor to return mock_client
    with patch("paho.mqtt.client.Client", return_value=mock_client), \
         patch("teamwork_s2s.src.core.mqtt_handler.PAHO_AVAILABLE", True):
         
        handler = MQTTHandler(event_bus=eb)
        handler.client = mock_client
        
        handler.process_event("EMERGENCY_HALT", {"halt": "critical"})
        
        # Assert client publish is executed with QoS 2
        mock_client.publish.assert_called_once_with(
            "robot/action/override",
            json.dumps({"halt": "critical"}),
            qos=2
        )
        
        handler.disconnect()

def test_mqtt_run_unit_test():
    eb = EventBus()
    with patch("teamwork_s2s.src.core.mqtt_handler.PAHO_AVAILABLE", False):
        handler = MQTTHandler(event_bus=eb)
        res = handler.run_unit_test()
        assert res is True
        assert len(handler.published_messages) >= 2
        handler.disconnect()
