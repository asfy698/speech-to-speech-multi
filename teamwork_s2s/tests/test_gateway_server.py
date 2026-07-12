import pytest
import json
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from teamwork_s2s.src.api.gateway_server import GatewayServer

def test_rest_override_endpoint():
    mock_orch = MagicMock()
    mock_orch.event_bus = MagicMock()
    
    server = GatewayServer(orchestrator=mock_orch)
    client = TestClient(server.app)
    
    response = client.post("/api/override", json={"reason": "obstacle_detected"})
    assert response.status_code == 200
    assert response.json()["status"] == "halted"
    
    mock_orch.event_bus.publish.assert_called_with(
        "EMERGENCY_HALT", {"reason": "obstacle_detected"}
    )

def test_rest_config_endpoint():
    mock_orch = MagicMock()
    mock_orch.config_manager = MagicMock()
    
    server = GatewayServer(orchestrator=mock_orch)
    client = TestClient(server.app)
    
    response = client.post("/api/config", json={"vad_threshold": 0.05})
    assert response.status_code == 200
    mock_orch.config_manager.update_config.assert_called_with({"vad_threshold": 0.05})

def test_websocket_audio_endpoint():
    mock_orch = MagicMock()
    mock_orch.event_bus = MagicMock()
    
    server = GatewayServer(orchestrator=mock_orch)
    client = TestClient(server.app)
    
    with client.websocket_connect("/audio") as ws:
        pcm_chunk = b"\x00\x00" * 320  # 320 samples of PCM audio
        ws.send_bytes(pcm_chunk)
        
        # Give a small sleep to let background task run
        import time
        time.sleep(0.1)
        
        # Verify the WS endpoint did not block and successfully routed the payload
        mock_orch.event_bus.publish.assert_called_with(
            "EVENT_AUDIO_RECEIVED", {"data": pcm_chunk}
        )
    
    # Verify that unsubscribe was called after websocket closed
    mock_orch.event_bus.unsubscribe.assert_called()
    called_args = mock_orch.event_bus.unsubscribe.call_args[0]
    assert called_args[0] == "speech_response"
    assert callable(called_args[1])


@pytest.mark.asyncio
async def test_twilio_non_blocking_post():
    mock_orch = MagicMock()
    mock_orch.config_manager = MagicMock()
    mock_orch.config_manager.get.side_effect = lambda key, default="": {
        "twilio_account_sid": "AC123",
        "twilio_auth_token": "auth123",
        "twilio_from_number": "whatsapp:+14155238886",
        "alert_phone": "+15550001111",
    }.get(key, default)
    server = GatewayServer(orchestrator=mock_orch)
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MagicMock(status_code=201)
        
        payload = {"emotion": "sad", "confidence": 0.95}
        
        # Test that calling the twilio notification triggers async post in background
        res = server.push_notification_to_twilio(payload)
        assert res is True


@pytest.mark.asyncio
async def test_twilio_message_body_mentions_sad_emotion():
    mock_orch = MagicMock()
    mock_orch.config_manager = MagicMock()
    mock_orch.config_manager.get.side_effect = lambda key, default="": {
        "twilio_account_sid": "AC123",
        "twilio_auth_token": "auth123",
        "twilio_from_number": "whatsapp:+14155238886",
        "alert_phone": "+15550001111",
    }.get(key, default)
    server = GatewayServer(orchestrator=mock_orch)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MagicMock(status_code=201)
        payload = {"emotion": "sad", "confidence": 0.92}
        server.push_notification_to_twilio(payload)
        await asyncio.sleep(0.05)

        assert mock_post.called
        kwargs = mock_post.call_args.kwargs
        assert kwargs["data"]["To"] == "whatsapp:+15550001111"
        assert kwargs["data"]["From"] == "whatsapp:+14155238886"
        assert "sad" in kwargs["data"]["Body"].lower()
