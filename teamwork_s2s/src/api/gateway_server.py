import asyncio
import os
import logging
import threading
from typing import Any, Dict
from fastapi import FastAPI, WebSocket, Request
import requests

logger = logging.getLogger(__name__)

class GatewayServer:
    """FastAPI and WebSocket web server managing Pi audio/vision ingestion and external Twilio integration."""

    def __init__(self, orchestrator: Any) -> None:
        """Initializes GatewayServer with a reference to the PipelineOrchestrator."""
        self.orchestrator = orchestrator
        self.app = FastAPI()

        # Register REST routes
        self.app.post("/api/override")(self.override_endpoint)
        self.app.post("/api/config")(self.config_endpoint)
        self.app.post("/api/test")(self.test_endpoint)
        self.app.post("/move_forward")(self.move_forward)
        self.app.post("/move_backward")(self.move_backward)
        self.app.post("/stop")(self.stop_robot)

        # Register WebSocket routes
        self.app.websocket("/audio")(self.audio_endpoint)
        self.app.websocket("/vision")(self.vision_endpoint)

        # Subscribe to Event Bus for distress notifications
        if hasattr(self.orchestrator, "event_bus") and self.orchestrator.event_bus:
            self.orchestrator.event_bus.subscribe("EVENT_USER_DISTRESSED", self.push_notification_to_twilio)
            logger.info("GatewayServer: Subscribed to 'EVENT_USER_DISTRESSED'")

    def _esp32_base_url(self) -> str:
        host = os.getenv("ESP32_BASE_URL") or os.getenv("ESP32_IP") or "192.168.137.208"
        if host.startswith("http://") or host.startswith("https://"):
            return host.rstrip("/")
        return f"http://{host}".rstrip("/")

    def _call_esp32(self, endpoint: str) -> Dict[str, Any]:
        url = f"{self._esp32_base_url()}{endpoint}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return {
            "status_code": response.status_code,
            "text": response.text,
            "url": url,
        }

    async def audio_endpoint(self, websocket: Any) -> None:
        """Bidirectional WebSocket connection receiving PCM audio from Pi and streaming response back."""
        ws: WebSocket = websocket
        await ws.accept()
        logger.info("Audio WebSocket connected.")

        loop = asyncio.get_running_loop()

        # Send response back to websocket when speech_response is triggered
        async def send_response(payload: Dict[str, Any]):
            try:
                if "audio" in payload and payload["audio"]:
                    await ws.send_bytes(payload["audio"])
                if "text" in payload and payload["text"]:
                    await ws.send_json({"type": "text", "content": payload["text"]})
            except Exception as e:
                logger.error(f"Error sending audio/text response over WebSocket: {e}")

        def on_speech_response(payload: Dict[str, Any]):
            asyncio.run_coroutine_threadsafe(send_response(payload), loop)

        # Subscribe to speech response
        if hasattr(self.orchestrator, "event_bus") and self.orchestrator.event_bus:
            self.orchestrator.event_bus.subscribe("speech_response", on_speech_response)

        try:
            try:
                while True:
                    data = await ws.receive()
                    if "bytes" in data:
                        chunk = data["bytes"]
                        if hasattr(self.orchestrator, "process_audio_chunk") and self.orchestrator.process_audio_chunk:
                            # Run in thread pool to prevent blocking event loop with CPU-heavy processing
                            await loop.run_in_executor(None, self.orchestrator.process_audio_chunk, chunk)

                        # Also publish EVENT_AUDIO_RECEIVED for compatibility with verification tests
                        if hasattr(self.orchestrator, "event_bus") and self.orchestrator.event_bus:
                            self.orchestrator.event_bus.publish("EVENT_AUDIO_RECEIVED", {"data": chunk})
                    else:
                        break
            except Exception as e:
                logger.info(f"Audio WebSocket session ended: {e}")
        finally:
            if hasattr(self.orchestrator, "event_bus") and self.orchestrator.event_bus:
                if hasattr(self.orchestrator.event_bus, "unsubscribe"):
                    self.orchestrator.event_bus.unsubscribe("speech_response", on_speech_response)
                    logger.info("GatewayServer: Unsubscribed from 'speech_response'")


    async def vision_endpoint(self, websocket: Any) -> None:
        """WebSocket connection receiving continuous JPEG vision frames from Pi."""
        ws: WebSocket = websocket
        await ws.accept()
        logger.info("Vision WebSocket connected.")

        loop = asyncio.get_running_loop()
        try:
            while True:
                data = await ws.receive()
                if "bytes" in data:
                    chunk = data["bytes"]
                    if hasattr(self.orchestrator, "store_image_buffer") and self.orchestrator.store_image_buffer:
                        await loop.run_in_executor(None, self.orchestrator.store_image_buffer, chunk)
                else:
                    break
        except Exception as e:
            logger.info(f"Vision WebSocket session ended: {e}")

    async def override_endpoint(self, request: Any) -> Dict[str, Any]:
        """FastAPI REST POST endpoint /api/override to trigger instant software emergency stop."""
        payload = {}
        if isinstance(request, dict):
            payload = request
        elif hasattr(request, "json") and callable(request.json):
            try:
                payload = await request.json()
            except Exception:
                pass
        elif hasattr(request, "body") and callable(request.body):
            try:
                payload = await request.json()
            except Exception:
                pass
        else:
            payload = request

        if hasattr(self.orchestrator, "event_bus") and self.orchestrator.event_bus:
            self.orchestrator.event_bus.publish("EMERGENCY_HALT", payload or {"halt": True})
            return {"status": "halted", "message": "Emergency halt triggered."}
        
        return {"status": "error", "message": "Event bus not available."}

    async def config_endpoint(self, request: Any) -> Dict[str, Any]:
        """FastAPI REST POST endpoint /api/config to update runtime thresholds dynamically."""
        payload = {}
        if isinstance(request, dict):
            payload = request
        elif hasattr(request, "json") and callable(request.json):
            try:
                payload = await request.json()
            except Exception:
                pass
        elif hasattr(request, "body") and callable(request.body):
            try:
                payload = await request.json()
            except Exception:
                pass
        else:
            payload = request

        if hasattr(self.orchestrator, "config_manager") and self.orchestrator.config_manager:
            self.orchestrator.config_manager.update_config(payload)
            if hasattr(self.orchestrator, "hot_reload") and self.orchestrator.hot_reload:
                self.orchestrator.hot_reload()
            return {"status": "success", "config": self.orchestrator.config_manager.config}

        return {"status": "error", "message": "Config manager not available."}

    async def test_endpoint(self, request: Any) -> Dict[str, Any]:
        """FastAPI REST POST endpoint /api/test to trigger the diagnostics run and return JSON report."""
        if hasattr(self.orchestrator, "execute_diagnostic_run") and self.orchestrator.execute_diagnostic_run:
            report = self.orchestrator.execute_diagnostic_run()
            return {"status": "success", "report": report}
        elif hasattr(self.orchestrator, "logger") and self.orchestrator.logger:
            report = self.orchestrator.logger.generate_test_report()
            return {"status": "success", "report": report}
        
        return {"status": "success", "report": {"status": "executed"}}

    async def move_forward(self, request: Any) -> Dict[str, Any]:
        """FastAPI REST POST endpoint /move_forward to drive the ESP32 robot forward."""
        logger.info("MOVE FORWARD RECEIVED")
        try:
            result = await asyncio.to_thread(self._call_esp32, "/forward")
            logger.info("ESP STATUS: %s", result["status_code"])
            logger.info("ESP RESPONSE: %s", result["text"])
            return {"status": "forward", **result}
        except Exception as exc:
            logger.exception("Failed to move robot forward")
            return {"status": "error", "message": str(exc)}

    async def move_backward(self, request: Any) -> Dict[str, Any]:
        """FastAPI REST POST endpoint /move_backward to drive the ESP32 robot backward."""
        logger.info("MOVE BACKWARD RECEIVED")
        try:
            result = await asyncio.to_thread(self._call_esp32, "/backward")
            logger.info("ESP STATUS: %s", result["status_code"])
            logger.info("ESP RESPONSE: %s", result["text"])
            return {"status": "backward", **result}
        except Exception as exc:
            logger.exception("Failed to move robot backward")
            return {"status": "error", "message": str(exc)}

    async def stop_robot(self, request: Any) -> Dict[str, Any]:
        """FastAPI REST POST endpoint /stop to halt the ESP32 robot."""
        logger.info("STOP RECEIVED")
        try:
            result = await asyncio.to_thread(self._call_esp32, "/stop")
            logger.info("ESP STATUS: %s", result["status_code"])
            logger.info("ESP RESPONSE: %s", result["text"])
            return {"status": "stop", **result}
        except Exception as exc:
            logger.exception("Failed to stop robot")
            return {"status": "error", "message": str(exc)}

    def push_notification_to_twilio(self, payload: Dict[str, Any]) -> bool:
        """Sends asynchronous WhatsApp alert to parent via Twilio API when user distress is detected."""
        config = getattr(self.orchestrator, "config_manager", None)

        def _cfg(key: str, default: str = "") -> str:
            if config is not None and hasattr(config, "get"):
                value = config.get(key, default)
                if value is not None:
                    return str(value)
            env_key = {
                "twilio_account_sid": "TWILIO_ACCOUNT_SID",
                "twilio_auth_token": "TWILIO_AUTH_TOKEN",
                "twilio_from_number": "TWILIO_FROM_NUMBER",
                "alert_phone": "ALERT_PHONE",
            }.get(key, key.upper())
            return os.getenv(env_key, default)

        account_sid = _cfg("twilio_account_sid")
        auth_token = _cfg("twilio_auth_token")
        from_number = _cfg("twilio_from_number", "whatsapp:+14155238886")
        to_number = _cfg("alert_phone", "")

        if not account_sid or not auth_token or not to_number:
            logger.warning("Twilio alert skipped because credentials or phone number are missing.")
            return False

        # Normalize numbers
        if not to_number.startswith("whatsapp:"):
            to_number = f"whatsapp:{to_number}"
        if not from_number.startswith("whatsapp:"):
            from_number = f"whatsapp:{from_number}"

        emotion = str(payload.get("emotion", "sad"))
        confidence = payload.get("confidence")
        message_body = payload.get("message") or (
            f"Alert: the camera detected {emotion} emotion"
            + (f" with confidence {confidence:.2f}" if isinstance(confidence, (int, float)) else "")
            + ". Please check in with me."
        )

        async def send_sms_async():
            try:
                import httpx
                url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        url,
                        auth=(account_sid, auth_token),
                        data={"To": to_number, "From": from_number, "Body": message_body},
                    )
                    if response.status_code == 201:
                        logger.info("Twilio WhatsApp notification sent successfully.")
                    else:
                        logger.error(f"Failed to send Twilio notification: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error sending Twilio notification: {e}")

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(send_sms_async())
        except RuntimeError:
            def run_in_thread():
                asyncio.run(send_sms_async())
            threading.Thread(target=run_in_thread, daemon=True).start()

        return True
