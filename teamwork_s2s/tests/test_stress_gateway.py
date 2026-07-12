import time
import asyncio
import threading
import uvicorn
import pytest
import httpx
import websockets
from typing import Any, Dict
from unittest.mock import MagicMock
from teamwork_s2s.src.api.gateway_server import GatewayServer
from teamwork_s2s.src.core.event_bus import EventBus

class MockOrchestrator:
    def __init__(self) -> None:
        self.event_bus = EventBus()
        self.config_manager = MagicMock()
        self.called_chunks = []

    def process_audio_chunk(self, chunk: bytes) -> None:
        # Simulate CPU-heavy blocking processing
        self.called_chunks.append(chunk)
        time.sleep(0.05)  # 50ms synchronous blocking sleep

@pytest.mark.asyncio
async def test_websocket_non_blocking_stress():
    """
    Stress test verifying that concurrent WebSocket audio streams processed
    via run_in_executor do not block the FastAPI asyncio event loop.
    We measure this by sending concurrent /ping HTTP requests during streaming.
    """
    orchestrator = MockOrchestrator()
    server = GatewayServer(orchestrator=orchestrator)

    # Register a test ping endpoint to measure event loop responsiveness
    @server.app.get("/ping")
    async def ping():
        return {"pong": True}

    # Run the server on a unique port
    port = 8092
    config = uvicorn.Config(app=server.app, host="127.0.0.1", port=port, log_level="warning")
    uv_server = uvicorn.Server(config)
    
    # Start uvicorn in a background thread
    server_thread = threading.Thread(target=uv_server.run, daemon=True)
    server_thread.start()
    
    # Allow uvicorn time to start
    await asyncio.sleep(1.0)

    client_errors = []
    ping_latencies = []
    running = True

    async def ping_worker():
        async with httpx.AsyncClient() as client:
            while running:
                t0 = time.perf_counter()
                try:
                    res = await client.get(f"http://127.0.0.1:{port}/ping", timeout=1.0)
                    assert res.status_code == 200
                    assert res.json() == {"pong": True}
                    latency = time.perf_counter() - t0
                    ping_latencies.append(latency)
                except Exception as e:
                    client_errors.append(f"Ping failed: {e}")
                await asyncio.sleep(0.02)  # ping every 20ms

    async def ws_client_worker(client_id: int):
        uri = f"ws://127.0.0.1:{port}/audio"
        try:
            async with websockets.connect(uri) as ws:
                for i in range(10):  # Send 10 chunks of PCM audio
                    chunk = f"chunk_data_{client_id}_{i}".encode()
                    await ws.send(chunk)
                    await asyncio.sleep(0.01)  # 10ms interval
        except Exception as e:
            client_errors.append(f"WS Client {client_id} failed: {e}")

    # Start pinging to measure responsiveness
    ping_task = asyncio.create_task(ping_worker())

    # Spawn 8 concurrent clients streaming audio
    num_clients = 8
    clients = [asyncio.create_task(ws_client_worker(i)) for i in range(num_clients)]
    
    # Wait for all clients to finish streaming
    await asyncio.gather(*clients)
    
    # Wait a bit longer to ensure executor has finished processing all chunks
    await asyncio.sleep(0.5)
    
    # Stop the ping worker
    running = False
    await ping_task

    # Shutdown the uvicorn server
    uv_server.should_exit = True
    server_thread.join(timeout=2.0)

    # Assertions
    assert len(client_errors) == 0, f"Encountered client errors: {client_errors}"
    assert len(orchestrator.called_chunks) == num_clients * 10, "Not all audio chunks were processed"
    
    # If the loop was blocked, ping latencies would soar to > 50ms.
    # We assert that the maximum latency for ping is well below 50ms, verifying it stayed non-blocking.
    assert len(ping_latencies) > 0, "No ping latencies were recorded"
    max_latency = max(ping_latencies)
    print(f"Max ping latency during audio streaming stress: {max_latency:.4f}s")
    # A responsive loop should keep ping latencies under 40ms, typically < 10ms.
    assert max_latency < 0.04, f"Event loop blocked! Max ping latency was {max_latency:.4f}s"

if __name__ == "__main__":
    asyncio.run(test_websocket_non_blocking_stress())
