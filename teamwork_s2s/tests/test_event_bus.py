import asyncio
import pytest
from teamwork_s2s.src.core.event_bus import EventBus

def test_event_bus_sync():
    eb = EventBus()
    received = []
    eb.subscribe("test_topic", lambda p: received.append(p))
    eb.publish("test_topic", {"val": 1})
    assert received == [{"val": 1}]

def test_event_bus_async_no_loop():
    eb = EventBus()
    received = []
    
    async def async_cb(p):
        received.append(p)
        
    eb.subscribe("test_topic_async", async_cb)
    eb.publish("test_topic_async", {"val": 2})
    assert received == [{"val": 2}]

@pytest.mark.asyncio
async def test_event_bus_in_loop():
    eb = EventBus()
    sync_done = asyncio.Event()
    async_done = asyncio.Event()
    
    payloads = {}
    
    def sync_cb(p):
        payloads["sync"] = p
        sync_done.set()
        
    async def async_cb(p):
        payloads["async"] = p
        async_done.set()
        
    eb.subscribe("event", sync_cb)
    eb.subscribe("event", async_cb)
    
    eb.publish("event", {"data": "ok"})
    
    await asyncio.wait_for(asyncio.gather(sync_done.wait(), async_done.wait()), timeout=1.0)
    assert payloads["sync"] == {"data": "ok"}
    assert payloads["async"] == {"data": "ok"}

def test_event_bus_error_isolation():
    eb = EventBus()
    received = []
    
    def broken_cb(p):
        raise ValueError("Simulated failure")
        
    def working_cb(p):
        received.append(p)
        
    eb.subscribe("broken_topic", broken_cb)
    eb.subscribe("broken_topic", working_cb)
    
    # Should not raise exception out of publish
    eb.publish("broken_topic", {"val": 42})
    assert received == [{"val": 42}]

def test_event_bus_unsubscribe():
    eb = EventBus()
    received = []
    def cb(p):
        received.append(p)
    eb.subscribe("test_topic", cb)
    eb.publish("test_topic", {"val": 1})
    assert received == [{"val": 1}]
    eb.unsubscribe("test_topic", cb)
    eb.publish("test_topic", {"val": 2})
    assert received == [{"val": 1}]

