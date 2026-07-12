import asyncio
import threading
import logging
from typing import Dict, List, Callable, Any

logger = logging.getLogger(__name__)

# Strict contract: Callback receives Dict[str, Any], returns Any
CallbackType = Callable[[Dict[str, Any]], Any]

class EventBus:
    """Thread-safe event router supporting async and sync subscription callbacks."""

    def __init__(self) -> None:
        """Initializes EventBus with a registry and thread lock."""
        self.subscribers: Dict[str, List[CallbackType]] = {}
        self._lock = threading.Lock()

    def subscribe(self, topic: str, callback: CallbackType) -> None:
        """Subscribes a callback function to a specific topic."""
        with self._lock:
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            self.subscribers[topic].append(callback)
            logger.debug(f"Subscribed callback to topic '{topic}'")

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        """Publishes an event and payload to all subscribers of a topic."""
        with self._lock:
            if topic not in self.subscribers:
                return
            # Create a copy of callbacks list under lock to safely iterate over it
            callbacks = list(self.subscribers[topic])

        for callback in callbacks:
            try:
                # Retrieve current running loop
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop is not None and loop.is_running():
                    if asyncio.iscoroutinefunction(callback):
                        # Schedule coroutine on active loop
                        loop.create_task(callback(payload))
                    else:
                        # Schedule synchronous callback in thread pool to prevent blocking loop
                        loop.run_in_executor(None, callback, payload)
                else:
                    if asyncio.iscoroutinefunction(callback):
                        # Run coroutine synchronously when no loop is active
                        asyncio.run(callback(payload))
                    else:
                        # Direct synchronous call
                        callback(payload)
            except Exception as e:
                logger.error(f"Error executing callback for topic '{topic}': {e}", exc_info=True)

    def unsubscribe(self, topic: str, callback: CallbackType) -> None:
        """Unsubscribes a callback function from a specific topic."""
        with self._lock:
            if topic in self.subscribers:
                try:
                    self.subscribers[topic].remove(callback)
                    logger.debug(f"Unsubscribed callback from topic '{topic}'")
                    if not self.subscribers[topic]:
                        del self.subscribers[topic]
                except ValueError:
                    logger.warning(f"Callback not found in subscribers for topic '{topic}'")

