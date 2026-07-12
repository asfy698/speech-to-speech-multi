import os
import sys
import pytest
import asyncio
from typing import Generator

# Ensure parent directory is in sys.path in case PYTHONPATH env is missing
sys.path.insert(0, r"C:\Users\assha\OneDrive\Desktop\speech-to-speech-main")

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create a persistent event loop for asynchronous test cases."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
