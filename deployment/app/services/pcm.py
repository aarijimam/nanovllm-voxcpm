from __future__ import annotations

from typing import AsyncIterator

import numpy as np

from app.services.mp3 import float32_to_s16le_bytes


async def stream_pcm(
    wav_chunks: AsyncIterator[np.ndarray],
) -> AsyncIterator[bytes]:
    """Convert float32 mono waveform chunks to raw int16 LE PCM bytes.

    No background thread needed — PCM conversion is a cheap NumPy cast.
    """
    async for chunk in wav_chunks:
        yield float32_to_s16le_bytes(chunk)
