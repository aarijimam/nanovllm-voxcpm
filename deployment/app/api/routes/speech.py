from __future__ import annotations

import base64
import inspect
import time
from typing import Any, AsyncIterator

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from numpy.typing import NDArray

from app.api.deps import get_server
from app.core.metrics import (
    GENERATE_AUDIO_SECONDS_TOTAL,
    GENERATE_STREAM_BYTES_TOTAL,
    GENERATE_TTFB_SECONDS,
)
from app.schemas.http import ErrorResponse, SpeechRequest
from app.services.pcm import stream_pcm

router = APIRouter(tags=["generation"])


@router.post(
    "/v1/audio/speech",
    response_class=StreamingResponse,
    summary="Generate audio — OpenAI-compatible PCM streaming",
    responses={
        200: {
            "description": "Raw PCM byte stream (int16 LE at model sample rate)",
            "content": {"audio/pcm": {"schema": {"type": "string", "format": "binary"}}},
            "headers": {
                "X-Audio-Sample-Rate": {"description": "Sample rate in Hz.", "schema": {"type": "integer"}},
                "X-Audio-Channels": {"description": "Number of channels.", "schema": {"type": "integer"}},
            },
        },
        400: {"description": "Invalid input", "model": ErrorResponse},
        503: {"description": "Model server not ready", "model": ErrorResponse},
        500: {"description": "Internal error", "model": ErrorResponse},
    },
)
async def speech(
    req: SpeechRequest,
    request: Request,
    server: Any = Depends(get_server),
) -> StreamingResponse:
    """Generate speech as a streamed raw PCM byte stream (int16 LE at the model sample rate).

    Maps the OpenAI Audio Speech interface:
    - `input`      → target_text
    - `ref_audio`  → encoded via server.encode_latents() → ref_audio_latents
    - `voice`      → ignored (voice is controlled by ref_audio)
    """
    model_info = await server.get_model_info()
    sample_rate = int(model_info["sample_rate"])
    channels = int(model_info["channels"])

    if channels != 1:
        raise HTTPException(status_code=500, detail=f"Only mono is supported (channels={channels})")

    ref_audio_latents: bytes | None = None
    if req.ref_audio is not None:
        try:
            wav_bytes = base64.b64decode(req.ref_audio)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 in ref_audio: {e}") from e
        try:
            ref_audio_latents = await server.encode_latents(wav_bytes, req.ref_audio_format)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    if ref_audio_latents is not None:
        if "ref_audio_latents" not in inspect.signature(server.generate).parameters:
            raise HTTPException(status_code=400, detail="Reference audio is not supported by the loaded model")

    generate_kwargs: dict[str, Any] = {
        "target_text": req.input,
        "prompt_latents": None,
        "prompt_text": "",
        "max_generate_length": 2000,
        "temperature": 1.0,
        "cfg_value": 1.5,
    }
    if ref_audio_latents is not None:
        generate_kwargs["ref_audio_latents"] = ref_audio_latents

    gen = server.generate(**generate_kwargs)

    first_chunk: NDArray[np.float32] | None = None
    stream_exhausted = False
    try:
        first_chunk = await anext(gen)
    except StopAsyncIteration:
        stream_exhausted = True
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    start_t = time.perf_counter()
    ttfb_recorded = False

    async def wav_chunks() -> AsyncIterator[NDArray[np.float32]]:
        if first_chunk is not None:
            GENERATE_AUDIO_SECONDS_TOTAL.inc(float(first_chunk.shape[0]) / float(sample_rate))
            yield first_chunk
        if stream_exhausted:
            return
        try:
            async for chunk in gen:
                GENERATE_AUDIO_SECONDS_TOTAL.inc(float(chunk.shape[0]) / float(sample_rate))
                yield chunk
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    async def body() -> AsyncIterator[bytes]:
        nonlocal ttfb_recorded
        async for b in stream_pcm(wav_chunks()):
            if not ttfb_recorded:
                GENERATE_TTFB_SECONDS.observe(time.perf_counter() - start_t)
                ttfb_recorded = True
            GENERATE_STREAM_BYTES_TOTAL.inc(len(b))
            yield b
        if not ttfb_recorded:
            GENERATE_TTFB_SECONDS.observe(time.perf_counter() - start_t)

    return StreamingResponse(
        body(),
        media_type="audio/pcm",
        headers={
            "X-Audio-Sample-Rate": str(sample_rate),
            "X-Audio-Channels": str(channels),
        },
    )
