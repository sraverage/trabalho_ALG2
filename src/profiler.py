from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
import tracemalloc
from typing import Iterator


@dataclass(frozen=True)
class SubroutineStats:
    elapsed_sec: float
    peak_memory_kb: float


@contextmanager
def profile_block(telemetry: dict[str, float], prefix: str) -> Iterator[None]:
    """Mede tempo de parede e pico de memória de um trecho de código.

    Se o tracemalloc já estiver ativo em um bloco externo, usa reset_peak() para
    isolar o pico desta subrotina sem interromper o rastreamento do contexto pai.
    """
    already_tracing = tracemalloc.is_tracing()
    prev_peak = 0

    if already_tracing:
        _, prev_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()

    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        _, sub_peak = tracemalloc.get_traced_memory()
        if not already_tracing:
            tracemalloc.stop()

        peak_kb = sub_peak / 1024.0
        time_key = f"{prefix}_time"
        mem_key = f"{prefix}_peak_kb"

        telemetry[time_key] = telemetry.get(time_key, 0.0) + elapsed
        telemetry[mem_key] = max(telemetry.get(mem_key, 0.0), peak_kb)
        telemetry["peak_memory_kb"] = max(
            telemetry.get("peak_memory_kb", 0.0),
            peak_kb,
            prev_peak / 1024.0,
        )
