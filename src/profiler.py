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
    """Mede tempo de parede e pico de alocação de memória de uma subrotina crítica.

    Caso o tracemalloc já esteja ativo em um escopo externo (como no Branch-and-Bound
    ou no pipeline global), utiliza reset_peak() após capturar o pico acumulado anterior
    para isolar o pico específico da subrotina sem interromper o rastreamento pai.
    """
    already_tracing = tracemalloc.is_tracing()
    prev_peak = 0
    if not already_tracing:
        tracemalloc.start()
    else:
        _, prev_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()

    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        _, sub_peak = tracemalloc.get_traced_memory()
        peak_kb = sub_peak / 1024.0
        if not already_tracing:
            tracemalloc.stop()
        elif prev_peak > sub_peak:
            pass

        time_key = f"{prefix}_time"
        mem_key = f"{prefix}_peak_kb"
        telemetry[time_key] = telemetry.get(time_key, 0.0) + elapsed
        telemetry[mem_key] = max(telemetry.get(mem_key, 0.0), peak_kb)
        telemetry["peak_memory_kb"] = max(
            telemetry.get("peak_memory_kb", 0.0),
            peak_kb,
            prev_peak / 1024.0,
        )
