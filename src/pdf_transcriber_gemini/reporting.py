"""Optional structured progress events; normal CLI output remains available.

A context-local sink avoids threading UI parameters through the Gemini SDK-facing
functions. Batch mode captures events for one redrawable terminal dashboard, while
plain `go` and redirected output continue to emit readable text.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator

EventSink = Callable[[str, str], None]
_sink: ContextVar[EventSink | None] = ContextVar("pdf_transcriber_progress_sink", default=None)


@contextmanager
def capture(sink: EventSink | None) -> Iterator[None]:
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


def emit(event: str, message: str) -> None:
    sink = _sink.get()
    if sink is not None:
        sink(event, message)
    else:
        destination = sys.stderr if event in {"warning", "note", "recovery"} else sys.stdout
        print(message, file=destination, flush=True)
