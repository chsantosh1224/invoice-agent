"""Structured JSON logging. Every LLM call logs run_id, model, prompt version and usage."""

import logging

import structlog


def configure_logging(level: str = "INFO", *, json: bool = True) -> None:
    """Set up structlog so every log line is a timestamped, structured event.

    Call once at process start (CLI, API server, MCP server). Values bound with
    `structlog.contextvars.bind_contextvars(run_id=...)` are added to every line.

    Args:
        level: Minimum log level to emit, e.g. 'DEBUG', 'INFO', 'WARNING'.
        json: True for one JSON object per line (production, log shipping);
            False for colored human-readable output (local debugging).

    Returns:
        None. Configures the global structlog and stdlib logging state.
    """
    logging.basicConfig(format="%(message)s", level=level)
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
    )
