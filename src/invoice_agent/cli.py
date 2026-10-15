"""Command-line entry point: `invoice-agent --help`."""

import json
import time
from typing import Annotated

import httpx
import typer

from invoice_agent.config import get_settings
from invoice_agent.models_catalog import CANDIDATES, ModelInfo, load_catalog

app = typer.Typer(no_args_is_help=True, add_completion=False)

_SMOKE_INVOICE = """\
ACME SUPPLIES PVT LTD                       TAX INVOICE
GSTIN: 29ABCDE1234F1Z5
Invoice No: INV-2026-0042        Invoice Date: 03/04/2026
PO Number: PO-7781               Due Date: 03/05/2026
Bill To: Globex India, 12 MG Road, Bengaluru 560001

| Description        | Qty | Unit Price | Amount    |
|--------------------|-----|------------|-----------|
| A4 paper (box)     | 10  | 450.00     | 4,500.00  |
| Toner cartridge    | 2   | 3,200.00   | 6,400.00  |

Subtotal: ₹10,900.00   IGST 18%: ₹1,962.00   Total: ₹12,862.00
Terms: Net 30
"""


def _roles(info: ModelInfo) -> str:
    """Summarize which project roles a model qualifies for.

    Args:
        info: The model's catalog entry.

    Returns:
        A comma-separated list such as 'extractor, agent', or '-' if none.
    """
    roles = [r for r, ok in (("extractor", info.can_extract), ("agent", info.can_agent)) if ok]
    return ", ".join(roles) or "-"


@app.command()
def models(
    ids: Annotated[
        list[str] | None, typer.Argument(help="Model slugs (default: candidates)")
    ] = None,
    free: Annotated[bool, typer.Option("--free", help="List every free model instead")] = False,
    refresh: Annotated[bool, typer.Option(help="Re-download the catalog")] = False,
) -> None:
    """Print a table of which models can serve which role (extractor / agent).

    Reads OpenRouter's public catalog (cached for 24h), so it is free and needs no API key.
    Usage: `invoice-agent models`, `invoice-agent models --free`, or
    `invoice-agent models <slug> <slug> ...`.

    Args:
        ids: OpenRouter model slugs to check. If omitted, checks CANDIDATES.
        free: If True, ignore `ids` and list every ':free' model on OpenRouter.
        refresh: If True, bypass the 24h catalog cache.

    Returns:
        None. Prints one row per model: vision / tools support, structured-output mode,
        price per million tokens, and the roles it qualifies for.

    Raises:
        typer.Exit: With code 2 if OpenRouter cannot be reached after retries.
    """
    try:
        catalog = load_catalog(get_settings().cache_dir, refresh=refresh)
    except httpx.HTTPError as e:
        typer.echo(f"Could not reach OpenRouter ({e}). Check your connection and retry.", err=True)
        raise typer.Exit(2) from e
    selected = sorted(m for m in catalog if m.endswith(":free")) if free else (ids or CANDIDATES)
    typer.echo(
        f"{'model':52} {'vision':6} {'tools':5} {'output':11} {'$in/M':>6} {'$out/M':>6}  roles"
    )
    for mid in selected:
        info = catalog.get(mid)
        if info is None:
            typer.echo(f"{mid:52} NOT FOUND on OpenRouter")
            continue
        typer.echo(
            f"{mid:52} {info.vision!s:6} {info.tools!s:5} {info.output_mode:11} "
            f"{info.usd_in_per_m:6.2f} {info.usd_out_per_m:6.2f}  {_roles(info)}"
        )


@app.command()
def smoke(
    model: Annotated[str | None, typer.Option(help="Defaults to EXTRACTOR_MODEL")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Always call the API")] = False,
) -> None:
    """Live end-to-end check: one real LLM call on a small built-in invoice.

    Sends `_SMOKE_INVOICE` to the model using its best structured-output mode, validates
    and normalizes the result, and prints it with timing, attempts and token usage.
    Free on ':free' models; well under $0.01 on paid ones. Repeat runs hit the cache.
    Usage: `invoice-agent smoke` or `invoice-agent smoke --model <slug>`.

    Args:
        model: OpenRouter slug to test. Defaults to EXTRACTOR_MODEL from settings.
        no_cache: If True, skip the response cache and call the API.

    Returns:
        None. Prints the normalized Invoice as JSON, then a summary with model, output
        mode, seconds, attempts, cache hit, token usage and normalization issues.

    Raises:
        MissingAPIKeyError: If OPENROUTER_API_KEY is not set.
        typer.Exit: With code 1 if the model never produced valid output.
    """
    from langchain_core.messages import HumanMessage

    from invoice_agent.llm import get_chat_model
    from invoice_agent.llm_cache import ResponseCache
    from invoice_agent.models_catalog import output_mode_for
    from invoice_agent.normalize import to_invoice
    from invoice_agent.schemas import RawInvoice
    from invoice_agent.structured import StructuredOutputError, invoke_structured

    settings = get_settings()
    model = model or settings.extractor_model
    mode = output_mode_for(model, settings.cache_dir)
    use_cache = settings.llm_cache_enabled and not no_cache
    cache = ResponseCache(settings.cache_dir / "llm.sqlite") if use_cache else None
    prompt = (
        "Transcribe this invoice into the schema. Copy every value exactly as printed.\n\n"
        + _SMOKE_INVOICE
    )
    started = time.perf_counter()
    try:
        result = invoke_structured(
            get_chat_model(model, settings),
            RawInvoice,
            [HumanMessage(prompt)],
            model=model,
            mode=mode,
            cache=cache,
        )
    except StructuredOutputError as e:
        typer.echo(f"FAILED: {e}", err=True)
        raise typer.Exit(1) from e
    finally:
        if cache is not None:
            cache.close()
    elapsed = time.perf_counter() - started
    invoice, issues = to_invoice(result.value)
    typer.echo(invoice.model_dump_json(indent=2))
    typer.echo(
        json.dumps(
            {
                "model": model,
                "output_mode": mode,
                "seconds": round(elapsed, 2),
                "attempts": result.attempts,
                "cached": result.cached,
                "usage": result.usage,
                "normalize_issues": [i.model_dump() for i in issues],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
