"""Command-line entry point: `invoice-agent --help`."""

import json
import time
from typing import Annotated

import httpx
import typer

from invoice_agent.config import get_settings
from invoice_agent.models_catalog import CANDIDATES, fetch_catalog

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


@app.command()
def models(
    ids: Annotated[
        list[str] | None, typer.Argument(help="Model slugs (default: candidates)")
    ] = None,
) -> None:
    """Show which candidate models can serve each role (from OpenRouter's public catalog)."""
    try:
        catalog = fetch_catalog()
    except httpx.HTTPError as e:
        typer.echo(f"Could not reach OpenRouter ({e}). Check your connection and retry.", err=True)
        raise typer.Exit(2) from e
    header = f"{'model':40} {'vision':6} {'tools':5} {'schema':6} {'$in/M':>6} {'$out/M':>6}"
    typer.echo(f"{header}  roles")
    for mid in ids or CANDIDATES:
        info = catalog.get(mid)
        if info is None:
            typer.echo(f"{mid:40} NOT FOUND on OpenRouter")
            continue
        roles = (
            ", ".join(
                r for r, ok in (("extractor", info.can_extract), ("agent", info.can_agent)) if ok
            )
            or "-"
        )
        typer.echo(
            f"{mid:40} {info.vision!s:6} {info.tools!s:5} {info.json_schema!s:6} "
            f"{info.usd_in_per_m:6.2f} {info.usd_out_per_m:6.2f}  {roles}"
        )


@app.command()
def smoke(
    model: Annotated[str | None, typer.Option(help="Defaults to EXTRACTOR_MODEL")] = None,
) -> None:
    """Live check: extract a tiny text invoice with strict JSON-schema output."""
    from invoice_agent.llm import get_chat_model
    from invoice_agent.normalize import to_invoice
    from invoice_agent.schemas import RawInvoice

    model = model or get_settings().extractor_model
    llm = get_chat_model(model).with_structured_output(
        RawInvoice, method="json_schema", strict=True, include_raw=True
    )
    started = time.perf_counter()
    out = llm.invoke(
        "Transcribe this invoice into the schema. Copy every value exactly as printed.\n\n"
        + _SMOKE_INVOICE
    )
    elapsed = time.perf_counter() - started
    raw = out["parsed"]
    if raw is None:
        typer.echo(f"FAILED to parse: {out['parsing_error']}", err=True)
        raise typer.Exit(1)
    invoice, issues = to_invoice(raw)
    typer.echo(invoice.model_dump_json(indent=2))
    usage = getattr(out["raw"], "usage_metadata", None)
    typer.echo(
        json.dumps(
            {
                "model": model,
                "seconds": round(elapsed, 2),
                "usage": usage,
                "normalize_issues": [i.model_dump() for i in issues],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
