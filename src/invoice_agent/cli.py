"""Command-line entry point: `invoice-agent --help`."""

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import httpx
import typer

from invoice_agent.config import get_settings
from invoice_agent.models_catalog import CANDIDATES, ModelInfo, load_catalog

if TYPE_CHECKING:
    from invoice_agent.evals.ingestion_report import Ingestor

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


@app.command()
def synth(
    out: Annotated[Path, typer.Option(help="Output folder")] = Path("data/samples"),
    digital: Annotated[int, typer.Option(help="Number of digital invoices")] = 36,
    scanned: Annotated[int, typer.Option(help="How many also get a scanned copy")] = 24,
    seed: Annotated[int, typer.Option(help="Same seed -> identical dataset")] = 42,
    test_fraction: Annotated[float, typer.Option(help="Share held out for testing")] = 0.4,
    channel: Annotated[
        str | None, typer.Option(help="Installed browser to use, e.g. 'msedge' or 'chrome'")
    ] = None,
) -> None:
    """Generate the synthetic invoice dataset (PDFs/PNGs + truth JSON + manifest).

    Usage: `invoice-agent synth --channel msedge` (uses the Edge already on Windows, so
    no Chromium download is needed).

    Args:
        out: Folder to write the documents, truth files and manifest.json into.
        digital: Number of digital invoices to generate.
        scanned: How many of those also get a degraded, image-only copy.
        seed: Master random seed; the dataset is fully reproducible from it.
        test_fraction: Fraction of invoices assigned to the held-out test split.
        channel: Browser for HTML->PDF. None uses Playwright's bundled Chromium.

    Returns:
        None. Writes files and prints a summary (counts by kind, split and template).
    """
    from collections import Counter

    from invoice_agent.synth.generate import generate_dataset

    started = time.perf_counter()
    manifest = generate_dataset(
        out,
        n_digital=digital,
        n_scanned=scanned,
        seed=seed,
        test_fraction=test_fraction,
        channel=channel,
    )
    edge = Counter(e for doc in manifest if doc["kind"] == "digital" for e in doc["edge_cases"])
    summary = {
        "documents": len(manifest),
        "by_kind": dict(Counter(d["kind"] for d in manifest)),
        "by_split": dict(Counter(d["split"] for d in manifest)),
        "by_template": dict(Counter(d["template"] for d in manifest if d["kind"] == "digital")),
        "by_country": dict(Counter(d["country"] for d in manifest if d["kind"] == "digital")),
        "edge_cases": dict(edge),
        "multi_page_docs": sum(d["pages"] > 1 for d in manifest),
        "seconds": round(time.perf_counter() - started, 1),
        "out": str(out),
    }
    typer.echo(json.dumps(summary, indent=2))


def _engine(name: str) -> "Ingestor":
    """Create an ingestion engine by name.

    Args:
        name: 'light', 'docling' (keeps page headers/footers) or 'docling-default'
            (Docling's stock output, which drops them).

    Returns:
        A ready-to-use engine instance.

    Raises:
        typer.BadParameter: For an unknown engine name.
        ImportError: For 'docling' when the optional extra is not installed.
    """
    if name == "light":
        from invoice_agent.ingest.light import LightIngestor

        return LightIngestor()
    if name in ("docling", "docling-default"):
        from invoice_agent.ingest.docling_engine import DoclingIngestor

        return DoclingIngestor(keep_furniture=name == "docling")
    raise typer.BadParameter(f"unknown engine {name!r}; use light, docling or docling-default")


@app.command()
def ingest(
    file: Annotated[Path, typer.Argument(help="PDF or image to ingest", exists=True)],
    engine: Annotated[str, typer.Option(help="'light' or 'docling'")] = "light",
) -> None:
    """Show exactly what an ingestion engine extracts from one document.

    Usage: `invoice-agent ingest data/samples/inv-002-gst_india-IN-scan.pdf --engine docling`.

    Args:
        file: The document to ingest.
        engine: Which engine to use.

    Returns:
        None. Prints a one-line summary (kind, pages, spans, OCR confidence, seconds)
        followed by the layout text an LLM would receive.
    """
    started = time.perf_counter()
    doc = _engine(engine).ingest(file)
    spans = sum(len(p.spans) for p in doc.pages)
    typer.echo(
        f"# engine={doc.engine} kind={doc.kind} pages={len(doc.pages)} spans={spans} "
        f"ocr_conf={doc.ocr_confidence:.3f} seconds={time.perf_counter() - started:.2f}\n"
    )
    typer.echo(doc.text)


@app.command("ingest-eval")
def ingest_eval(
    engines: Annotated[str, typer.Option(help="Comma-separated engines")] = "light,docling",
    data: Annotated[Path, typer.Option(help="Dataset folder")] = Path("data/samples"),
    split: Annotated[str, typer.Option(help="'dev', 'test' or 'all'")] = "dev",
    report: Annotated[Path, typer.Option(help="Report path")] = Path(
        "reports/ingestion_comparison.md"
    ),
) -> None:
    """Compare ingestion engines on the synthetic dataset (no LLM, no cost).

    Scores value recall and line-item row integrity per engine, digital vs scanned,
    plus speed and memory, and writes a Markdown report. Each engine's text per
    document is saved under reports/raw/ingestion/<engine>/ for inspection.

    Args:
        engines: Engines to compare, e.g. 'light,docling'.
        data: Dataset folder with manifest.json.
        split: Which split to evaluate; decisions should be made on 'dev'.
        report: Where to write the Markdown report.

    Returns:
        None. Prints headline numbers as JSON and writes the report.
    """
    from invoice_agent.evals.ingestion_report import run_engines, summary_json, write_report

    instances = [_engine(name.strip()) for name in engines.split(",")]
    runs, peak = run_engines(instances, data, split=split, out_dir=Path("reports/raw/ingestion"))
    write_report(runs, peak, report, split)
    typer.echo(json.dumps({"summary": summary_json(runs), "peak_rss_mb": peak}, indent=2))
    typer.echo(f"report: {report}")


if __name__ == "__main__":
    app()
