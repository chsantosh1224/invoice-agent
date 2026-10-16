"""Model bake-off: run (model x input) configurations over dev invoices and compare them.

Free-tier friendly: every successful reply is cached, failures (rate limits, invalid JSON)
are recorded instead of aborting, and re-running only re-tries what failed.
"""

import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from invoice_agent.config import Settings
from invoice_agent.evals.extraction import score_extraction
from invoice_agent.extraction.messages import build_messages
from invoice_agent.extraction.prompts import PROMPT_VERSION
from invoice_agent.ingest.pages import page_images
from invoice_agent.llm import get_chat_model
from invoice_agent.llm_cache import ResponseCache
from invoice_agent.models_catalog import output_mode_for
from invoice_agent.normalize import to_invoice
from invoice_agent.schemas import RawInvoice
from invoice_agent.structured import StructuredOutputError, invoke_structured

log = structlog.get_logger()


@dataclass(frozen=True)
class Config:
    """One bake-off contender: a model and what it is shown.

    Attributes:
        name: Short label for reports, e.g. 'gemma-4-31b · image+light' (no '|': it breaks tables).
        model: OpenRouter slug.
        inputs: Any of 'image', 'light', 'docling' (page images / engine text).
    """

    name: str
    model: str
    inputs: tuple[str, ...]


DEFAULT_CONFIGS = (
    Config("gemma-4-31b · image+light", "google/gemma-4-31b-it:free", ("image", "light")),
    Config("gemma-4-31b · image", "google/gemma-4-31b-it:free", ("image",)),
    Config("dots-3 · image+light", "dots-studio/dots-3-note-preview:free", ("image", "light")),
    Config("dots-3 · image", "dots-studio/dots-3-note-preview:free", ("image",)),
    Config("nemotron-omni · image+light",
           "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", ("image", "light")),
    Config("nemotron-super · light", "nvidia/nemotron-3-super-120b-a12b:free", ("light",)),
    Config("nemotron-super · docling", "nvidia/nemotron-3-super-120b-a12b:free", ("docling",)),
)  # fmt: skip


def select_documents(manifest: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Pick a small, varied dev subset: every template, both kinds, at least one multi-page.

    Args:
        manifest: The dataset manifest.
        n: How many documents to pick.

    Returns:
        Up to n dev-split documents, rotating through templates and alternating
        digital/scanned; the last pick is swapped for a multi-page document if none
        was chosen.
    """
    dev = [d for d in manifest if d["split"] == "dev"]
    templates = sorted({d["template"] for d in dev})
    chosen: list[dict[str, Any]] = []
    for i in range(min(n, len(dev))):
        template, kind = templates[i % len(templates)], ("digital", "scanned")[i % 2]
        pool = [d for d in dev if d not in chosen]
        pick = next(
            (d for d in pool if d["template"] == template and d["kind"] == kind),
            next((d for d in pool if d["template"] == template), pool[0]),
        )
        chosen.append(pick)
    if chosen and not any(d["pages"] > 1 for d in chosen):
        multi = next((d for d in dev if d["pages"] > 1 and d not in chosen), None)
        if multi:
            chosen[-1] = multi
    return chosen


def _document_text(engine: str, doc: dict[str, Any], data_dir: Path) -> str:
    """Get an ingestion engine's text for a document, reusing the ingestion-eval output.

    Args:
        engine: 'light' or 'docling'.
        doc: Manifest entry.
        data_dir: Dataset folder.

    Returns:
        The engine's layout text. Read from reports/raw/ingestion/<engine>/<id>.txt when
        `invoice-agent ingest-eval` already produced it; otherwise ingested now.
    """
    saved = Path("reports/raw/ingestion") / engine / f"{doc['id']}.txt"
    if saved.exists():
        return saved.read_text(encoding="utf-8")
    if engine == "light":
        from invoice_agent.ingest.light import LightIngestor

        return LightIngestor().ingest(data_dir / doc["file"]).text
    from invoice_agent.ingest.docling_engine import DoclingIngestor

    return DoclingIngestor().ingest(data_dir / doc["file"]).text


def run_one(
    config: Config, doc: dict[str, Any], data_dir: Path, settings: Settings, cache: ResponseCache
) -> dict[str, Any]:
    """Extract one document with one configuration and score it.

    Args:
        config: The contender.
        doc: Manifest entry of the document.
        data_dir: Dataset folder.
        settings: App settings (API key, retries, ...).
        cache: Response cache shared by the whole bake-off.

    Returns:
        A flat, JSON-serializable record: status ('ok', 'invalid_output' or 'error'),
        timing, attempts, token usage and - when ok - the scores.
    """
    record: dict[str, Any] = {
        "config": config.name, "model": config.model, "doc": doc["id"], "kind": doc["kind"],
        "template": doc["template"], "pages": doc["pages"], "prompt_version": PROMPT_VERSION,
    }  # fmt: skip
    path = data_dir / doc["file"]
    images = page_images(path, dpi=150) if "image" in config.inputs else None
    texts = [_document_text(e, doc, data_dir) for e in ("light", "docling") if e in config.inputs]
    messages = build_messages(images=images, text="\n\n".join(texts) or None)
    mode = output_mode_for(config.model, settings.cache_dir)
    truth = json.loads((data_dir / f"{doc['id']}.truth.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    try:
        result = invoke_structured(
            get_chat_model(config.model, settings), RawInvoice, messages,
            model=config.model, mode=mode, cache=cache,
        )  # fmt: skip
    except StructuredOutputError as e:
        return record | {"status": "invalid_output", "error": str(e)[:300],
                         "seconds": time.perf_counter() - started}  # fmt: skip
    except Exception as e:  # rate limits, provider errors, timeouts: record and move on
        return record | {"status": "error", "error": f"{type(e).__name__}: {str(e)[:300]}",
                         "seconds": time.perf_counter() - started}  # fmt: skip
    invoice, issues = to_invoice(result.value)
    score = score_extraction(invoice, truth["invoice"])
    return record | {
        "status": "ok", "mode": mode, "cached": result.cached, "attempts": result.attempts,
        "seconds": time.perf_counter() - started, "usage": result.usage,
        "field_accuracy": score.field_accuracy, "items_f1": score.items_f1,
        "items_matched": score.items_matched, "descriptions_ok": score.descriptions_ok,
        "fields": score.fields, "hallucinated": score.hallucinated, "missed": score.missed,
        "normalize_issues": [i.code for i in issues],
    }  # fmt: skip


def run_bakeoff(
    configs: list[Config],
    data_dir: Path,
    settings: Settings,
    *,
    n_docs: int = 10,
    pause_s: float = 3.0,
    out: Path = Path("reports/raw/bakeoff/results.jsonl"),
) -> list[dict[str, Any]]:
    """Run every configuration on the selected dev documents.

    Args:
        configs: Contenders to compare.
        data_dir: Dataset folder with manifest.json.
        settings: App settings.
        n_docs: Number of dev documents (see `select_documents`).
        pause_s: Seconds to wait after each live (non-cached) call, to stay under
            free-tier per-minute limits.
        out: JSONL file receiving one record per (config, document).

    Returns:
        All records.
    """
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    docs = select_documents(manifest, n_docs)
    cache = ResponseCache(settings.cache_dir / "llm.sqlite")
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        with out.open("w", encoding="utf-8") as fh:
            for config in configs:
                for doc in docs:
                    rec = run_one(config, doc, data_dir, settings, cache)
                    log.info("bakeoff", config=config.name, doc=doc["id"], status=rec["status"],
                             acc=rec.get("field_accuracy"), cached=rec.get("cached"))  # fmt: skip
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    records.append(rec)
                    if not rec.get("cached"):
                        time.sleep(pause_s)
    finally:
        cache.close()
    return records


def _pct(x: float) -> str:
    """Format a ratio as a percentage.

    Args:
        x: A ratio in 0..1.

    Returns:
        e.g. '93.3%'.
    """
    return f"{x:.1%}"


def write_report(records: list[dict[str, Any]], path: Path) -> str:
    """Summarize bake-off records as a Markdown report.

    Accuracy is shown two ways: over documents that produced output, and end-to-end with
    failures counted as 0 - a model that often fails must not look good.

    Args:
        records: Records from `run_bakeoff`.
        path: Report file to write.

    Returns:
        The Markdown text.
    """
    configs = list(dict.fromkeys(r["config"] for r in records))
    docs = sorted({r["doc"] for r in records})
    lines = [
        "# Model bake-off",
        "",
        f"Prompt `{PROMPT_VERSION}` · {len(docs)} dev invoices · free OpenRouter models.",
        "",
        "Decisions drawn from these numbers: [docs/decisions.md](../docs/decisions.md).",
        "",
        "- **Acc (ok)**: header-field accuracy on documents that produced valid output.",
        "- **Acc (e2e)**: same, counting failed documents as 0.",
        "- **Items F1**: line items with qty, unit price and amount all correct.",
        "- **Halluc./doc**: values returned where the invoice has none (worst error type).",
        "",
        "| Config | OK | Acc (ok) | Acc (e2e) | Items F1 | Desc. ok | Halluc./doc "
        "| Repairs | Median s | Failures |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in configs:
        rs = [r for r in records if r["config"] == c]
        ok = [r for r in rs if r["status"] == "ok"]
        acc_ok = statistics.mean(r["field_accuracy"] for r in ok) if ok else 0.0
        acc_e2e = sum(r["field_accuracy"] for r in ok) / len(rs) if rs else 0.0
        f1 = statistics.mean(r["items_f1"] for r in ok) if ok else 0.0
        matched = sum(r["items_matched"] for r in ok)
        desc = sum(r["descriptions_ok"] for r in ok) / matched if matched else 0.0
        halluc = sum(len(r["hallucinated"]) for r in ok) / len(ok) if ok else 0.0
        repairs = sum(r["attempts"] - 1 for r in ok if not r["cached"])
        live = [r["seconds"] for r in ok if not r["cached"]]
        fails = Counter(r["status"] for r in rs if r["status"] != "ok")
        lines.append(
            f"| {c} | {len(ok)}/{len(rs)} | {_pct(acc_ok)} | {_pct(acc_e2e)} | {_pct(f1)} "
            f"| {_pct(desc)} | {halluc:.2f} | {repairs} "
            f"| {statistics.median(live) if live else 0:.1f} "
            f"| {', '.join(f'{k} {v}' for k, v in fails.items()) or '-'} |"
        )

    field_names = sorted({f for r in records if r["status"] == "ok" for f in r["fields"]})
    header = "| Field | " + " | ".join(configs) + " |"
    lines += ["", "## Field accuracy by config", "", header, "|---|" + "---|" * len(configs)]
    for f in field_names:
        cells = []
        for c in configs:
            ok = [r for r in records if r["config"] == c and r["status"] == "ok"]
            cells.append(_pct(sum(r["fields"][f] for r in ok) / len(ok)) if ok else "-")
        lines.append(f"| {f} | " + " | ".join(cells) + " |")

    lines += ["", "## By document kind (Acc ok / Items F1)", "",
              "| Config | digital | scanned |", "|---|---|---|"]  # fmt: skip
    for c in configs:
        cells = []
        for kind in ("digital", "scanned"):
            ok = [r for r in records if r["config"] == c and r["status"] == "ok"
                  and r["kind"] == kind]  # fmt: skip
            cells.append(
                f"{_pct(statistics.mean(r['field_accuracy'] for r in ok))} / "
                f"{_pct(statistics.mean(r['items_f1'] for r in ok))}"
                if ok
                else "-"
            )
        lines.append(f"| {c} | " + " | ".join(cells) + " |")

    errors = [r for r in records if r["status"] != "ok"]
    if errors:
        lines += ["", "## Failures", ""]
        lines += [f"- `{r['config']}` on `{r['doc']}`: {r['error'][:160]}" for r in errors]
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
