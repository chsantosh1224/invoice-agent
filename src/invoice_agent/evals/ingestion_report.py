"""Run ingestion engines over the dataset and write a Markdown comparison report."""

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import psutil

from invoice_agent.evals.ingestion import DocScore, score_document
from invoice_agent.ingest.document import Document


class Ingestor(Protocol):
    """Anything that can turn a file into a Document (LightIngestor, DoclingIngestor)."""

    name: str

    def ingest(self, path: Path) -> Document:
        """Ingest one file.

        Args:
            path: A PDF or image file.

        Returns:
            The ingested Document.
        """
        ...


@dataclass
class Run:
    """One engine's results on one document.

    Attributes:
        engine: Engine name.
        kind: 'digital' or 'scanned' (from the dataset manifest).
        template: Dataset template name.
        score: Recall and row-integrity scores.
        seconds: Wall-clock ingestion time.
        pages: Page count.
    """

    engine: str
    kind: str
    template: str
    score: DocScore
    seconds: float
    pages: int


def run_engines(
    engines: list[Ingestor], data_dir: Path, *, split: str, out_dir: Path | None = None
) -> tuple[list[Run], dict[str, float]]:
    """Ingest every document of a split with every engine and score the output.

    Args:
        engines: Engines to compare.
        data_dir: Dataset folder containing manifest.json and the documents.
        split: 'dev', 'test' or 'all'.
        out_dir: If given, each engine's text is saved as <out_dir>/<engine>/<doc>.txt
            so you can read exactly what the LLM would see.

    Returns:
        (runs, peak_rss_mb): every Run, and each engine's peak process memory in MB.
    """
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    docs = [d for d in manifest if split == "all" or d["split"] == split]
    proc = psutil.Process()
    runs: list[Run] = []
    peak: dict[str, float] = {}
    for engine in engines:
        peak[engine.name] = proc.memory_info().rss / 2**20
        for meta in docs:
            truth = json.loads((data_dir / f"{meta['id']}.truth.json").read_text(encoding="utf-8"))[
                "invoice"
            ]
            started = time.perf_counter()
            doc = engine.ingest(data_dir / meta["file"])
            seconds = time.perf_counter() - started
            peak[engine.name] = max(peak[engine.name], proc.memory_info().rss / 2**20)
            if out_dir is not None:
                (out_dir / engine.name).mkdir(parents=True, exist_ok=True)
                (out_dir / engine.name / f"{meta['id']}.txt").write_text(doc.text, encoding="utf-8")
            runs.append(
                Run(
                    engine.name,
                    meta["kind"],
                    meta["template"],
                    score_document(meta["id"], doc.text, truth),
                    seconds,
                    meta["pages"],
                )
            )
    return runs, peak


def _summary(runs: list[Run]) -> dict[str, float]:
    """Aggregate a group of runs into headline numbers.

    Args:
        runs: Runs of one engine (optionally filtered further).

    Returns:
        Dict with recall (micro-averaged over all checked values), row integrity,
        seconds per page and document count.
    """
    found = sum(sum(r.score.found.values()) for r in runs)
    checked = sum(len(r.score.found) for r in runs)
    rows = sum(r.score.rows_total for r in runs)
    intact = sum(r.score.rows_intact for r in runs)
    pages = sum(r.pages for r in runs)
    return {
        "recall": found / checked if checked else 1.0,
        "rows": intact / rows if rows else 1.0,
        "sec_per_page": sum(r.seconds for r in runs) / pages if pages else 0.0,
        "docs": len(runs),
    }


def write_report(runs: list[Run], peak_rss_mb: dict[str, float], path: Path, split: str) -> str:
    """Write the comparison as a Markdown report.

    Args:
        runs: All runs from `run_engines`.
        peak_rss_mb: Peak memory per engine.
        path: Where to write the report (parent dirs are created).
        split: The split that was evaluated, for the report header.

    Returns:
        The Markdown text that was written.
    """
    engines = sorted({r.engine for r in runs})
    lines = [
        "# Ingestion comparison",
        "",
        f"Split: **{split}** · documents: {len({r.score.doc_id for r in runs})} · no LLM involved.",
        "",
        "- **Value recall**: share of truth values (ids, names, dates, amounts) present in the "
        "engine's text. A value missing here cannot be extracted from text.",
        "- **Row integrity**: share of line items whose qty, unit price and amount sit on one "
        "line. Low = a text-only model must guess which numbers belong together.",
        "",
        "## Overall",
        "",
        "| Engine | Kind | Docs | Value recall | Row integrity | Sec/page |",
        "|---|---|---|---|---|---|",
    ]
    for engine in engines:
        for kind in ("digital", "scanned", "all"):
            group = [r for r in runs if r.engine == engine and kind in ("all", r.kind)]
            s = _summary(group)
            lines.append(
                f"| {engine} | {kind} | {s['docs']:.0f} | {s['recall']:.1%} | {s['rows']:.1%} "
                f"| {s['sec_per_page']:.2f} |"
            )
    lines += ["", "## Peak memory (process RSS)", "", "| Engine | Peak MB |", "|---|---|"]
    lines += [f"| {e} | {peak_rss_mb[e]:.0f} |" for e in engines]

    lines += ["", "## By template (all kinds)", "", "| Template | "
              + " | ".join(f"{e} recall | {e} rows" for e in engines) + " |",
              "|---|" + "---|---|" * len(engines)]  # fmt: skip
    for template in sorted({r.template for r in runs}):
        cells = []
        for e in engines:
            s = _summary([r for r in runs if r.engine == e and r.template == template])
            cells += [f"{s['recall']:.1%}", f"{s['rows']:.1%}"]
        lines.append(f"| {template} | " + " | ".join(cells) + " |")

    misses: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in runs:
        for field_name, ok in r.score.found.items():
            if not ok:
                misses[r.engine][field_name.split("[")[0]] += 1
    lines += ["", "## Most-missed fields", ""]
    for e in engines:
        top = sorted(misses[e].items(), key=lambda kv: -kv[1])[:6]
        lines.append(f"- **{e}**: " + (", ".join(f"{k} ({v})" for k, v in top) or "none"))

    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def summary_json(runs: list[Run]) -> dict[str, Any]:
    """Headline numbers per engine and kind, for printing in the CLI.

    Args:
        runs: All runs.

    Returns:
        {engine: {kind: summary}} with the numbers from `_summary`.
    """
    out: dict[str, Any] = {}
    for e in sorted({r.engine for r in runs}):
        out[e] = {
            k: _summary([r for r in runs if r.engine == e and k in ("all", r.kind)])
            for k in ("digital", "scanned", "all")
        }
    return out
