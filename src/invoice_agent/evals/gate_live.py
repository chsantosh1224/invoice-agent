"""Calibrate the validation gate on real model output.

Runs the fast path on every document of a split, compares each extraction with the truth,
and measures what matters to accounts payable:
- auto-accept rate: share of invoices accepted without a human
- silent error rate: share of accepted invoices that are actually wrong (target <= 1%)
- catch rate: share of wrong extractions the gate sent to review
A threshold sweep shows how these trade off.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import structlog

from invoice_agent.config import Settings
from invoice_agent.evals.extraction import score_extraction
from invoice_agent.extraction.fast_path import extract_document
from invoice_agent.llm_cache import ResponseCache

log = structlog.get_logger()
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


@dataclass
class LiveTrial:
    """The fast path's outcome on one document, compared with the truth.

    Attributes:
        doc: Document id.
        kind: 'digital' or 'scanned'.
        extracted: False if no model produced output (rate limits etc.).
        has_errors: The gate found at least one error-severity issue.
        score: The gate's confidence score.
        correct: The extraction is fully right (every header field and every line item).
        wrong_fields: Fields that were wrong, for the report.
        issues: Issue codes the gate raised.
        seconds: Wall-clock time.
    """

    doc: str
    kind: str
    extracted: bool
    has_errors: bool
    score: float
    correct: bool
    wrong_fields: list[str]
    issues: list[str]
    seconds: float

    def accepted(self, threshold: float) -> bool:
        """Whether the gate would auto-accept this document at a threshold.

        Args:
            threshold: Minimum score.

        Returns:
            True if extraction succeeded, no gate errors, and score >= threshold.
        """
        return self.extracted and not self.has_errors and self.score >= threshold


def run_live(data_dir: Path, settings: Settings, *, split: str = "dev") -> list[LiveTrial]:
    """Run the fast path on every document of a split and score it against the truth.

    Args:
        data_dir: Dataset folder with manifest.json.
        settings: App settings (models, API key).
        split: 'dev', 'test' or 'all'.

    Returns:
        One LiveTrial per document.
    """
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = ResponseCache(settings.cache_dir / "llm.sqlite")
    trials = []
    try:
        for meta in (d for d in manifest if split == "all" or d["split"] == split):
            truth = json.loads((data_dir / f"{meta['id']}.truth.json").read_text(encoding="utf-8"))[
                "invoice"
            ]
            result, seconds = extract_document(data_dir / meta["file"], settings, cache=cache)
            extracted = not any(i.code == "extract.failed" for i in result.issues)
            score = score_extraction(result.invoice, truth) if extracted else None
            wrong = [f for f, ok in score.fields.items() if not ok] if score else []
            if score and score.items_f1 < 1:
                wrong.append("line_items")
            trials.append(
                LiveTrial(
                    doc=meta["id"],
                    kind=meta["kind"],
                    extracted=extracted,
                    has_errors=any(i.severity == "error" for i in result.issues),
                    score=result.score,
                    correct=extracted and not wrong,
                    wrong_fields=wrong,
                    issues=[i.code for i in result.issues],
                    seconds=seconds,
                )
            )
            log.info("gate_live", doc=meta["id"], status=result.status, correct=trials[-1].correct)
    finally:
        cache.close()
    return trials


def write_report(trials: list[LiveTrial], path: Path, *, chosen: float) -> str:
    """Write the calibration report: outcome table, threshold sweep, and every mistake.

    Args:
        trials: Results of `run_live`.
        path: Report file.
        chosen: The threshold currently configured (highlighted in the sweep).

    Returns:
        The Markdown text.
    """
    done = [t for t in trials if t.extracted]
    lines = [
        "# Validation gate: calibration on real extractions",
        "",
        f"Fast path on {len(trials)} dev documents ({len(done)} extracted; the rest failed "
        "at the provider). 'Correct' = every header field and every line item right.",
        "",
        "## Threshold sweep",
        "",
        "| Threshold | Auto-accepted | Silent errors (accepted but wrong) | Wrong caught |",
        "|---|---|---|---|",
    ]
    wrong = [t for t in done if not t.correct]
    for th in THRESHOLDS:
        acc = [t for t in done if t.accepted(th)]
        silent = [t for t in acc if not t.correct]
        caught = [t for t in wrong if not t.accepted(th)]
        mark = " **(configured)**" if abs(th - chosen) < 1e-9 else ""
        lines.append(
            f"| {th}{mark} | {len(acc)}/{len(done)} ({len(acc) / max(len(done), 1):.0%}) "
            f"| {len(silent)} ({len(silent) / max(len(acc), 1):.1%} of accepted) "
            f"| {len(caught)}/{len(wrong)} |"
        )
    lines += ["", "## Every document", "",
              "| Doc | Kind | Correct | Gate | Score | Wrong fields | Gate issues |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    for t in trials:
        verdict = "failed" if not t.extracted else ("accept" if t.accepted(chosen) else "review")
        lines.append(
            f"| {t.doc} | {t.kind} | {'yes' if t.correct else '**no**'} | {verdict} "
            f"| {t.score:.2f} | {', '.join(t.wrong_fields) or '-'} "
            f"| {', '.join(sorted(set(t.issues))) or '-'} |"
        )
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
