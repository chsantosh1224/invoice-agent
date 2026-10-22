"""Test the validation gate without any LLM, by injecting realistic extraction errors.

Start from the truth (a perfect extraction), apply one mutation that mimics an error LLMs
really make (several were seen in our own runs), and check that the gate rejects it.
The unmutated truth must pass - a gate that rejects correct invoices is useless too.
"""

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from invoice_agent.extraction.validate import GateReport, validate
from invoice_agent.ingest.document import Document
from invoice_agent.ingest.light import LightIngestor
from invoice_agent.schemas import Invoice

Mutation = Callable[[Invoice, random.Random], Invoice | None]


def _swap_dates(inv: Invoice, rng: random.Random) -> Invoice | None:
    """Invoice date and due date swapped (seen: models mixing up two dates)."""
    if not (inv.invoice_date and inv.due_date):
        return None
    inv.invoice_date, inv.due_date = inv.due_date, inv.invoice_date
    return inv


def _due_day_month_swap(inv: Invoice, rng: random.Random) -> Invoice | None:
    """Due date read MM/DD instead of DD/MM (seen: dots-3 on '03/05/2026' in the smoke test)."""
    d = inv.due_date
    if not d or d.day > 12 or d.day == d.month:
        return None
    inv.due_date = date(d.year, d.day, d.month)
    return inv


def _total_misread(inv: Invoice, rng: random.Random) -> Invoice | None:
    """One digit of the total misread (e.g. 8 read as 3)."""
    if inv.total is None:
        return None
    inv.total += Decimal(rng.choice([5, 50, 500]))
    return inv


def _line_amount_misread(inv: Invoice, rng: random.Random) -> Invoice | None:
    """One digit of a line amount misread."""
    if not inv.line_items or inv.line_items[0].amount is None:
        return None
    inv.line_items[0].amount += Decimal(rng.choice([1, 10, 100]))
    return inv


def _drop_line(inv: Invoice, rng: random.Random) -> Invoice | None:
    """A line item skipped (e.g. the last row before a page break)."""
    if len(inv.line_items) < 2:
        return None
    inv.line_items.pop()
    return inv


def _invent_discount(inv: Invoice, rng: random.Random) -> Invoice | None:
    """A discount returned that is not on the invoice (seen: 'discount: 0.00' and worse)."""
    if inv.discount is not None or inv.subtotal is None:
        return None
    inv.discount = (inv.subtotal * Decimal("0.05")).quantize(Decimal("0.01"))
    return inv


def _copy_vendor_tax_id(inv: Invoice, rng: random.Random) -> Invoice | None:
    """Vendor's tax ID copied onto the buyer (seen: dots-3 in the smoke test)."""
    if not inv.vendor.tax_id or inv.buyer.tax_id == inv.vendor.tax_id:
        return None
    inv.buyer.tax_id = inv.vendor.tax_id
    return inv


def _invoice_number_ocr(inv: Invoice, rng: random.Random) -> Invoice | None:
    """An OCR-style character confusion in the invoice number (0->O, 1->7, 5->S...)."""
    if not inv.invoice_number:
        return None
    swaps = {"0": "O", "1": "7", "5": "S", "8": "B", "2": "Z"}
    positions = [i for i, c in enumerate(inv.invoice_number) if c in swaps]
    if not positions:
        return None
    i = rng.choice(positions)
    s = inv.invoice_number
    inv.invoice_number = s[:i] + swaps[s[i]] + s[i + 1 :]
    return inv


def _hallucinate_po(inv: Invoice, rng: random.Random) -> Invoice | None:
    """A PO number returned for an invoice that has none."""
    if inv.po_number is not None:
        return None
    inv.po_number = f"PO-{rng.randint(1000, 9999)}"
    return inv


def _vendor_is_buyer(inv: Invoice, rng: random.Random) -> Invoice | None:
    """Vendor and buyer confused (Bill-from read from the Bill-to block)."""
    if not inv.buyer.name or inv.buyer.name == inv.vendor.name:
        return None
    inv.vendor.name = inv.buyer.name
    return inv


def _swap_parties(inv: Invoice, rng: random.Random) -> Invoice | None:
    """Vendor and buyer fully swapped (names and tax IDs) - the harder variant."""
    if not inv.buyer.name or inv.buyer.name == inv.vendor.name:
        return None
    inv.vendor, inv.buyer = inv.buyer, inv.vendor
    return inv


MUTATIONS: dict[str, Mutation] = {
    "swap_dates": _swap_dates,
    "due_day_month_swap": _due_day_month_swap,
    "total_misread": _total_misread,
    "line_amount_misread": _line_amount_misread,
    "drop_line_item": _drop_line,
    "invent_discount": _invent_discount,
    "copy_vendor_tax_id": _copy_vendor_tax_id,
    "invoice_number_ocr": _invoice_number_ocr,
    "hallucinate_po": _hallucinate_po,
    "vendor_is_buyer": _vendor_is_buyer,
    "swap_parties": _swap_parties,
}


@dataclass
class GateTrial:
    """One gate decision in the mutation test.

    Attributes:
        doc: Document id.
        kind: 'digital' or 'scanned'.
        mutation: Mutation name, or 'clean' for the unmodified truth.
        report: The gate's report.
    """

    doc: str
    kind: str
    mutation: str
    report: GateReport


def run_mutation_test(data_dir: Path, *, split: str = "dev", seed: int = 7) -> list[GateTrial]:
    """Run the gate on every truth invoice of a split, clean and with each mutation.

    Args:
        data_dir: Dataset folder with manifest.json.
        split: 'dev', 'test' or 'all'.
        seed: Seed for the random parts of mutations.

    Returns:
        One GateTrial per (document, clean-or-applicable-mutation).
    """
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    engine = LightIngestor()
    rng = random.Random(seed)
    trials = []
    for meta in (d for d in manifest if split == "all" or d["split"] == split):
        truth: dict[str, Any] = json.loads(
            (data_dir / f"{meta['id']}.truth.json").read_text(encoding="utf-8")
        )["invoice"]
        doc: Document = engine.ingest(data_dir / meta["file"])
        trials.append(GateTrial(meta["id"], meta["kind"], "clean",
                                validate(Invoice.model_validate(truth), doc)))  # fmt: skip
        for name, mutate in MUTATIONS.items():
            mutated = mutate(Invoice.model_validate(truth), rng)
            if mutated is not None:
                trials.append(GateTrial(meta["id"], meta["kind"], name, validate(mutated, doc)))
    return trials


def _rate(trials: list[GateTrial], *, want_pass: bool) -> str:
    """Share of trials where the gate gave the desired verdict.

    Args:
        trials: Gate trials of one group (e.g. one mutation on digital documents).
        want_pass: True if the desired verdict is "passed" (clean truth), False if
            "rejected" (an injected error).

    Returns:
        e.g. '95% (21/22)', or '-' for an empty group.
    """
    if not trials:
        return "-"
    good = sum(t.report.passed == want_pass for t in trials)
    return f"{good / len(trials):.0%} ({good}/{len(trials)})"


def write_report(trials: list[GateTrial], path: Path) -> str:
    """Write the mutation-test results as Markdown.

    Args:
        trials: Results of `run_mutation_test`.
        path: Report file.

    Returns:
        The Markdown text.
    """
    lines = [
        "# Validation gate: mutation test",
        "",
        "Each truth invoice (a perfect extraction) is run through the gate unmodified, then "
        "once per injected error. No LLM involved.",
        "",
        "| Case | Digital | Scanned | All |",
        "|---|---|---|---|",
    ]

    for case in ["clean", *MUTATIONS]:
        group = [t for t in trials if t.mutation == case]
        want_pass = case == "clean"
        label = "clean truth: **passed**" if want_pass else f"{case}: **caught**"
        lines.append(
            f"| {label} | {_rate([t for t in group if t.kind == 'digital'], want_pass=want_pass)} "
            f"| {_rate([t for t in group if t.kind == 'scanned'], want_pass=want_pass)} "
            f"| {_rate(group, want_pass=want_pass)} |"
        )
    mutated = [t for t in trials if t.mutation != "clean"]
    caught = sum(not t.report.passed for t in mutated)
    lines += ["", f"**Overall catch rate: {caught / len(mutated):.1%}** ({caught}/{len(mutated)})"]

    rejected = [t for t in trials if t.mutation == "clean" and not t.report.passed]
    if rejected:
        lines += ["", "## Correct invoices the gate rejected (false alarms)", ""]
        for t in rejected:
            why = "; ".join(f"{i.code} ({i.field})" for i in t.report.errors) or (
                f"score {t.report.score:.2f}"
            )
            lines.append(f"- `{t.doc}`: {why}")
    missed = [t for t in mutated if t.report.passed]
    if missed:
        lines += ["", "## Injected errors the gate missed", ""]
        lines += [f"- `{t.doc}`: {t.mutation}" for t in missed]
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
