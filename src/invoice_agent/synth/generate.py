"""Generate the synthetic evaluation dataset: PDFs/PNGs + truth JSON + manifest.

Output layout (in `out_dir`):
    inv-001-classic-IN.pdf             digital invoice (has a text layer)
    inv-001-classic-IN.truth.json      exact expected extraction + metadata
    inv-001-classic-IN-scan.pdf|png    degraded copy of the same invoice (no text layer)
    inv-001-classic-IN-scan.truth.json
    manifest.json                      one entry per document (no invoice payload)

A scanned copy always shares its source's dev/test split, so the test set never
contains a scan of an invoice we tuned on.
"""

import json
import random
from pathlib import Path
from typing import Any

from faker import Faker

from invoice_agent.synth.build import PROFILES, Country, InvoiceSpec, build_invoice
from invoice_agent.synth.render import TEMPLATES, PdfRenderer, render_html
from invoice_agent.synth.scan import page_count, scan_pdf

# Which regions each template is realistic for.
TEMPLATE_COUNTRIES: dict[str, tuple[Country, ...]] = {
    "classic": ("IN", "US", "UK"),
    "gst_india": ("IN",),
    "modern": ("US", "EU", "UK", "IN"),
    "minimal": ("IN", "US"),
    "euro": ("EU", "UK"),
}


def plan_documents(
    n_digital: int, n_scanned: int, *, seed: int, test_fraction: float
) -> list[dict[str, Any]]:
    """Decide every document's template, region, size, split and scan format up front.

    Templates rotate so all five are equally represented; regions, multi-page
    invoices (~15%), the split and which invoices get scanned are drawn from `seed`.

    Args:
        n_digital: Number of digital invoices.
        n_scanned: How many of them also get a scanned copy (<= n_digital).
        seed: Master seed; the same seed always gives the same plan.
        test_fraction: Share of digital invoices held out as the test set.

    Returns:
        One dict per digital invoice with keys: index, id, template, country,
        multi_page, split, scan (None, 'pdf' or 'png').

    Raises:
        ValueError: If n_scanned > n_digital.
    """
    if n_scanned > n_digital:
        raise ValueError("cannot scan more invoices than are generated")
    rng = random.Random(seed)
    plans = []
    for i in range(n_digital):
        template = TEMPLATES[i % len(TEMPLATES)]
        country = rng.choice(TEMPLATE_COUNTRIES[template])
        plans.append(
            {
                "index": i,
                "id": f"inv-{i + 1:03d}-{template}-{country}",
                "template": template,
                "country": country,
                "multi_page": rng.random() < 0.15,
                "split": "dev",
                "scan": None,
            }
        )
    for p in rng.sample(plans, round(n_digital * test_fraction)):
        p["split"] = "test"
    for p in rng.sample(plans, n_scanned):
        # PNG only makes sense for single-page invoices.
        p["scan"] = "png" if not p["multi_page"] and rng.random() < 0.25 else "pdf"
    return plans


def _truth_record(
    doc_id: str, file: str, kind: str, plan: dict[str, Any], spec: InvoiceSpec, pages: int
) -> dict[str, Any]:
    """Assemble the ground-truth JSON for one document.

    Args:
        doc_id: Unique document id.
        file: File name of the document inside the dataset folder.
        kind: 'digital' or 'scanned'.
        plan: The document's plan from `plan_documents`.
        spec: The generated invoice.
        pages: Page count.

    Returns:
        A JSON-serializable dict: id, file, kind, split, template, country, pages,
        edge_cases and the expected `invoice` (normalized, money as strings).
    """
    return {
        "id": doc_id,
        "file": file,
        "kind": kind,
        "split": plan["split"],
        "template": plan["template"],
        "country": plan["country"],
        "pages": pages,
        "edge_cases": spec.edge_cases,
        "invoice": spec.truth.model_dump(mode="json"),
    }


def generate_dataset(
    out_dir: Path,
    *,
    n_digital: int = 30,
    n_scanned: int = 20,
    seed: int = 42,
    test_fraction: float = 0.4,
    channel: str | None = None,
) -> list[dict[str, Any]]:
    """Generate the full dataset and write it to disk.

    Args:
        out_dir: Destination folder (created if missing; existing files overwritten).
        n_digital: Number of digital invoices.
        n_scanned: How many digital invoices also get a degraded scanned copy.
        seed: Master seed; identical seeds give identical invoices and truth.
        test_fraction: Share of invoices held out for final evaluation.
        channel: Browser for PDF rendering: None = Playwright's Chromium,
            'msedge'/'chrome' = an installed browser.

    Returns:
        The manifest: one entry per document (truth metadata without the invoice).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    with PdfRenderer(channel=channel) as renderer:
        for plan in plan_documents(n_digital, n_scanned, seed=seed, test_fraction=test_fraction):
            rng = random.Random(seed * 1000 + plan["index"])
            fake = Faker(PROFILES[plan["country"]].faker_locale)
            fake.seed_instance(seed * 1000 + plan["index"])
            spec = build_invoice(rng, fake, country=plan["country"], multi_page=plan["multi_page"])

            pdf_path = out_dir / f"{plan['id']}.pdf"
            renderer.to_pdf(render_html(plan["template"], spec.view), pdf_path)
            docs = [(plan["id"], pdf_path.name, "digital", page_count(pdf_path))]

            if plan["scan"]:
                scan_id = f"{plan['id']}-scan"
                scan_path = out_dir / f"{scan_id}.{plan['scan']}"
                pages = scan_pdf(pdf_path, scan_path, rng)
                docs.append((scan_id, scan_path.name, "scanned", pages))

            for doc_id, file, kind, pages in docs:
                record = _truth_record(doc_id, file, kind, plan, spec, pages)
                (out_dir / f"{doc_id}.truth.json").write_text(
                    json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                manifest.append({k: v for k, v in record.items() if k != "invoice"})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
