# Decision log

Each decision records the evidence behind it, so it can be revisited when the numbers change.

## D1 (Day 3): Lightweight ingestion is the default; Docling stays optional

**Evidence.** Two evaluations, both on the dev split:

- [Ingestion comparison](../reports/ingestion_comparison.md) (37 docs, no LLM): lightweight 99.9% value recall, 0.8 s/page, ~0.5 GB RAM. Docling 98.8%, 9.4 s/page, ~1.2 GB RAM, PyTorch plus 722 MB of models. Docling also scrambled label-above-value headers on the dense template.
- [Bake-off](../reports/model_selection.md), end to end with text-only Nemotron Super: lightweight text gave 68.7% field accuracy and 90% items F1; Docling text gave 64.0% and 70%.

**Known weakness.** Inside wrapped multi-line table cells, lightweight can put description lines in the wrong order. Revisit if real invoices show it hurting accuracy.

## D2 (Day 3): Extractor = dots-3 (free) with page images + lightweight text

**Evidence** ([bake-off](../reports/model_selection.md), 10 dev invoices: 5 digital, 5 scanned, all 6 templates):

| Config | OK | Field acc. | Items F1 |
|---|---|---|---|
| dots-3 · image+light | 10/10 | 98.7% | 100% |
| dots-3 · image | 10/10 | 96.7% | 100% |
| nemotron-omni · image+light | 5/10 (capacity errors) | 98.7% | 100% |
| gemma-4-31b (free) | 0/20 (rate-limited) | – | – |

- Adding the extracted text helps most on scans (97.3% vs 93.3%). It fixed misread digits in vendor tax IDs.
- Fallback: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`.
- **Risks:** dots-3 is a free *preview* and may disappear. It is slow (median 53 s per invoice; it is a reasoning model). The sample is small (10 docs, 1 run). Day 6 measures accuracy and consistency on the held-out test split.

## D3 (Day 3): Remaining error = "Unregistered" copied as buyer GSTIN

Every error made by the top configs was the same: on GST invoices printing `GSTIN: Unregistered`, the model returned "Unregistered" as `buyer.tax_id`. The truth is null.

**Fix (Day 4):** `normalize.py` treats a tax ID with no digits as null, and the validation gate checks the GSTIN checksum.

## D4 (Day 3, open): The agent model needs re-evaluation

Text-only Nemotron Super reads invoices poorly (68.7%). It mostly returns null for totals, currency and terms on the totals block or on page 2. Day 5 will compare vision-capable tool-calling models (nemotron-omni, dots-3) against it on tool-calling reliability before choosing `AGENT_MODEL`.

## D5 (Day 4): Validation gate design and threshold 0.8

**Checks** (`extraction/validate.py`): required fields; arithmetic (line math ±0.01, lines = subtotal, subtotal − discount + tax = total); grounding of every value in the document text (an error on digital PDFs, a confidence-lowering warning on scans because OCR may be the side that's wrong); label proximity (a same-row label beats one above); GSTIN checksum; due date not before invoice date; vendor ≠ buyer; vendor not found only under a Bill-To label. The LLM's self-reported confidence is never used.

**Evidence:**
- [Mutation test](../reports/gate_mutation.md) (no LLM, 37 dev truths, 11 error types, 333 injected errors): 0 false alarms; 95% caught. 10 of 11 types caught 100%. A full vendor/buyer swap is caught 57%.
- [Live calibration](../reports/gate_calibration.md) (real dots-3 output, 37 dev invoices): 35 fully correct, all auto-accepted; 2 wrong (missing currency, one misread GSTIN character), both caught. **0 silent errors.** Thresholds from 0.5 to 0.95 behave the same; 0.99 would reject correct scans. 0.8 keeps that margin.

**Known gap:** a full party swap on layouts with no buyer label. Fix in Week 2: match the vendor against the ERP vendor master (name + tax ID).
