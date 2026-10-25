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

## D4 (Day 3, resolved Day 5): Agent model = Nemotron 3 Super (free)

The worry from the bake-off was that text-only Nemotron Super *transcribes* invoices poorly (68.7%). But the agent's job is different: it starts from the vision model's draft plus the gate's specific problems, and has to investigate with tools. Tested on the two invoices the fast path got wrong in calibration:

| Invoice (fast-path error) | Nemotron Super | Nemotron Omni |
|---|---|---|
| inv-034-minimal-US (currency missing) | fixed in 2 turns, 20 s, accepted | fixed in 2 turns, 73 s, accepted |
| inv-036-dense-IN-scan (GSTIN `0` read for `O`, checksum fails) | fixed in 8 turns, 300 s, accepted; corrected GSTIN equals the truth | provider capacity error (free tier); fast-path verdict kept |

**Decision:** `AGENT_MODEL=nvidia/nemotron-3-super-120b-a12b:free`. It is reliable at tool calling, faster, and was available when Omni was not. Note that it sends `notes` as a string rather than a list; the graph now accepts both.

**Risk:** this is two invoices. Day 6 measures the router end to end on the held-out test split.

## D5 (Day 4): Validation gate design and threshold 0.8

**Checks** (`extraction/validate.py`): required fields; arithmetic (line math ±0.01, lines = subtotal, subtotal − discount + tax = total); grounding of every value in the document text (an error on digital PDFs, a confidence-lowering warning on scans because OCR may be the side that's wrong); label proximity (a same-row label beats one above); GSTIN checksum; due date not before invoice date; vendor ≠ buyer; vendor not found only under a Bill-To label. The LLM's self-reported confidence is never used.

**Evidence:**
- [Mutation test](../reports/gate_mutation.md) (no LLM, 37 dev truths, 11 error types, 333 injected errors): 0 false alarms; 95% caught. 10 of 11 types caught 100%. A full vendor/buyer swap is caught 57%.
- [Live calibration](../reports/gate_calibration.md) (real dots-3 output, 37 dev invoices): 35 fully correct, all auto-accepted; 2 wrong (missing currency, one misread GSTIN character), both caught. **0 silent errors.** Thresholds from 0.5 to 0.95 behave the same; 0.99 would reject correct scans. 0.8 keeps that margin.

**Known gap:** a full party swap on layouts with no buyer label. Fix in Week 2: match the vendor against the ERP vendor master (name + tax ID).

## D6 (Week 1): Currency from printed amounts; a bare "$" is never assumed silently

**Trigger:** a real third-party sample (Sliced Invoices, WordPress PDF plugin) printed `$93.50` on its amounts but no separate currency line. The model returned `currency: null`, and the gate reported "currency is missing". The invoice also mixes US and Australian signals (an Arizona vendor; a Melbourne buyer; ANZ bank details with a BSB number), so `$` may well mean AUD.

**Decision:**
1. If the currency field is empty, it is taken from the symbols printed on the amounts (`currency.inferred`). Amounts in different currencies are an error (`currency.conflicting_symbols`).
2. A bare `$` (or `¥`) resolves to its most common currency (USD/JPY) but is reported as `currency.ambiguous_symbol`. The gate caps currency confidence at 0.6, below the 0.8 threshold, so a person confirms it. Prefixed dollars (`US$`, `A$`, `C$`, `NZ$`, `S$`, `HK$`), ISO codes and ₹/€/£ are unambiguous.

**Impact** ([re-calibration](../reports/gate_calibration.md), 37 dev invoices, cached answers): fully correct 35 → 36; auto-accepted 35 → 34 (92%), because two correct US invoices printing only `$` now go to review; **silent errors stay 0**.

**Known gap:** when the model writes `USD` itself for a document that prints only `$`, the assumption is not detected, because the currency field is not grounded against the page. Possible fixes: ground the currency field, or tighten the prompt. Later, the ERP vendor master or PO currency can settle `$` automatically (Week 2).

## D7 (Week 1): extract-v2 - totals as labelled rows, more document types, line columns

**Why:** a real Magento sample showed that fixed totals fields can't represent shipping (every run raised `math.total`), and that dates can carry a time. A fixed field per charge does not scale: handling, round-off, surcharges, advances, tax-inclusive VAT…

**Design:**
- The model copies every totals row as `summary_lines` (`label`, `amount`, `kind`). `kind` is one of subtotal, discount, tax, tax_included, charge, deduction, adjustment, total or other.
- Code (`summary.py`) applies the kinds and checks them with arithmetic. An `other` row, or a single wrong sign, is resolved only when exactly one reading adds up; it is then flagged and its confidence capped, so the invoice goes to review. A row that cannot be resolved is an error.
- Document types: 8 instead of 2. Anything that is not an invoice or credit note fails with `document.not_payable`.
- Line items can carry their own discount and tax columns, and the line arithmetic accepts every combination the printed columns allow.
- Dates may include a time. `Mon dd, yyyy` is now a recognized format.
- The agent has a fallback model (`AGENT_FALLBACK_MODEL` = Nemotron 3 Ultra) after NVIDIA's free endpoint for Nemotron Super started returning 404. If neither model runs, the result carries `agent.unavailable` instead of looking like a plain fast-path result.

**Evidence** (same 10 dev invoices and config as D2, dots-3 image + text, 1 run each):

| | extract-v1 | extract-v2 |
|---|---|---|
| Field accuracy | 98.7% | **99.3%** |
| Line items F1 | 100% | 100% |
| Hallucinations / doc | 0.20 | **0.00** |
| Median time | 53 s | 43 s |

Subtotal, discount, tax and total were correct on all 10 invoices, so the `kind` labels look reliable.

**Watch:** one new miss. A PO printed as "CUSTOMER ORDER REF." was returned as null; v1 extracted it. The PO is not a required field, so the gate does not flag the miss. Fix candidates: PO label examples in the prompt, or a `references` list (Week 2). The sample has no shipping, round-off, tax-inclusive, line-discount or non-invoice documents yet, so add those to the synthetic dataset before trusting the new paths.
