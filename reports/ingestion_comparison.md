# Ingestion comparison

Split: **dev** · documents: 37 · no LLM involved.

- **Value recall**: share of truth values (ids, names, dates, amounts) present in the engine's text. A value missing here cannot be extracted from text.
- **Row integrity**: share of line items whose qty, unit price and amount sit on one line. Low = a text-only model must guess which numbers belong together.

## Overall

| Engine | Kind | Docs | Value recall | Row integrity | Sec/page |
|---|---|---|---|---|---|
| docling | digital | 22 | 100.0% | 97.5% | 6.99 |
| docling | scanned | 15 | 96.3% | 89.5% | 13.42 |
| docling | all | 37 | 98.8% | 95.3% | 9.40 |
| docling-default | digital | 22 | 99.8% | 97.5% | 5.06 |
| docling-default | scanned | 15 | 96.3% | 89.5% | 12.17 |
| docling-default | all | 37 | 98.7% | 95.3% | 7.72 |
| light | digital | 22 | 100.0% | 100.0% | 0.13 |
| light | scanned | 15 | 99.7% | 100.0% | 1.98 |
| light | all | 37 | 99.9% | 100.0% | 0.82 |

## Peak memory (process RSS)

| Engine | Peak MB |
|---|---|
| docling | 1230 |
| docling-default | 1788 |
| light | 512 |

## By template (all kinds)

| Template | docling recall | docling rows | docling-default recall | docling-default rows | light recall | light rows |
|---|---|---|---|---|---|---|
| classic | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| dense | 93.2% | 80.0% | 93.2% | 80.0% | 99.2% | 100.0% |
| euro | 98.5% | 88.9% | 98.5% | 88.9% | 100.0% | 100.0% |
| gst_india | 99.6% | 98.6% | 99.1% | 98.6% | 100.0% | 100.0% |
| minimal | 100.0% | 97.2% | 100.0% | 97.2% | 100.0% | 100.0% |
| modern | 100.0% | 95.8% | 100.0% | 95.8% | 100.0% | 100.0% |

## Most-missed fields

- **docling**: line_items (4), vendor.tax_id (3), subtotal (2), discount (1), total (1)
- **docling-default**: line_items (4), subtotal (3), vendor.tax_id (3), discount (1), total (1)
- **light**: vendor.tax_id (1)
