# Validation gate: mutation test

Each truth invoice (a perfect extraction) is run through the gate unmodified, then once per injected error. No LLM involved.

| Case | Digital | Scanned | All |
|---|---|---|---|
| clean truth: **passed** | 100% (22/22) | 100% (15/15) | 100% (37/37) |
| swap_dates: **caught** | 100% (21/21) | 100% (14/14) | 100% (35/35) |
| due_day_month_swap: **caught** | 100% (3/3) | 100% (3/3) | 100% (6/6) |
| total_misread: **caught** | 100% (22/22) | 100% (15/15) | 100% (37/37) |
| line_amount_misread: **caught** | 100% (22/22) | 100% (15/15) | 100% (37/37) |
| drop_line_item: **caught** | 100% (22/22) | 100% (15/15) | 100% (37/37) |
| invent_discount: **caught** | 100% (13/13) | 100% (8/8) | 100% (21/21) |
| copy_vendor_tax_id: **caught** | 100% (22/22) | 100% (15/15) | 100% (37/37) |
| invoice_number_ocr: **caught** | 100% (21/21) | 100% (14/14) | 100% (35/35) |
| hallucinate_po: **caught** | 100% (8/8) | 100% (6/6) | 100% (14/14) |
| vendor_is_buyer: **caught** | 100% (22/22) | 100% (15/15) | 100% (37/37) |
| swap_parties: **caught** | 73% (16/22) | 33% (5/15) | 57% (21/37) |

**Overall catch rate: 95.2%** (317/333)

## Injected errors the gate missed

- `inv-002-gst_india-IN-scan`: swap_parties
- `inv-006-dense-US`: swap_parties
- `inv-008-gst_india-IN-scan`: swap_parties
- `inv-012-dense-US`: swap_parties
- `inv-012-dense-US-scan`: swap_parties
- `inv-014-gst_india-IN-scan`: swap_parties
- `inv-017-euro-UK`: swap_parties
- `inv-017-euro-UK-scan`: swap_parties
- `inv-024-dense-US`: swap_parties
- `inv-024-dense-US-scan`: swap_parties
- `inv-025-classic-IN-scan`: swap_parties
- `inv-029-euro-UK`: swap_parties
- `inv-029-euro-UK-scan`: swap_parties
- `inv-032-gst_india-IN-scan`: swap_parties
- `inv-036-dense-IN`: swap_parties
- `inv-036-dense-IN-scan`: swap_parties
