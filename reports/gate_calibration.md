# Validation gate: calibration on real extractions

Fast path on 37 dev documents (37 extracted; the rest failed at the provider). 'Correct' = every header field and every line item right.

## Threshold sweep

| Threshold | Auto-accepted | Silent errors (accepted but wrong) | Wrong caught |
|---|---|---|---|
| 0.5 | 35/37 (95%) | 0 (0.0% of accepted) | 2/2 |
| 0.6 | 35/37 (95%) | 0 (0.0% of accepted) | 2/2 |
| 0.7 | 35/37 (95%) | 0 (0.0% of accepted) | 2/2 |
| 0.8 **(configured)** | 35/37 (95%) | 0 (0.0% of accepted) | 2/2 |
| 0.9 | 35/37 (95%) | 0 (0.0% of accepted) | 2/2 |
| 0.95 | 35/37 (95%) | 0 (0.0% of accepted) | 2/2 |
| 0.99 | 33/37 (89%) | 0 (0.0% of accepted) | 2/2 |

## Every document

| Doc | Kind | Correct | Gate | Score | Wrong fields | Gate issues |
|---|---|---|---|---|---|---|
| inv-001-classic-UK | digital | yes | accept | 1.00 | - | - |
| inv-002-gst_india-IN | digital | yes | accept | 1.00 | - | - |
| inv-002-gst_india-IN-scan | scanned | yes | accept | 0.99 | - | - |
| inv-004-minimal-IN | digital | yes | accept | 1.00 | - | - |
| inv-006-dense-US | digital | yes | accept | 1.00 | - | - |
| inv-008-gst_india-IN | digital | yes | accept | 1.00 | - | - |
| inv-008-gst_india-IN-scan | scanned | yes | accept | 1.00 | - | - |
| inv-009-modern-IN | digital | yes | accept | 1.00 | - | - |
| inv-009-modern-IN-scan | scanned | yes | accept | 0.99 | - | - |
| inv-010-minimal-US | digital | yes | accept | 1.00 | - | - |
| inv-010-minimal-US-scan | scanned | yes | accept | 0.99 | - | - |
| inv-012-dense-US | digital | yes | accept | 1.00 | - | - |
| inv-012-dense-US-scan | scanned | yes | accept | 0.99 | - | - |
| inv-014-gst_india-IN | digital | yes | accept | 1.00 | - | - |
| inv-014-gst_india-IN-scan | scanned | yes | accept | 0.99 | - | - |
| inv-015-modern-UK | digital | yes | accept | 1.00 | - | - |
| inv-017-euro-UK | digital | yes | accept | 1.00 | - | - |
| inv-017-euro-UK-scan | scanned | yes | accept | 0.99 | - | - |
| inv-020-gst_india-IN | digital | yes | accept | 1.00 | - | - |
| inv-022-minimal-US | digital | yes | accept | 1.00 | - | - |
| inv-024-dense-US | digital | yes | accept | 1.00 | - | - |
| inv-024-dense-US-scan | scanned | yes | accept | 0.98 | - | - |
| inv-025-classic-IN | digital | yes | accept | 1.00 | - | - |
| inv-025-classic-IN-scan | scanned | yes | accept | 1.00 | - | - |
| inv-027-modern-IN | digital | yes | accept | 1.00 | - | - |
| inv-027-modern-IN-scan | scanned | yes | accept | 0.99 | - | - |
| inv-028-minimal-US | digital | yes | accept | 1.00 | - | - |
| inv-029-euro-UK | digital | yes | accept | 1.00 | - | - |
| inv-029-euro-UK-scan | scanned | yes | accept | 0.99 | - | - |
| inv-032-gst_india-IN | digital | yes | accept | 1.00 | - | - |
| inv-032-gst_india-IN-scan | scanned | yes | accept | 1.00 | - | - |
| inv-033-modern-EU | digital | yes | accept | 1.00 | - | - |
| inv-033-modern-EU-scan | scanned | yes | accept | 0.99 | - | - |
| inv-034-minimal-US | digital | **no** | review | 1.00 | currency | required.missing |
| inv-034-minimal-US-scan | scanned | yes | accept | 1.00 | - | - |
| inv-036-dense-IN | digital | yes | accept | 1.00 | - | - |
| inv-036-dense-IN-scan | scanned | **no** | review | 0.99 | vendor.tax_id | format.gstin_checksum |
