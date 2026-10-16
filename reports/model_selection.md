# Model bake-off

Prompt `extract-v1` · 10 dev invoices · free OpenRouter models.

Decisions drawn from these numbers: [docs/decisions.md](../docs/decisions.md).

- **Acc (ok)**: header-field accuracy on documents that produced valid output.
- **Acc (e2e)**: same, counting failed documents as 0.
- **Items F1**: line items with qty, unit price and amount all correct.
- **Halluc./doc**: values returned where the invoice has none (worst error type).

| Config | OK | Acc (ok) | Acc (e2e) | Items F1 | Desc. ok | Halluc./doc | Repairs | Median s | Failures |
|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b · image+light | 0/10 | 0.0% | 0.0% | 0.0% | 0.0% | 0.00 | 0 | 0.0 | error 10 |
| gemma-4-31b · image | 0/10 | 0.0% | 0.0% | 0.0% | 0.0% | 0.00 | 0 | 0.0 | error 10 |
| dots-3 · image+light | 10/10 | 98.7% | 98.7% | 100.0% | 100.0% | 0.20 | 0 | 53.0 | - |
| dots-3 · image | 10/10 | 96.7% | 96.7% | 100.0% | 100.0% | 0.20 | 0 | 56.7 | - |
| nemotron-omni · image+light | 5/10 | 98.7% | 49.3% | 100.0% | 100.0% | 0.20 | 0 | 67.8 | error 5 |
| nemotron-super · light | 10/10 | 68.7% | 68.7% | 90.0% | 100.0% | 0.10 | 0 | 56.4 | - |
| nemotron-super · docling | 10/10 | 64.0% | 64.0% | 70.0% | 100.0% | 0.20 | 0 | 57.2 | - |

## Field accuracy by config

| Field | gemma-4-31b · image+light | gemma-4-31b · image | dots-3 · image+light | dots-3 · image | nemotron-omni · image+light | nemotron-super · light | nemotron-super · docling |
|---|---|---|---|---|---|---|---|
| buyer.name | - | - | 100.0% | 100.0% | 100.0% | 100.0% | 90.0% |
| buyer.tax_id | - | - | 80.0% | 70.0% | 80.0% | 60.0% | 30.0% |
| currency | - | - | 100.0% | 100.0% | 100.0% | 40.0% | 20.0% |
| discount | - | - | 100.0% | 100.0% | 100.0% | 90.0% | 70.0% |
| document_type | - | - | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| due_date | - | - | 100.0% | 100.0% | 100.0% | 80.0% | 100.0% |
| invoice_date | - | - | 100.0% | 100.0% | 100.0% | 70.0% | 90.0% |
| invoice_number | - | - | 100.0% | 100.0% | 100.0% | 90.0% | 90.0% |
| payment_terms | - | - | 100.0% | 100.0% | 100.0% | 20.0% | 20.0% |
| po_number | - | - | 100.0% | 100.0% | 100.0% | 70.0% | 60.0% |
| subtotal | - | - | 100.0% | 100.0% | 100.0% | 30.0% | 0.0% |
| tax | - | - | 100.0% | 100.0% | 100.0% | 90.0% | 80.0% |
| total | - | - | 100.0% | 100.0% | 100.0% | 50.0% | 60.0% |
| vendor.name | - | - | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| vendor.tax_id | - | - | 100.0% | 80.0% | 100.0% | 40.0% | 50.0% |

## By document kind (Acc ok / Items F1)

| Config | digital | scanned |
|---|---|---|
| gemma-4-31b · image+light | - | - |
| gemma-4-31b · image | - | - |
| dots-3 · image+light | 100.0% / 100.0% | 97.3% / 100.0% |
| dots-3 · image | 100.0% / 100.0% | 93.3% / 100.0% |
| nemotron-omni · image+light | 100.0% / 100.0% | 96.7% / 100.0% |
| nemotron-super · light | 76.0% / 100.0% | 61.3% / 80.0% |
| nemotron-super · docling | 73.3% / 100.0% | 54.7% / 40.0% |

## Failures

- `gemma-4-31b · image+light` on `inv-001-classic-UK`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `gemma-4-31b · image+light` on `inv-012-dense-US-scan`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `gemma-4-31b · image+light` on `inv-017-euro-UK`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `gemma-4-31b · image+light` on `inv-002-gst_india-IN-scan`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `gemma-4-31b · image+light` on `inv-004-minimal-IN`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `gemma-4-31b · image+light` on `inv-009-modern-IN-scan`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `gemma-4-31b · image+light` on `inv-025-classic-IN`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `gemma-4-31b · image+light` on `inv-024-dense-US-scan`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `gemma-4-31b · image+light` on `inv-029-euro-UK`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `gemma-4-31b · image+light` on `inv-008-gst_india-IN-scan`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `gemma-4-31b · image` on `inv-001-classic-UK`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `gemma-4-31b · image` on `inv-012-dense-US-scan`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `gemma-4-31b · image` on `inv-017-euro-UK`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Rate limit exceeded: free-models-per-min. ', 'code': 429, 'metadata': {'headers': {'X-RateLimit-L
- `gemma-4-31b · image` on `inv-002-gst_india-IN-scan`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `gemma-4-31b · image` on `inv-004-minimal-IN`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `gemma-4-31b · image` on `inv-009-modern-IN-scan`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `gemma-4-31b · image` on `inv-025-classic-IN`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `gemma-4-31b · image` on `inv-024-dense-US-scan`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Rate limit exceeded: free-models-per-min. ', 'code': 429, 'metadata': {'headers': {'X-RateLimit-L
- `gemma-4-31b · image` on `inv-029-euro-UK`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `gemma-4-31b · image` on `inv-008-gst_india-IN-scan`: OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is tempora
- `nemotron-omni · image+light` on `inv-012-dense-US-scan`: ValueError: {'message': 'Upstream error from Nvidia: ResourceExhausted: Worker local total request limit reached (16/16)', 'code': 502, 'metadata': {'error_type
- `nemotron-omni · image+light` on `inv-017-euro-UK`: ValueError: {'message': 'Upstream error from Nvidia: ResourceExhausted: Worker local total request limit reached (16/16)', 'code': 502, 'metadata': {'error_type
- `nemotron-omni · image+light` on `inv-009-modern-IN-scan`: ValueError: {'message': 'Upstream error from Nvidia: ResourceExhausted: Worker local total request limit reached (16/16)', 'code': 502, 'metadata': {'error_type
- `nemotron-omni · image+light` on `inv-025-classic-IN`: ValueError: {'message': 'Upstream error from Nvidia: ResourceExhausted: Worker local total request limit reached (16/16)', 'code': 502, 'metadata': {'error_type
- `nemotron-omni · image+light` on `inv-024-dense-US-scan`: ValueError: {'message': 'Upstream error from Nvidia: ResourceExhausted: Worker local total request limit reached (16/16)', 'code': 502, 'metadata': {'error_type
