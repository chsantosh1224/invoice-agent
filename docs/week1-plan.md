# Week 1 Plan: Invoice Extraction

## Context
This is the flagship portfolio project for AI Automation / AI Engineer roles: an invoice processing system that extracts invoices, matches them against POs in a mock ERP, and resolves exceptions with a human in the loop. Week 1 covers **extraction**: turning any invoice (digital PDF, scanned PDF or image) into validated, structured JSON with per-field confidence that is reproducible and auditable.

### Design principles (these are also the interview story)
1. **Workflow where the steps are known, agent where they aren't.** Clean invoices go through a deterministic fast-path workflow (one LLM call). Only hard invoices go to an **extraction agent** that decides how to read the document (zoom, re-OCR, cross-page tables, self-validation). A router chooses between them.
2. **Keep the layout.** Never send plain OCR text. Ingestion produces layout-aware Markdown (tables, reading order, key-value proximity) plus word bounding boxes, and the model also gets the page image when it supports vision.
3. **Probabilistic step, deterministic control.** The LLM is the only non-deterministic part. Code verifies its output (arithmetic, schema, values grounded in the OCR text); determinism controls reduce variance; low confidence goes to a human.
4. **Models chosen by measurement, set per role.** `EXTRACTOR_MODEL`, `AGENT_MODEL` and `JUDGE_MODEL` are configured via OpenRouter, and a bake-off picks them.
5. **Free models by default.** The defaults are `google/gemma-4-31b-it:free` (extractor), `nvidia/nemotron-3-super-120b-a12b:free` (agent) and `nvidia/nemotron-3-ultra-550b-a55b:free` (judge). Every call goes through `structured.invoke_structured`: the best output mode the model supports, then Pydantic validation, then a repair loop. Responses are cached on disk to save free-tier quota. `ALLOW_PROMPT_LOGGING=true` is for synthetic data only. Switching a role to a paid model is a one-line change, and the README will report the accuracy difference.

### Architecture
```
file ─► ingest (Docling: markdown + tables + bboxes + page images)
          │
          ▼
     fast-path extract (EXTRACTOR_MODEL, image + markdown, JSON schema)
          │
          ▼
     validate (schema · line math · totals · grounding in OCR text) ─► score
          │
   pass & high confidence? ──yes──► result (cached by file hash)
          │ no
          ▼
     extraction agent (AGENT_MODEL, tool loop, max N steps)
          │
          ▼
     validate ─► pass ─► result
               └ fail ─► needs_human_review (with agent notes)
```

## Environment
- The repo is `C:\Users\lavan\Documents\Invoice Agent`; the existing `.venv` is Python 3.14. **Recreate it with Python 3.12 via `uv`**, because Docling, torch and onnxruntime wheels lag behind on 3.14.
- Dependencies: `langgraph langchain-openai pydantic pydantic-settings docling pdfplumber pypdfium2 pillow fastapi uvicorn python-multipart mcp structlog tenacity typer faker jinja2 playwright`. Dev: `pytest pytest-asyncio ruff mypy`.
- Fallback if Docling is too heavy or slow on your machine: `rapidocr-onnxruntime` plus our own reconstruction of rows and columns from the bounding boxes.

## Project layout
```
pyproject.toml · .env.example · .github/workflows/ci.yml · README.md
src/invoice_agent/
  config.py              # pydantic-settings; per-role models, OpenRouter provider pinning
  schemas.py             # Invoice, Party, LineItem, ExtractionResult, ValidationIssue
  llm.py                 # OpenRouter ChatOpenAI factory; structured output + JSON fallback + retry
  normalize.py           # dates → ISO, currency symbols → ISO code, "1,23,456.00" → Decimal
  ingest/
    detect.py            # pdf w/ text layer | scanned pdf | image
    docling_ingest.py    # → Document(pages[markdown, tables, words+bbox, image])
    zoom.py              # crop bbox at high DPI + re-OCR (used by agent)
  extraction/
    prompts.py           # versioned fast-path prompts (PROMPT_VERSION logged)
    validate.py          # THE gate, shared by fast path, agent, eval: schema, math, grounding, label proximity, format
    fast_path.py         # LangGraph workflow: ingest → extract → validate → score
    router.py            # fast path first, escalate to agent, else human review
    cache.py             # sha256(file) + model + prompt_version → result (SQLite)
    agent/               # the agent = compiled graph (decide + tools + state)
      __init__.py        # build_extraction_agent()
      state.py           # AgentState (small + serializable: doc_id, not the Document)
      nodes.py           # decide, should_continue, finalize
      tools.py           # READ_ONLY_TOOLS + submit_invoice (also imported by mcp_server)
      prompts.py
  cli.py · api.py · mcp_server.py
data/synth/generate.py · data/synth/templates/ · data/samples/
tests/unit/ · tests/eval/
reports/model_selection.md · reports/extraction_baseline.md
```

## Day-by-day (5 hrs/day)

### Day 1: Foundation and model check
- `uv init`, Python 3.12, deps, `git init`, public GitHub repo, CI (ruff + mypy + unit tests).
- `config.py`, `llm.py`: OpenRouter via `ChatOpenAI(base_url="https://openrouter.ai/api/v1")`. Use temperature 0, `seed`, and provider pinning (`provider.order`, `allow_fallbacks: false`), and prefer zero-data-retention providers.
- `schemas.py`: `Invoice` (invoice_number, invoice_date, due_date, vendor/buyer `Party` with name/address/tax_id, po_number, currency, line_items[desc, qty, unit_price, amount], subtotal, tax, total, payment_terms, doc_type invoice/credit_note). Money as `Decimal`.
- **Capability check script:** for each candidate model (Nemotron, a Gemini Flash–class model, Claude Haiku 5.5, Claude Sonnet 5.5, a Qwen VL model), confirm that it accepts images, supports `json_schema` and makes tool calls. Record the exact slugs.

### Day 2: Synthetic data
- `generate.py`: Faker + 5 Jinja2 HTML templates (different layouts, two-column headers, tables that continue onto page 2, INR/USD/EUR, DD/MM vs MM/DD dates, GSTIN/VAT IDs), rendered to PDF with Playwright, plus `*.truth.json`.
- Scanned versions: rasterize, then rotate, blur, add noise and JPEG artifacts with Pillow.
- Edge cases from your finance background: missing PO, tax-inclusive totals, credit notes, discounts, freight lines, multi-page invoices.
- Target: about 50 invoices (30 digital, 20 scanned). Split them into a **dev** set and a held-out **test** set.

### Day 3: Ingestion and model bake-off
- `docling_ingest.py`: per-page Markdown with tables, word bounding boxes, page images. `detect.py` routes digital vs scanned.
- `zoom.py`: crop a bounding box, re-render at 300+ DPI, re-OCR.
- Unit tests: tables survive as Markdown, and two-column headers stay separated.
- **Bake-off:** 10 dev invoices × candidate models × 3 runs, in single-call extraction mode. Measure field accuracy, line-item F1, JSON validity rate, consistency across runs, cost/invoice and p50 latency. Pick `EXTRACTOR_MODEL` and write up the result in `reports/model_selection.md`.

### Day 4: Fast-path workflow and validation
- `fast_path.py` (LangGraph): ingest → extract (image + Markdown → schema) → validate → score.
- `validate.py`: Pydantic schema; line amount = qty × unit_price (with a tolerance); subtotal + tax − discount = total; due_date ≥ invoice_date; **grounding** (every extracted value must appear in the OCR text after normalization).
- **Label proximity** (using bounding boxes): each value must sit near its label, which catches swapped fields such as invoice date vs due date.
- `score`: per-field confidence from grounding, label proximity, arithmetic consistency and OCR confidence. **The LLM's self-reported confidence is never used.**
- "Clean" = all hard checks pass AND score ≥ threshold. The threshold is **calibrated on the dev set** so the *silent error rate* (auto-accepted but wrong) is ≤1%.
- `normalize.py`, `cache.py`. Use tenacity retries with timeouts, and structlog JSON logs (run_id, model, provider, prompt_version, tokens, cost).

### Day 5: Extraction agent and router
- `agent/` package: a custom StateGraph with `START → decide → should_continue → tools → decide … → finalize → END`. One tool call per step. On the final step, `tool_choice` forces a submit. `finalize` re-runs `validate.py`, so the agent never certifies itself. Step limit 8; read-only tools:
  `get_page_overview(page)`, `get_page_image(page)`, `zoom_region(page, bbox)`, `get_tables(page)`, `search_text(pattern)`, `validate_draft(invoice)`, `submit_invoice(invoice, confidence, notes)`.
- The agent's notes are saved as an audit trail (for example, "total corrected via zoom re-OCR").
- `router.py`: if the fast path passes with high confidence, done. Otherwise run the agent. If that still fails, mark `needs_human_review`.
- Pick `AGENT_MODEL` from the Day 1 tool-calling results. Test Nemotron here, with a fallback model ready.

### Day 6: Interfaces and evaluation
- `cli.py`: `invoice-agent extract <file> [--mode fast|agent|router] --json`.
- `mcp_server.py` (FastMCP): `extract_invoice(path)` plus the agent's document tools, so Claude Desktop can drive extraction too.
- `api.py`: `POST /extract` (size/type limits) and `GET /health`.
- `tests/eval/` (marked `pytest -m eval`, kept out of CI): on the **test** set, compare **fast-only vs agent-only vs router**, digital vs scanned. Measure field accuracy, line-item F1, consistency (3 runs), escalation rate, cost/invoice and p50/p95 latency. Write the results to `reports/extraction_baseline.md`.

### Day 7: Iterate and publish
- Fix the 2–3 worst fields on the **dev** set only (prompts, few-shot examples, table handling), then re-run on test once.
- README: architecture diagram (Mermaid), quickstart, results tables, design decisions (workflow vs agent, layout preservation, determinism controls, model choice by role, data retention).
- A short LinkedIn/X post with the numbers.

**If you fall behind:** move the agent and router (Day 5) to the start of Week 2 and keep the fast path, eval and README in Week 1. The Week 2 exception-resolution agent then moves back by a day or two.

## Week 1 done means
- `invoice-agent extract <file>` returns a validated `Invoice` with per-field confidence for digital and scanned input.
- The router beats fast-only on scanned accuracy at an acceptable cost. Targets: ≥95% field accuracy on digital and ≥85% on scanned; consistency ≥98%.
- MCP tools work from an MCP client; the API runs locally; CI is green; the repo is public with a README and both reports.

## Verification
- `uv run pytest tests/unit`
- `uv run pytest -m eval`, then check `reports/extraction_baseline.md`
- `uv run invoice-agent extract data/samples/<scanned>.pdf --mode router --json`, then check the agent notes in the logs
- `npx @modelcontextprotocol/inspector uv run python -m invoice_agent.mcp_server`
- `uv run uvicorn invoice_agent.api:app`, then `curl -F file=@sample.pdf localhost:8000/extract`

## Looking ahead
- **Week 2:** mock ERP (SQLite + FastAPI), 3-way match rules, and an **exception-resolution agent** (MCP tools: get_po, search_pos, get_vendor, get_goods_receipts, get_tolerance_policy, escalate_to_human) with LangGraph `interrupt()` for human-in-the-loop.
- **Week 3:** RAGAS + Playwright eval suite, judged by a different model family.
- **Week 4:** deploy to a subdomain of aicruit.in, with monitoring and security.

**Daily rhythm:** 5 hrs project · 1 hr NeetCode 150 · job search starts week 2.
