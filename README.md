# Invoice Agent

AI invoice processing built on three ideas:

- a **deterministic extraction workflow** that keeps the page layout,
- an **extraction agent** that is used only when the workflow's output fails validation,
- a validation gate that the LLM's output always has to pass.

Live demo: https://invoice.aicruit.in (accounts on request). Deployment:
[docs/deploy.md](docs/deploy.md).

## Quickstart

```bash
uv sync
cp .env.example .env              # add your OPENROUTER_API_KEY
uv run invoice-agent models       # which models can serve which role
uv run invoice-agent smoke        # live strict-JSON extraction of a tiny invoice
uv run pytest                     # unit tests (no LLM calls)
```

## The harness

The model is one component. Everything around it decides whether its output can be
trusted:

```
upload -> daily limit -> read the page (hidden text set aside)
       -> fast path: 1 LLM call -> validation gate --pass--> accepted
                                       |
                                       +-- security flag ------------> human review
                                       +-- fail -> agent (tools, 8 steps)
                                                   -> gate again --pass--> accepted
                                                                 --fail--> human review
human review -> approve (gate re-checks) / reject -> Langfuse scores, eval cases
```

| Concern | How it is handled | Where |
|---|---|---|
| Agent runtime | LangGraph graphs: a fixed fast path, and an agent loop `decide -> tools -> decide` with one router function | `extraction/fast_path.py`, `extraction/agent/graph.py`, `extraction/router.py` |
| Tool dispatch | Read-only tools (`search_text`, `get_page_text`, `reread_region`, `check_draft`) run by LangGraph's `ToolNode`; arguments validated by Pydantic; the document id is injected by the code, never chosen by the model; one tool per turn; `submit_invoice` is handled by its own node | `extraction/agent/tools.py` |
| Context | The document stays in a workspace keyed by `doc_id`; the model pulls what it needs through tools. The first message holds only the failed draft, the gate's issues and the page text. Nudges ("you are out of turns") are added for one call and never stored | `extraction/agent/workspace.py`, `extraction/agent/prompts.py` |
| Memory | Short-term only: the conversation in the graph state (`add_messages`). Nothing persists between invoices, by design: each invoice is judged on its own evidence | `extraction/agent/graph.py` |
| Guardrails | The **validation gate**: arithmetic, tax-ID checksums, every value grounded in the document's text, plausibility checks, a confidence score. The agent's own submission goes through the same gate. Step limit, forced submit on the last turn | `extraction/validate.py` |
| Security | Logins (scrypt hashes, lockout, admin approval); keys only on the server; upload type and size checks; read-only tools; whether model providers may log prompts is a setting (`ALLOW_PROMPT_LOGGING`: on for the free-model demo, off for real invoices). **Prompt injection**: invisible PDF text is kept from the model, instructions aimed at an AI are flagged, flagged documents skip the agent and go to a person. Per-user daily limit (cost) | `auth/`, `ingest/light.py`, `extraction/validate.py`, `api.py` |
| Human approval | Flagged invoices wait in a review queue; an admin corrects and approves (the gate re-checks; overriding needs a written reason) or rejects | `review/`, `web/review.html` |
| Recovery | Invalid JSON is shown back to the model (2 repairs); fallback model per role; client retries with backoff on rate limits; a rejected submission loops back to the agent; provider failures end in review, never in a crash | `structured.py`, `extraction/router.py` |
| Observability | Structured logs with a `run_id`; one Langfuse trace per invoice (every LLM call, tool step, tokens, cost, user, outcome) | `log.py`, `tracing.py` |
| Evaluation | Offline: model bake-off, gate calibration, gate mutation test, ingestion comparison, prompt injection. Online: each review decision becomes Langfuse scores; approved corrections export as new eval cases | `evals/`, `reports/`, `review/export.py` |

Results: [reports/](reports/) (e.g. [prompt injection](reports/prompt_injection.md),
[gate mutation test](reports/gate_mutation.md)).
