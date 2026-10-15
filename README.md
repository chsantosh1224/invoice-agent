# Invoice Agent

AI invoice processing built on three ideas:

- a **deterministic extraction workflow** that keeps the page layout,
- an **extraction agent** that is used only when the workflow's output fails validation,
- a validation gate that the LLM's output always has to pass.

Status: Week 1 (extraction) in progress. See [docs/week1-plan.md](docs/week1-plan.md).

## Quickstart

```bash
uv sync
cp .env.example .env              # add your OPENROUTER_API_KEY
uv run invoice-agent models       # which models can serve which role
uv run invoice-agent smoke        # live strict-JSON extraction of a tiny invoice
uv run pytest                     # unit tests (no LLM calls)
```
