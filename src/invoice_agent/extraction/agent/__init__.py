"""The extraction agent: an LLM that investigates invoices the fast path could not verify.

The agent is the compiled graph as a whole (decide -> tools -> decide ... -> finalize);
`decide` is only its reasoning step, `tools.py` holds its read-only actions.
"""
