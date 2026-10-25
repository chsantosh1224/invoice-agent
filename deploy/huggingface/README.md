---
title: Invoice Agent
emoji: 🧾
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: AI invoice extraction, verified by code
---

# Invoice Agent

Upload an invoice (PDF, PNG or JPG) and get its fields, line items and totals as structured
data. An LLM reads the document; a deterministic validation gate checks every value
(arithmetic, tax-ID checksums, grounding in the page text) before anything is accepted, and
an extraction agent fixes what fails.

**Demo only:** please upload sample or synthetic invoices. This demo runs on free models
whose providers may log what they receive.

Source code: https://github.com/chsantosh1224/invoice-agent
