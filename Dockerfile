# Demo image: FastAPI app + lightweight ingestion (pdfplumber, RapidOCR). No Docling, no
# synthetic-data tools. Run on AWS Lightsail behind Caddy (deploy/lightsail, docs/deploy.md);
# works anywhere with `docker run -p 7860:7860 -e OPENROUTER_API_KEY=... <image>`.
FROM python:3.12-slim

# Shared libraries OpenCV (used by RapidOCR) and onnxruntime load at run time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

# Hugging Face runs containers as user 1000; give it a home it can write to.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /home/user/app

# Dependencies first, so code changes don't reinstall them.
COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-default-groups --no-install-project
COPY --chown=user src ./src
RUN uv sync --locked --no-default-groups

# The user database lives here; Compose mounts a volume on it so accounts survive rebuilds.
RUN mkdir -p /home/user/data

# Free OpenRouter endpoints may log prompts: allowed because the demo takes sample data
# only (the page says so). The cache lives in /tmp and is lost on restart, which is fine.
# FORWARDED_ALLOW_IPS: trust Caddy's X-Forwarded-Proto, so the app knows the site is HTTPS
# (secure cookies, https invite links). Safe: the port is only reachable through Caddy.
ENV ALLOW_PROMPT_LOGGING=true \
    CACHE_DIR=/tmp/invoice-agent-cache \
    AUTH_DB=/home/user/data/auth.sqlite \
    FORWARDED_ALLOW_IPS=* \
    LOG_LEVEL=INFO

# OPENROUTER_API_KEY comes from the host's secrets, never the image.
EXPOSE 7860
CMD ["invoice-agent", "serve", "--host", "0.0.0.0", "--port", "7860"]
