# Deploying the demo (Hugging Face Spaces)

The demo runs as a **Docker Space** on Hugging Face's free CPU tier and is reached at
`invoice.aicruit.in`, which GoDaddy forwards to the Space.

- Page: `https://huggingface.co/spaces/<hf-user>/<space>`
- App (share this one): `https://<hf-user>-<space>.hf.space`
- Login: HTTP Basic auth (user `demo`, password from the `DEMO_PASSWORD` secret).
- The free Space sleeps after a period without visitors; the first visit after that
  takes about a minute. Open it a few minutes before a demo.

## One-time setup

1. **Create the Space.** On huggingface.co: *New Space*, then
   - name: something short, e.g. `invoice` (the app URL becomes `<hf-user>-invoice.hf.space`),
   - SDK: **Docker**, template **Blank**,
   - hardware: **CPU basic (free)**,
   - visibility: **Public** (the app's own password protects it; a private Space would
     require clients to have Hugging Face accounts).
2. **Add the secrets.** Space *Settings → Variables and secrets → New secret*:
   - `OPENROUTER_API_KEY`: your OpenRouter key,
   - `DEMO_PASSWORD`: a password you choose for clients.
   Optional variables: `DEMO_USER` (default `demo`), any model override such as
   `AGENT_MODEL`.
3. **Install the Hugging Face CLI and log in** (needs a token with *write* access, from
   *Settings → Access Tokens*; the CLI asks for it, so paste it there, nowhere else):
   ```bash
   pip install -U huggingface_hub
   hf auth login
   ```

## Deploy (first time and every update)

From the project root, with `<hf-user>/<space>` replaced:

```bash
hf upload <hf-user>/<space> deploy/huggingface/README.md README.md --repo-type space
hf upload <hf-user>/<space> . . --repo-type space --include "Dockerfile" "pyproject.toml" "uv.lock" "src/**" --exclude "**/__pycache__/**"
```

The first command uploads the Space card (its YAML header tells Hugging Face this is a
Docker app on port 7860). The second uploads only what the image needs. Each upload is a
commit; the Space rebuilds automatically (about 5 minutes) and shows the build log under
*Logs*.

Check it: `https://<hf-user>-<space>.hf.space/health` returns `{"status": "ok", ...}`
without a password; `/` asks for the login.

## invoice.aicruit.in (GoDaddy forwarding)

GoDaddy: *My Products → aicruit.in → DNS → Forwarding → Add forwarding → Subdomain*:

- subdomain: `invoice`
- destination: `https://<hf-user>-<space>.hf.space`
- type: **Permanent (301)**
- settings: **Forward only** (not "forward with masking": masking puts the app in a frame,
  where the browser won't show the login box)

Allow up to an hour for it to take effect. Visitors then see the `.hf.space` address after
the redirect; to move the demo elsewhere later (e.g. AWS), change only the destination.
