# Deploying the demo (AWS Lightsail)

The demo runs on one Lightsail instance in Mumbai (`ap-south-1`): the app in Docker, and
[Caddy](https://caddyserver.com) in front of it for HTTPS. It is reached at
`https://invoice.aicruit.in`.

- Login: the browser asks for user `demo` and the password set with `invoice-setup`.
- `/health` is open without a password, for checks.
- Files: [deploy/lightsail](../deploy/lightsail): `launch-script.sh` (pasted into the
  console), `bootstrap.sh` (first boot), `setup.sh` (installed as `invoice-setup`),
  `compose.yaml`, `Caddyfile`.
- Secrets (OpenRouter key, demo password) live only on the server, in
  `/etc/invoice-agent` (root-only). They are typed in over SSH, never committed.
- Cost: the plan's monthly price whether the instance is running or stopped; delete the
  instance (and release the static IP) to stop paying.

## 1. Create the instance (Lightsail console)

1. Region **Mumbai (ap-south-1)**; platform **Linux/Unix**; blueprint **OS Only → Ubuntu 24.04 LTS**.
2. *Add launch script*: paste the contents of `deploy/lightsail/launch-script.sh`.
3. Plan: **2 GB RAM** (the 1 GB plan works but builds slowly).
4. Name it `invoice-agent` and create it.

## 2. Static IP and HTTPS port (instance → *Networking* tab)

1. *Attach static IP* → create one and attach it. Without it the IP changes whenever the
   instance stops and starts, and the domain breaks.
2. *IPv4 Firewall → Add rule* → **HTTPS (TCP 443)**. Keep SSH (22) and HTTP (80): HTTP is
   needed for the certificate and redirects to HTTPS.

## 3. DNS (GoDaddy)

*My Products → aicruit.in → DNS → Add new record*:

- type **A**, name `invoice`, value: the static IP, TTL 600 (or the lowest offered).

Remove any forwarding set up earlier for `invoice`. Check from your PC with
`nslookup invoice.aicruit.in` until it shows the static IP.

## 4. Finish on the server

The launch script needs about 10 minutes after creation (Docker install and image build).
Then, in the console: instance → *Connect using SSH* (opens a terminal in the browser):

```bash
sudo invoice-setup
```

It asks for the domain (Enter for `invoice.aicruit.in`), your OpenRouter API key and a
demo password, starts the app, and waits until it answers. Caddy then gets the HTTPS
certificate by itself, provided DNS already points at the server.

If `invoice-setup` is "command not found", the first-boot script is still running or
failed: `tail -f /var/log/cloud-init-output.log`.

## Day to day

```bash
sudo invoice-setup update   # after pushing to GitHub: pull, rebuild, restart
sudo invoice-setup logs     # follow the app's logs
sudo invoice-setup          # change the API key or the demo password
```

## Alternative: Hugging Face Spaces

`deploy/huggingface/README.md` is a ready Space card for the same Dockerfile (port 7860).
Docker Spaces now need a paid Hugging Face plan (PRO). With one: create a Docker Space,
add `OPENROUTER_API_KEY` and `DEMO_PASSWORD` as secrets, and upload with
`hf upload <user>/<space> deploy/huggingface/README.md README.md --repo-type space` and
`hf upload <user>/<space> . . --repo-type space --include "Dockerfile" "pyproject.toml" "uv.lock" "src/**"`.
