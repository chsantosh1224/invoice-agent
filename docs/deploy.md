# Deploying the demo (AWS Lightsail)

The demo runs on one Lightsail instance (currently Sydney, `ap-southeast-2`): the app in
Docker, and [Caddy](https://caddyserver.com) in front of it for HTTPS. It is reached at
`https://invoice.aicruit.in`.

- Access: personal accounts. Visitors ask at `/request-access`; the admin approves them at
  `/admin` and sends each one a one-time invite link to set their password (see
  [Accounts](#accounts)). `/health` is open without logging in, for checks.
- Files: [deploy/lightsail](../deploy/lightsail): `launch-script.sh` (pasted into the
  console), `bootstrap.sh` (first boot), `setup.sh` (installed as `invoice-setup`),
  `compose.yaml`, `Caddyfile`.
- The OpenRouter key lives only on the server, in `/etc/invoice-agent` (root-only). User
  accounts live in the app's SQLite database on the Docker volume `invoice_app_data`
  (kept across rebuilds; passwords are stored only as scrypt hashes).
- Cost: the plan's monthly price whether the instance is running or stopped; delete the
  instance (and release the static IP) to stop paying.

## 1. Create the instance (Lightsail console)

1. Pick a region (Mumbai `ap-south-1` is closest to India; any works); platform
   **Linux/Unix**; blueprint **OS Only → Ubuntu 24.04 LTS**.
2. *Add launch script*: paste the contents of `deploy/lightsail/launch-script.sh`.
3. Plan: **2 GB RAM** (the 1 GB plan works but builds slowly).
4. Create it.

## 2. Static IP and HTTPS port (instance → *Networking* tab)

1. *Attach static IP* → create one and attach it. Without it the IP changes whenever the
   instance stops and starts, and the domain breaks.
2. *IPv4 Firewall → Add rule* → **HTTPS (TCP 443)**, source *any IPv4 address*. Keep
   SSH (22) and HTTP (80): HTTP is needed for the certificate and redirects to HTTPS.

## 3. DNS (GoDaddy)

*My Products → aicruit.in → DNS → Add new record*:

- type **A**, name `invoice`, value: the static IP, TTL 600 (or the lowest offered).

Remove any forwarding set up earlier for `invoice`. Check from your PC with
`nslookup invoice.aicruit.in` until it shows the static IP.

## 4. Finish on the server

The launch script needs about 10 minutes after creation (Docker install and image build).
Then, in the console: instance → *Connect using SSH* (opens a terminal in the browser):

```bash
sudo invoice-setup         # domain + OpenRouter key; starts the app
sudo invoice-setup admin   # your admin login: email, name, password (hidden)
```

Caddy gets the HTTPS certificate by itself, provided DNS already points at the server.

If `invoice-setup` is "command not found", the first-boot script is still running or
failed: `tail -f /var/log/cloud-init-output.log`.

## Accounts

1. A visitor fills in `/request-access` (name, email, optional company and reason).
2. You log in and open **Admin** (top right): pending requests are listed first.
3. **Approve** shows a one-time invite link (valid 7 days). Send it yourself, e.g. on
   Upwork, email or WhatsApp. Don't open it yourself: that uses it up.
4. They open it, choose a password and are logged in. Logins last 14 days.

Also on the admin page: **Reject** a request; **Disable** someone (logs them out
everywhere); **New invite link** / **Password reset link** if a link expired or a password
was forgotten (the old password works until the new link is used). Five wrong passwords
for one email block that email for 15 minutes. Forgot your own admin password? Run
`sudo invoice-setup admin` again with the same email.

## Day to day

```bash
sudo invoice-setup update   # after pushing to GitHub: pull, rebuild, restart
sudo invoice-setup logs     # follow the app's logs
sudo invoice-setup admin    # create the admin, or reset its password
sudo invoice-setup          # change the domain or the API key
```

## Alternative: Hugging Face Spaces

`deploy/huggingface/README.md` is a ready Space card for the same Dockerfile (port 7860).
Docker Spaces now need a paid Hugging Face plan (PRO), and Spaces restart with an empty
disk, so user accounts would need persistent storage there. With one: create a Docker
Space, add `OPENROUTER_API_KEY` as a secret, and upload with
`hf upload <user>/<space> deploy/huggingface/README.md README.md --repo-type space` and
`hf upload <user>/<space> . . --repo-type space --include "Dockerfile" "pyproject.toml" "uv.lock" "src/**"`.
