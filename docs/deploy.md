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
3. **Approve** emails them a one-time invite link (valid 7 days), once email is set up
   (see [Email](#email)); the admin page shows the link too, as a backup. Without email,
   send the link yourself. Don't open it yourself: that uses it up.
4. They open it, choose a password and are logged in. Logins last 14 days.

Also on the admin page: **Reject** a request; **Disable** someone (logs them out
everywhere); **New invite link** / **Password reset link** if a link expired or a password
was forgotten (the old password works until the new link is used). Five wrong passwords
for one email block that email for 15 minutes. Forgot your own admin password? Run
`sudo invoice-setup admin` again with the same email.

## Human review

Invoices the validation gate flags go to a review queue instead of being accepted. Admins
see **Review (n)** in the app's header (or open `/admin/reviews`):

1. Pick an invoice: the document is shown beside its fields, with the flagged ones
   highlighted and the gate's reasons (and the agent's notes) above.
2. Correct what the document shows, then **Approve**. The gate re-checks the corrected
   invoice against the document; if it still fails, approval needs a note and is recorded
   as an **override**. **Reject** (with a reason) for duplicates, non-invoices and so on.
3. Deciding deletes the uploaded file. With tracing on, the decision is added to the run's
   Langfuse trace as scores: `human_review` (approved / corrected / rejected) and
   `fields_corrected`.

`invoice-agent export-reviews` turns approved reviews into eval cases (answer keys in the
format of `data/samples`) and lists the fields reviewers corrected most. To re-run them as
regression tests, keep the documents: set `REVIEW_KEEP_FILES=true` (only for demo data). On
the server: `sudo docker compose -f /opt/invoice-agent/deploy/lightsail/compose.yaml exec
app invoice-agent export-reviews --out /home/user/data/reviewed`.

## Email

Invites and "new access request" notices go out through [Resend](https://resend.com)
(free up to 100 emails a day) from `invites@aicruit.in`. One-time setup:

1. Sign up at resend.com → *Domains → Add domain* → `aicruit.in` (region of your choice).
2. Resend lists DNS records (a DKIM `TXT`, plus an `MX` and a `TXT` for the `send`
   subdomain; optionally DMARC). Add each in GoDaddy (*aicruit.in → DNS → Add new
   record*), copying name and value exactly. Note GoDaddy adds `.aicruit.in` to names
   itself: enter `resend._domainkey`, not `resend._domainkey.aicruit.in`.
3. Back in Resend, *Verify*; wait until the domain shows **Verified** (minutes to hours).
4. Resend → *API Keys → Create* with **Sending access** only.
5. On the server: `sudo invoice-setup email` → From address (Enter for
   `invites@aicruit.in`) and the API key (hidden). It restarts the app and asks where to
   send a test email.

New links for people who already have access (*New invite link*, *Password reset link*)
are not emailed: copy them from the admin page. If an email fails, the admin page says
why and shows the link to send by hand.

## Tracing (Langfuse)

Optional. Each upload becomes one trace in [Langfuse Cloud](https://cloud.langfuse.com):
the fast path's LLM call, every agent step and tool call, tokens, latency, cost, the user
(account id) and the outcome. One-time setup:

1. Sign up at cloud.langfuse.com (free Hobby plan) → create an organisation and a project
   (pick the EU or US region).
2. Project → *Settings → API Keys → Create new API keys*.
3. On the server: `sudo invoice-setup langfuse` → region, public key, secret key (hidden).
   It restarts the app.
4. Upload an invoice; it shows up under *Tracing* within a minute.

Traces contain the invoice text and the models' answers, so use demo invoices, or tell a
client before running their real ones. To turn tracing off, delete the three `LANGFUSE_`
lines from `/etc/invoice-agent/app.env` and run `sudo invoice-setup update`.

## Day to day

```bash
sudo invoice-setup update   # after pushing to GitHub: pull, rebuild, restart
sudo invoice-setup logs     # follow the app's logs
sudo invoice-setup admin    # create the admin, or reset its password
sudo invoice-setup email    # set or change the email settings (sends a test)
sudo invoice-setup langfuse # set or change the Langfuse keys
sudo invoice-setup          # change the domain or the API key
```

## Alternative: Hugging Face Spaces

`deploy/huggingface/README.md` is a ready Space card for the same Dockerfile (port 7860).
Docker Spaces now need a paid Hugging Face plan (PRO), and Spaces restart with an empty
disk, so user accounts would need persistent storage there. With one: create a Docker
Space, add `OPENROUTER_API_KEY` as a secret, and upload with
`hf upload <user>/<space> deploy/huggingface/README.md README.md --repo-type space` and
`hf upload <user>/<space> . . --repo-type space --include "Dockerfile" "pyproject.toml" "uv.lock" "src/**"`.
