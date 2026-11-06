#!/usr/bin/env bash
# invoice-setup: configure and run the Invoice Agent demo on this server.
#
#   sudo invoice-setup          ask for the domain and OpenRouter key; start
#   sudo invoice-setup admin    create your admin login (or reset its password)
#   sudo invoice-setup email    set up sending email (Resend); sends a test message
#   sudo invoice-setup langfuse turn on Langfuse tracing (Langfuse Cloud keys)
#   sudo invoice-setup update   pull the latest code from GitHub, rebuild and restart
#   sudo invoice-setup logs     follow the app's logs (Ctrl+C to stop)
#
# Keys are stored in /etc/invoice-agent (readable by root only), never in the repo.
# User accounts live in the app's database, on the Docker volume invoice_app_data.
set -euo pipefail

APP_DIR=/opt/invoice-agent
CONF_DIR=/etc/invoice-agent
APP_ENV="$CONF_DIR/app.env"
COMPOSE=(docker compose -f "$APP_DIR/deploy/lightsail/compose.yaml")
SAFE='^[A-Za-z0-9._@%!+-]+$'  # characters that survive Compose's env-file parsing

if [[ $EUID -ne 0 ]]; then
    echo "Run with sudo: sudo invoice-setup $*"
    exit 1
fi

# set_env NAME VALUE: add or replace one line in app.env, keeping the others.
set_env() {
    install -d -m 700 "$CONF_DIR"
    touch "$APP_ENV"
    (umask 077 && grep -v "^$1=" "$APP_ENV" > "$APP_ENV.tmp" || true)
    printf '%s=%s\n' "$1" "$2" >> "$APP_ENV.tmp"
    mv "$APP_ENV.tmp" "$APP_ENV"
    chmod 600 "$APP_ENV"
}

configure() {
    local domain key
    read -rp "Domain [invoice.aicruit.in]: " domain
    domain=${domain:-invoice.aicruit.in}
    read -rsp "OpenRouter API key (input hidden): " key
    echo
    if [[ ! $key =~ $SAFE ]]; then
        echo "That doesn't look like an API key; nothing was changed."
        exit 1
    fi
    set_env OPENROUTER_API_KEY "$key"
    (umask 077 && printf 'DOMAIN=%s\n' "$domain" > "$CONF_DIR/caddy.env")
    echo "Saved."
}

configure_email() {
    local from key
    read -rp "Send emails from [invites@aicruit.in]: " from
    from=${from:-invites@aicruit.in}
    if [[ ! $from =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+$ ]]; then
        echo "That doesn't look like an email address; nothing was changed."
        exit 1
    fi
    read -rsp "Resend API key (starts with re_, input hidden): " key
    echo
    if [[ ! $key =~ $SAFE ]]; then
        echo "That doesn't look like an API key; nothing was changed."
        exit 1
    fi
    set_env SMTP_HOST smtp.resend.com
    set_env SMTP_PORT 587
    set_env SMTP_USER resend
    set_env SMTP_PASSWORD "$key"
    set_env MAIL_FROM "$from"
    echo "Saved. Restarting the app with the new settings..."
    "${COMPOSE[@]}" up -d --force-recreate app
    sleep 5
    "${COMPOSE[@]}" exec app invoice-agent send-test-email
}

configure_langfuse() {
    local region host public secret
    read -rp "Langfuse Cloud region, eu or us [eu]: " region
    case "${region:-eu}" in
        eu) host=https://cloud.langfuse.com ;;
        us) host=https://us.cloud.langfuse.com ;;
        *) echo "Enter eu or us; nothing was changed."; exit 1 ;;
    esac
    read -rp "Langfuse public key (pk-lf-...): " public
    read -rsp "Langfuse secret key (sk-lf-..., input hidden): " secret
    echo
    if [[ ! $public =~ ^pk-lf- || ! $public =~ $SAFE || ! $secret =~ ^sk-lf- || ! $secret =~ $SAFE ]]; then
        echo "Those don't look like Langfuse keys; nothing was changed."
        exit 1
    fi
    set_env LANGFUSE_HOST "$host"
    set_env LANGFUSE_PUBLIC_KEY "$public"
    set_env LANGFUSE_SECRET_KEY "$secret"
    echo "Saved. Restarting the app with tracing on..."
    "${COMPOSE[@]}" up -d --force-recreate app
    echo "Upload an invoice; its trace appears in Langfuse (Tracing) within a minute."
}

start() {
    "${COMPOSE[@]}" up -d --build
    echo -n "Waiting for the app to start"
    for _ in $(seq 60); do
        if curl -fsS http://127.0.0.1:7860/health > /dev/null 2>&1; then
            echo " - up."
            echo "Open https://$(cut -d= -f2 "$CONF_DIR/caddy.env") (HTTPS works once DNS points here)."
            echo "No admin login yet? Run: sudo invoice-setup admin"
            return
        fi
        echo -n "."
        sleep 5
    done
    echo
    echo "The app didn't answer within 5 minutes. See: sudo invoice-setup logs"
    exit 1
}

case "${1:-}" in
    "")
        configure
        start
        ;;
    admin)
        "${COMPOSE[@]}" exec app invoice-agent create-admin
        ;;
    email)
        configure_email
        ;;
    langfuse)
        configure_langfuse
        ;;
    update)
        git -C "$APP_DIR" pull --ff-only
        install -m 755 "$APP_DIR/deploy/lightsail/setup.sh" /usr/local/bin/invoice-setup
        start
        ;;
    logs)
        "${COMPOSE[@]}" logs -f --tail 100
        ;;
    *)
        echo "Usage: sudo invoice-setup [admin|email|langfuse|update|logs]"
        exit 1
        ;;
esac
