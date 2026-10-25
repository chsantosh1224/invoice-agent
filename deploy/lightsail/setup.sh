#!/usr/bin/env bash
# invoice-setup: configure and run the Invoice Agent demo on this server.
#
#   sudo invoice-setup          ask for the domain and OpenRouter key; start
#   sudo invoice-setup admin    create your admin login (or reset its password)
#   sudo invoice-setup update   pull the latest code from GitHub, rebuild and restart
#   sudo invoice-setup logs     follow the app's logs (Ctrl+C to stop)
#
# The API key is stored in /etc/invoice-agent (readable by root only), never in the repo.
# User accounts live in the app's database, on the Docker volume invoice_app_data.
set -euo pipefail

APP_DIR=/opt/invoice-agent
CONF_DIR=/etc/invoice-agent
COMPOSE=(docker compose -f "$APP_DIR/deploy/lightsail/compose.yaml")
SAFE='^[A-Za-z0-9._@%!+-]+$'  # characters that survive Compose's env-file parsing

if [[ $EUID -ne 0 ]]; then
    echo "Run with sudo: sudo invoice-setup $*"
    exit 1
fi

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
    install -d -m 700 "$CONF_DIR"
    umask 077
    printf 'OPENROUTER_API_KEY=%s\n' "$key" > "$CONF_DIR/app.env"
    printf 'DOMAIN=%s\n' "$domain" > "$CONF_DIR/caddy.env"
    echo "Saved."
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
    update)
        git -C "$APP_DIR" pull --ff-only
        install -m 755 "$APP_DIR/deploy/lightsail/setup.sh" /usr/local/bin/invoice-setup
        start
        ;;
    logs)
        "${COMPOSE[@]}" logs -f --tail 100
        ;;
    *)
        echo "Usage: sudo invoice-setup [admin|update|logs]"
        exit 1
        ;;
esac
