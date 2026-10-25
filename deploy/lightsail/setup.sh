#!/usr/bin/env bash
# invoice-setup: configure and run the Invoice Agent demo on this server.
#
#   sudo invoice-setup          ask for the domain, OpenRouter key and demo password; start
#   sudo invoice-setup update   pull the latest code from GitHub, rebuild and restart
#   sudo invoice-setup logs     follow the app's logs (Ctrl+C to stop)
#
# Secrets are stored in /etc/invoice-agent (readable by root only), never in the repo.
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
    local domain key pw pw2
    read -rp "Domain [invoice.aicruit.in]: " domain
    domain=${domain:-invoice.aicruit.in}
    read -rsp "OpenRouter API key (input hidden): " key
    echo
    if [[ ! $key =~ $SAFE ]]; then
        echo "That doesn't look like an API key; nothing was changed."
        exit 1
    fi
    while true; do
        read -rsp "Demo password for clients (input hidden): " pw
        echo
        read -rsp "Repeat the password: " pw2
        echo
        if [[ $pw != "$pw2" ]]; then
            echo "The passwords don't match; try again."
        elif [[ ${#pw} -lt 10 || ! $pw =~ $SAFE ]]; then
            echo "Use at least 10 characters: letters, digits and . _ @ % ! + - only."
        else
            break
        fi
    done
    install -d -m 700 "$CONF_DIR"
    umask 077
    printf 'OPENROUTER_API_KEY=%s\nDEMO_USER=demo\nDEMO_PASSWORD=%s\n' "$key" "$pw" > "$CONF_DIR/app.env"
    printf 'DOMAIN=%s\n' "$domain" > "$CONF_DIR/caddy.env"
    echo "Saved. Log in on the site as user 'demo' with that password."
}

start() {
    "${COMPOSE[@]}" up -d --build
    echo -n "Waiting for the app to start"
    for _ in $(seq 60); do
        if curl -fsS http://127.0.0.1:7860/health > /dev/null 2>&1; then
            echo " - up."
            echo "Open https://$(cut -d= -f2 "$CONF_DIR/caddy.env") (HTTPS works once DNS points here)."
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
    update)
        git -C "$APP_DIR" pull --ff-only
        install -m 755 "$APP_DIR/deploy/lightsail/setup.sh" /usr/local/bin/invoice-setup
        start
        ;;
    logs)
        "${COMPOSE[@]}" logs -f --tail 100
        ;;
    *)
        echo "Usage: sudo invoice-setup [update|logs]"
        exit 1
        ;;
esac
