#!/usr/bin/env bash
# One-time server preparation, run as root by the Lightsail launch script at first boot.
# Installs Docker, adds swap, builds the image and installs the `invoice-setup` command.
# Secrets are NOT set here: the owner types them in with `sudo invoice-setup`.
# Progress: /var/log/cloud-init-output.log
set -euxo pipefail

APP_DIR=/opt/invoice-agent
CONF_DIR=/etc/invoice-agent

# 2 GB swap: headroom for the image build and OCR on a small instance.
if ! swapon --show | grep -q /swapfile; then
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io docker-compose-v2
systemctl enable --now docker

# Empty settings files so Compose can build now; `invoice-setup` fills them in.
install -d -m 700 "$CONF_DIR"
install -m 600 /dev/null "$CONF_DIR/app.env"
install -m 600 /dev/null "$CONF_DIR/caddy.env"

install -m 755 "$APP_DIR/deploy/lightsail/setup.sh" /usr/local/bin/invoice-setup

# Build ahead of time, so the first `invoice-setup` only has to start containers.
docker compose -f "$APP_DIR/deploy/lightsail/compose.yaml" build

echo "Invoice Agent: server ready. Run 'sudo invoice-setup' to finish." > /etc/motd
