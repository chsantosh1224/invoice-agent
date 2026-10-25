#!/bin/bash
# Paste this into Lightsail's "Add launch script" box when creating the instance.
set -eux
apt-get update
apt-get install -y git
git clone https://github.com/chsantosh1224/invoice-agent.git /opt/invoice-agent
bash /opt/invoice-agent/deploy/lightsail/bootstrap.sh
