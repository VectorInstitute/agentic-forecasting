#!/bin/bash
set -e

# ─────────────────────────────────────────────
# VM Setup Script — Forecasting Bootcamp
# ─────────────────────────────────────────────

echo "==> Updating system and installing essentials..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl git wget unzip build-essential openssh-server libomp-dev

echo "==> Enabling SSH..."
sudo systemctl enable ssh
sudo systemctl start ssh

echo "==> Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

echo "==> Installing Python 3.12 via uv..."
uv python install 3.12
uv python pin 3.12

echo "==> Setting up SSH authorized_keys..."
mkdir -p ~/.ssh && chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys

# ─────────────────────────────────────────────
# Paste your public key below (replace the placeholder)
# Get it locally with: cat ~/.ssh/id_ed25519.pub
# ─────────────────────────────────────────────
PUBLIC_KEY="YOUR_PUBLIC_KEY_HERE"

if [ "$PUBLIC_KEY" != "YOUR_PUBLIC_KEY_HERE" ]; then
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    echo "==> Public key added."
else
    echo "==> WARNING: No public key set. Edit PUBLIC_KEY in this script before running, or add your key manually."
fi

echo ""
echo "==> Done! VM IP address:"
curl -s ifconfig.me
echo ""
echo ""
echo "Add this to ~/.ssh/config on your local machine:"
echo "------"
echo "Host forecasting-vm"
echo "    HostName $(curl -s ifconfig.me)"
echo "    User $USER"
echo "    IdentityFile ~/.ssh/id_ed25519" # Replace with your actual chosen key path
echo "------"