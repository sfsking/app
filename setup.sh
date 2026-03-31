#!/usr/bin/env bash
set -euo pipefail

# Install Node.js (18.x) and npm if not present
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "Installing Node.js 18.x..."
  curl -fsSL https://deb.nodesource.com/setup_18.x | bash - >/dev/null 2>&1 || true
  apt-get update -qq
  apt-get install -y nodejs >/dev/null 2>&1 || true
fi

# Ensure we're in the app directory (you can copy this repo under /kaggle/working/app)
cd "$(dirname "$0")"

# Install dependencies
echo "Installing npm dependencies..."
npm ci

# Start server + Pinggy tunnel
echo "Starting server and Pinggy tunnel..."
npm start
