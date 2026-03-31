#!/usr/bin/env bash
set -euo pipefail
if ! command -v node >/dev/null 2>&1; then
  echo "Installing Node.js 18.x..."
  curl -fsSL https://deb.nodesource.com/setup_18.x | bash - >/dev/null 2>&1 || true
  apt-get update -qq && apt-get install -y nodejs >/dev/null 2>&1 || true
fi
cd "$(dirname "$0")"
echo "Installing dependencies..."
npm ci
echo "Starting server..."
npm start
