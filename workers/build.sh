#!/bin/bash
# Build script for Cloudflare Workers Builds

set -e

echo "Installing pywrangler..."
pip install uv
uv tool install workers-py

echo "Building documentation..."
python build.py

echo "Deploying to Cloudflare Workers..."
pywrangler deploy
