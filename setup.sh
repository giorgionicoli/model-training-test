#!/bin/bash

# Exit immediately if any command fails
set -e

echo "Installing uv package manager..."
curl -Ls https://astral.sh/uv/install.sh | bash

# Ensure uv is available in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv installation failed or uv is not in PATH."
    exit 1
fi

echo "Syncing project dependencies using uv..."
uv sync

echo "All dependencies installed successfully in the uv-managed virtual environment."
