#!/bin/bash
set -e

echo "Setting up modal-fish-s2-pro..."

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Done. To activate:"
echo "  source .venv/bin/activate"
echo ""
echo "To deploy:"
echo "  modal setup   # first time only"
echo "  modal deploy modal_app.py"
