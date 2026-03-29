#!/usr/bin/env python3
"""Deploy server.py to Modal and print the app URL.

Usage:
  MODAL_TOKEN_ID=xxx MODAL_TOKEN_SECRET=xxx python deploy_modal.py

Or just let the Docker entrypoint handle it.
"""
import os
import re
import subprocess
import sys


def deploy():
    token_id = os.getenv("MODAL_TOKEN_ID", "")
    token_secret = os.getenv("MODAL_TOKEN_SECRET", "")

    if not token_id or not token_secret:
        print("No Modal credentials found. Skipping deploy.")
        print("Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env to auto-deploy.")
        sys.exit(0)

    env = {**os.environ, "MODAL_TOKEN_ID": token_id, "MODAL_TOKEN_SECRET": token_secret}

    print("Deploying GPU server to Modal (this may take 10-15 min on first run)...")
    result = subprocess.run(
        ["modal", "deploy", "server.py"],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )

    if result.returncode != 0:
        print(f"Modal deploy FAILED:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(result.stdout)

    # Extract server URL from output (skip health endpoint, get the actual server URL)
    # Output has lines like:
    #   ├── 🔨 Created web function health => https://xxx--health.modal.run
    #   └── 🔨 Created web function server =>\nhttps://xxx--aris-voice-server.modal.run
    url_match = re.search(r"server\s*=>\s*\n?\s*(https://\S+\.modal\.run)", result.stdout)
    if not url_match:
        # Fallback: grab last URL (usually the server, not health)
        all_urls = re.findall(r"https://\S+\.modal\.run", result.stdout)
        url_match = type('obj', (object,), {'group': lambda self, n=0: all_urls[-1] if all_urls else None})()
    if url_match:
        url = url_match.group(0)
        print(f"\nVOICE_SERVER_URL={url}")
        # Write to a file so the entrypoint can source it
        with open("/tmp/modal_url", "w") as f:
            f.write(url)
    else:
        print("WARNING: Could not parse URL from modal deploy output.", file=sys.stderr)
        print("Set VOICE_SERVER_URL manually in .env", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    deploy()
