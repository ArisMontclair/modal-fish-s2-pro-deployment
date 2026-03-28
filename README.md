# CLAWRION - Claw Audio Relay for Interactive Operations

Talk to OpenClaw from any browser. Phone, laptop, anything.

Open the page → tap Connect → speak → Aris responds with voice.

**What this does:**
- You talk into your browser → OpenClaw hears you and responds with voice
- OpenClaw has full access to his brain (OpenClaw) — memory, tools, coaching
- GPU only runs when you're talking (pay-per-use via Modal)
- Runs on your Synology via Docker

---

## What's in this repo

| File | Purpose |
|------|---------|
| `bot.py` | The web server + voice pipeline. Runs on Synology. |
| `dashboard.html` | The web page you open in your browser. Dark theme, status lights, log. |
| `server.py` | GPU server (Modal). Runs Whisper (speech-to-text) + Fish Speech (text-to-speech). |
| `whisper_stt.py` | Connects bot → Whisper on Modal |
| `fish_speech_tts.py` | Connects bot → Fish Speech on Modal |
| `Dockerfile` | Builds the bot container (includes OpenClaw CLI) |
| `docker-compose.yml` | Runs bot + Caddy (HTTPS) on Synology |
| `Caddyfile` | HTTPS reverse proxy config |

---

## How it works

```
You (iPhone browser)
    ↕  WebRTC (your voice goes up, OpenClaw's voice comes down)
Synology (this repo)
    ├── Bot (handles WebRTC, pipeline)
    └── Caddy (HTTPS)
    ↕  HTTP (text + audio)
Modal GPU (cloud, on-demand)
    ├── Whisper → turns your voice into text
    └── Fish Speech → turns OpenClaw's text into voice
    ↕  HTTP (text)
This server (OpenClaw's brain)
    └── OpenClaw → memory, tools, coaching, personality
```

When you speak:
1. Browser sends audio to Synology bot
2. Bot sends audio to Modal → Whisper turns it into text
3. Bot sends text to OpenClaw (OpenClaw's brain) on your server
4. OpenClaw thinks and writes a response
5. Bot sends response to Modal → Fish Speech turns it into voice
6. Voice plays in your browser

---

## Setup — 4 steps

### Step 1: Deploy GPU server on Modal (one-time, ~10 min)

Modal runs the GPU models so your Synology doesn't need a GPU.

```bash
# Install Modal CLI
pip install modal
modal setup

# From this repo directory:
modal deploy server.py
```

This deploys **Whisper + Fish Speech** on a single A10G GPU.

**First time:** Downloads ~10GB of model weights. Takes 10-15 minutes.
**After that:** Instant deploy.

When it finishes, it prints a URL. Save it. Example:
```
https://your-org--aris-voice-server.modal.run
```

### Step 2: Configure the bot

```bash
cp .env.example .env
```

Edit `.env` — fill in **3 things**:

```bash
# 1. The Modal URL from step 1
VOICE_SERVER_URL=https://your-org--aris-voice-server.modal.run

# 2. Your OpenClaw gateway (Aris's brain)
OPENCLAW_GATEWAY_URL=ws://192.168.178.134:18789
OPENCLAW_GATEWAY_TOKEN=REDACTED

# 3. A random secret (for the /speak endpoint)
BOT_SECRET=whatever-you-want
```

**That's it.** When `OPENCLAW_GATEWAY_URL` is set, all LLM logic goes through OpenClaw. No API keys needed.

<details>
<summary>Direct LLM mode (no OpenClaw)</summary>

If you want to run without OpenClaw, leave `OPENCLAW_GATEWAY_URL` empty and add:
```bash
OPENROUTER_API_KEY=sk-or-v1-xxxxx
LLM_MODEL=xiaomi/mimo-v2-pro
```
This uses OpenRouter directly — no memory, no tools, no coaching. Just a raw LLM.
</details>

### Step 3: Run on Synology

```bash
# Copy this repo to Synology
scp -r . synology:/path/to/aris-voice/

# On Synology:
cd /path/to/aris-voice
docker compose up -d
```

This starts:
- **Bot** on port 7860 (the voice agent)
- **Caddy** on ports 80/443 (HTTPS reverse proxy)

### Step 4: Set up HTTPS

WebRTC needs HTTPS for microphone access. Two options:

**Option A: Use Caddy with a domain (recommended)**

Edit `Caddyfile` — replace `OpenClaw.yourdomain.com` with your actual domain:
```
OpenClaw.yourdomain.com {
    reverse_proxy bot:7860
}
```

Point your domain's DNS to your Synology's public IP. Caddy automatically gets an HTTPS certificate via Let's Encrypt.

Then open `https://OpenClaw.yourdomain.com` from your iPhone.

**Option B: Quick test without a domain**

For local testing on your LAN, Chrome requires a flag:
1. Open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
2. Add `http://your-server-ip:7860`
3. Restart Chrome

Then open `http://your-server-ip:7860`.

---

## The dashboard

When you open the page, you see:

- **Three status cards** with colored dots: Modal GPU, OpenClaw, WebRTC
  - 🟢 Green = working
  - 🟡 Yellow = loading/connecting
  - 🔴 Red = down/unreachable
- **Big connect button** — tap to start talking to Aris
- **Settings panel** — tap ⚙️ to expand, edit URLs without restarting Docker
- **Log panel** — timestamped events for troubleshooting

Health checks run automatically every 30 seconds.

---

## Cost

Modal charges per-second for GPU usage:

- **A10G GPU:** ~$0.80/hour
- **Scale-down:** GPU shuts down 15 seconds after your last request
- **Typical usage:** ~$0.40/day for 30 minutes of conversation
- **Cold start:** First request after idle = 10-15 seconds (model loading)

You only pay when you're actively talking.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| 15 second delay on first message | Modal cold-starting GPU models | Normal. Subsequent messages are instant. |
| No mic access | Browser requires HTTPS | Set up Caddy with a domain, or use Chrome flag for local testing. |
| Modal GPU shows red | Server URL wrong or models still loading | Check `VOICE_SERVER_URL` in `.env`. Check `/health` endpoint. |
| OpenClaw shows red | Gateway unreachable from Docker | Check `OPENCLAW_GATEWAY_URL`. Make sure the server is on the same network. |
| Bot crashes on startup | Missing env vars | Check `.env` — `VOICE_SERVER_URL`, `OPENCLAW_GATEWAY_URL`, and `OPENCLAW_GATEWAY_TOKEN` must be set. |
| Docker build fails | Network issue | Run `docker compose build --no-cache` |
