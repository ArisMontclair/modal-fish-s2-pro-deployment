# CLAWRION — Claw Audio Relay for Interactive Operations

Talk to OpenClaw from any browser. Phone, laptop, anything.

Open the page → tap Connect → speak → OpenClaw responds with voice.

**What this does:**
- You talk into your browser → OpenClaw hears you and responds with voice
- OpenClaw has full access to his brain — memory, tools, coaching
- GPU only runs when you're talking (scale-to-zero on Modal)
- Runs on your Synology via Docker

---

## What's in this repo

| File | Purpose |
|------|---------|
| `bot.py` | Web server + voice pipeline. Runs on Synology. Handles WebRTC, routes to OpenClaw. |
| `dashboard.html` | Browser UI. Dark theme, status lights, log. |
| `server.py` | GPU server on Modal. Whisper STT + Fish Speech TTS. |
| `Dockerfile` | Builds the bot container (includes OpenClaw CLI) |
| `docker-compose.yml` | Runs bot + Caddy (HTTPS) on Synology |
| `Caddyfile` | HTTPS reverse proxy config |

---

## How it works

```
You (iPhone browser)
    ↕  WebRTC (your voice goes up, your agent's voice comes down)
Synology (this repo)
    ├── Bot (handles WebRTC, pipeline)
    └── Caddy (HTTPS)
    ↕  HTTP (text + audio)
Modal GPU (cloud, on-demand)
    ├── Whisper large-v3 → turns your voice into text
    └── Fish Speech S2 Pro → turns your agent's text into voice
    ↕  HTTP (text)
This server (OpenClaw's brain)
    └── OpenClaw → memory, tools, coaching, personality
```

When you speak:
1. Browser sends audio to Synology bot (WebRTC)
2. Bot sends audio to Modal → Whisper turns it into text
3. Bot sends text to OpenClaw on your server
4. OpenClaw thinks and writes a response
5. Bot sends response to Modal → Fish Speech turns it into voice
6. Voice plays in your browser

---

## server.py — Modal GPU Server

The GPU server does two things only: **speech-to-text** and **text-to-speech**. All LLM logic goes through OpenClaw — the GPU server never runs a language model.

### What it launches

On first request, the server:
1. Runs preflight checks (weights exist, fish-speech imports, torchaudio works)
2. Starts Fish Speech API server as a subprocess on `127.0.0.1:8081`
3. Loads Whisper large-v3 onto GPU
4. Marks ready

If any preflight check fails, the server stays in "loading" state and returns 503 on all endpoints. No silent fallback.

### Image design

The Modal image is intentionally minimal. Fish Speech manages its own dependencies:

```
nvidia/cuda:12.4.1 + Python 3.12
  ├── apt: git, ffmpeg
  ├── pip: faster-whisper
  ├── clone: fish-speech repo
  ├── pip: fish-speech [server] extras (manages torch, torchaudio, etc.)
  ├── pip: fastapi, uvicorn, httpx
  └── download: Fish Speech S2 Pro weights (~8GB, baked into image)
```

**Why fish-speech manages its own deps:** Pre-installing torch/torchaudio/sglang separately caused version conflicts. Fish-speech's `[server]` extras declare exact compatible versions. Letting pip resolve everything in one pass avoids mismatches.

### Scaling

- `max_containers=1` — never spin up a second GPU
- `min_containers=0` — scale to zero when idle (no cost when not talking)
- `scaledown_window=300` — stay warm 5 minutes after last request to avoid cold starts on short breaks

### Model weights

S2 Pro weights (~8GB) are downloaded during image build and baked in. No volumes, no runtime downloads. If the image is cached, deployment is instant. Rebuilding the image re-downloads weights only if layers before the download step change.

---

## Setup — 3 steps

### Step 1: Get a Modal token (free, 2 min)

1. Go to [modal.com/signup](https://modal.com/signup) — free account, $30/month GPU credit
2. Go to [modal.com/settings/tokens](https://modal.com/settings/tokens)
3. Create a token, copy the ID and secret

### Step 2: Configure the bot

```bash
cp .env.example .env
```

Edit `.env` — fill in **3 things**:

```bash
# 1. Your Modal token (from step 1)
MODAL_TOKEN_ID=your-token-id
MODAL_TOKEN_SECRET=your-token-secret

# 2. Your OpenClaw gateway (your agent's brain)
OPENCLAW_GATEWAY_URL=ws://your-server:18789
OPENCLAW_GATEWAY_TOKEN=your-token-here
```

The first `docker compose up` deploys the GPU server to Modal and grabs the URL automatically. If you already have a Modal deployment, set `VOICE_SERVER_URL` directly instead.

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
scp -r . synology:/path/to/CLAWRION/

# On Synology:
cd /path/to/CLAWRION
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
- **Big connect button** — tap to start talking to OpenClaw
- **Settings panel** — tap ⚙️ to expand, edit URLs without restarting Docker
- **Log panel** — timestamped events for troubleshooting

---

## Cost

Modal charges per-second for GPU usage:

- **A10G GPU:** ~$0.80/hour
- **Scale-down:** GPU shuts down 5 minutes after your last request (`scaledown_window=300`)
- **Idle cost:** $0 when scaled to zero (`min_containers=0`)
- **Cold start:** First request after idle = 60-120 seconds (loading Whisper + starting Fish Speech)
- **Warm start:** Instant (within 5 min window)

You only pay when the GPU is actually running.

### What NOT to do

- **Don't keep the dashboard tab open with health checks hitting Modal** — each ping keeps the GPU alive. The lightweight `/health` endpoint on Modal runs without GPU, but the bot's health check to the GPU server (`/health` on the A10G) will wake it.

---

## /speak — Push Voice to Connected Browsers

OpenClaw (or any service) can push voice messages to all connected browsers via the `/speak` endpoint.

```bash
curl -X POST https://your-domain/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hey John, time to wrap up for the night."}'
```

The text gets converted to speech via Fish Speech on Modal, then pushed through WebRTC to all connected browsers.

**Use cases:** OpenClaw cron jobs pushing coaching nudges, proactive alerts, ambient voice notifications.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Cold start delay (60-120s) | Modal loading models from image | Normal after idle. Subsequent requests are instant within 5 min. |
| No mic access | Browser requires HTTPS | Set up Caddy with a domain, or use Chrome flag for local testing. |
| Modal GPU shows red | Server URL wrong or models still loading | Check `VOICE_SERVER_URL` in `.env`. Wait for cold start. |
| TTS returns 503 | Fish Speech failed preflight | Check Modal logs: `modal app logs your-agent-voice`. Look for `FATAL:` messages. |
| OpenClaw shows red | Gateway unreachable from Docker | Check `OPENCLAW_GATEWAY_URL`. Make sure the server is on the same network. |
| Bot crashes on startup | Missing env vars | Check `.env` — `VOICE_SERVER_URL` must be set. |
| High GPU bill | Dashboard kept GPU alive via health checks | Scale to zero works. Don't poll the GPU health endpoint from the browser. |
