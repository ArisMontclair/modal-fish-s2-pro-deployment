# Aris Voice Agent

Talk to Aris from any browser. Open a URL, speak, get a voice response.

**What this is:** A Pipecat voice pipeline that connects your browser mic/speaker to Aris (OpenClaw) via self-hosted TTS and STT on Modal GPU.

**What you need:**
- A Modal account (free): https://modal.com
- An OpenRouter API key
- This machine (or Synology) with Docker

---

## QUICK START — Test locally (10 min)

### Step 1: Deploy GPU services on Modal

```bash
pip install modal
modal setup

# From the repo directory:
modal deploy stt_server.py    # deploys Whisper STT
modal deploy tts_server.py    # deploys Fish Speech TTS
```

Each deploy gives you a URL. Save them. Example:
```
https://your-org--whisper-stt-server.modal.run
https://your-org--fish-tts-server.modal.run
```

**Note:** First deploy takes 5-10 min (downloads ~8GB of model weights into Modal volume). Subsequent deploys are instant.

### Step 2: Configure

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENROUTER_API_KEY=sk-or-v1-your-key
WHISPER_STT_URL=https://your-org--whisper-stt-server.modal.run
FISH_TTS_URL=https://your-org--fish-tts-server.modal.run
BOT_SECRET=some-random-string
```

### Step 3: Run the bot locally

```bash
pip install -e .
python bot.py
```

Open **http://localhost:7860** in your browser (Chrome recommended).
Click **Connect** → allow mic → start talking.

**First message** will take 10-15 sec (Modal cold-starting GPU models).
**After that** — instant.

---

## DEPLOY TO SYNOLOGY

### Step 1: Copy files to Synology

```bash
scp -r . your-synology:/path/to/aris-voice/
```

### Step 2: Edit Caddyfile

Replace `aris.yourdomain.com` with your actual domain:
```
aris.yourdomain.com {
    reverse_proxy bot:7860
}
```

Make sure your domain's DNS points to your Synology's public IP.

### Step 3: Run

```bash
docker compose up -d
```

### Step 4: Access

Open `https://aris.yourdomain.com` from your iPhone.
HTTPS is automatic (Caddy + Let's Encrypt).

---

## OPENCLAW INTEGRATION

The bot can talk to Aris (your OpenClaw instance) instead of using an LLM directly. This gives voice-Aris access to memory, tools, coaching context — everything.

### On the server running OpenClaw:

1. Make sure the gateway is accessible from the bot's Docker container
2. Note the gateway URL: `ws://your-server-ip:18789`

### In `.env`:

```
OPENCLAW_GATEWAY_URL=ws://your-server-ip:18789
```

### How it works:

When you speak, the bot:
1. Sends audio to Modal Whisper → gets text
2. Calls `openclaw agent --to +31681299666 --message "your text" --json`
3. Gets Aris's response (full context: memory, tools, coaching)
4. Sends response to Modal Fish Speech → plays audio

**Voice-mode tag:** The bot prepends a `[VOICE MODE]` instruction to each message so Aris responds concisely (1-3 sentences, conversational, no bullet points).

---

## ARI TALKS TO YOU

Aris can push voice messages to a connected browser via the `/speak` endpoint:

```bash
curl -X POST https://aris.yourdomain.com/speak \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-bot-secret" \
  -d '{"text": "Morning John. Dominius deadline is tomorrow. How are we looking?"}'
```

The bot synthesizes via Fish Speech on Modal and plays it through the browser.

**From Aris (this server):** I can call this endpoint to push voice notes to you when you have the browser open.

---

## ARCHITECTURE

```
iPhone browser (WebRTC)
       ↕
Synology Docker
  ├── Pipecat bot (WebRTC + pipeline, no GPU)
  └── Caddy (HTTPS reverse proxy)
       ↕
Modal GPU (on-demand, 15s scale-down)
  ├── Whisper large-v3 (STT)
  └── Fish Speech S2 Pro (TTS)
       ↕
OpenClaw gateway (this server)
  └── Aris (memory, tools, coaching)
```

**Cost:** Modal A10G = ~$0.80/hr, only when active. 15 sec idle → GPU dies.
~$0.40/day for 30 min of conversation.

**Cold start:** First message after idle = 10-15 sec (Modal loading models).
Subsequent messages = instant.

---

## FILES

| File | What it does |
|------|-------------|
| `bot.py` | Main Pipecat pipeline + WebRTC + OpenClaw bridge |
| `whisper_stt.py` | Custom STT service → calls Modal Whisper via HTTP |
| `fish_speech_tts.py` | Custom TTS service → calls Modal Fish Speech via HTTP |
| `stt_server.py` | Modal deployment: Whisper large-v3 on A10G |
| `tts_server.py` | Modal deployment: Fish Speech S2 Pro on A10G |
| `docker-compose.yml` | Synology deployment (bot + Caddy) |
| `Dockerfile` | Bot container (includes OpenClaw CLI) |
| `Caddyfile` | HTTPS reverse proxy config |

---

## TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| 15 sec delay on first message | Normal — Modal cold-start. Subsequent messages instant |
| No audio in browser | HTTPS required for WebRTC mic. Use Caddy or `ngrok http 7860` for testing |
| `Connection refused` to Modal | Check URLs in `.env`. Run `modal deploy` again |
| OpenClaw not responding | Check `OPENCLAW_GATEWAY_URL` is reachable from Docker. Try `docker exec aris-voice-bot openclaw health` |
| `/speak` returns 401 | Check `BOT_SECRET` matches between `.env` and curl command |
| Modal OOM | S2 Pro needs ~12GB VRAM. A10G (24GB) is sufficient |
