# Aris Voice Agent

Talk to Aris from any browser. Open a URL, speak, get a voice response.

---

## QUICK START

### Step 1: Deploy GPU server on Modal

```bash
pip install modal
modal setup
modal deploy server.py
```

This deploys **both Whisper STT + Fish Speech TTS** on a single A10G GPU.
It returns a URL like `https://your-org--aris-voice-server.modal.run`.

**First deploy:** ~10 min (downloads ~10GB of model weights).
**After that:** instant.

### Step 2: Configure

```bash
cp .env.example .env
```

Fill in:
```
OPENROUTER_API_KEY=sk-or-v1-your-key
VOICE_SERVER_URL=https://your-org--aris-voice-server.modal.run
BOT_SECRET=some-random-string
```

### Step 3: Run the bot

```bash
pip install -e .
python bot.py
```

Open **http://localhost:7860** in Chrome. Click Connect → talk.

**First message:** 10-15 sec (Modal cold-start).
**After that:** instant.

---

## DEPLOY TO SYNOLOGY

```bash
# Copy repo to Synology
scp -r . synology:/path/to/aris-voice/

# Edit Caddyfile: replace aris.yourdomain.com with your domain
# Make sure DNS points to Synology's public IP

docker compose up -d
```

Then open `https://aris.yourdomain.com` from your iPhone.

---

## OPENCLAW INTEGRATION

Voice-Aris connects to your OpenClaw instance. Gives it memory, tools, coaching.

In `.env`:
```
OPENCLAW_GATEWAY_URL=ws://your-server-ip:18789
```

When you speak, the bot:
1. Audio → Modal Whisper → text
2. Text → `openclaw agent` → Aris's response (full context)
3. Response → Modal Fish Speech → audio → your speakers

---

## ARI TALKS TO YOU

```bash
curl -X POST https://your-domain.com/speak \
  -H "Authorization: Bearer your-bot-secret" \
  -H "Content-Type: application/json" \
  -d '{"text": "Dominius deadline is tomorrow. How are we looking?"}'
```

---

## ARCHITECTURE

```
Browser (WebRTC)
    ↕
Synology Docker
├── Pipecat bot (no GPU)
└── Caddy (HTTPS)
    ↕
Modal GPU (A10G, 15s scale-down)
├── Whisper large-v3 (STT)  ─ same container
└── Fish Speech S2 Pro (TTS) ─ 
    ↕
OpenClaw (this server)
```

**Cost:** ~$0.80/hr A10G, only when active. ~$0.40/day for 30 min use.

## FILES

| File | Purpose |
|------|---------|
| `bot.py` | Pipecat pipeline + WebRTC + OpenClaw bridge |
| `server.py` | Modal GPU server (Whisper + Fish Speech combined) |
| `whisper_stt.py` | Custom STT service (HTTP to Modal) |
| `fish_speech_tts.py` | Custom TTS service (HTTP to Modal) |
| `Dockerfile` | Bot container (includes OpenClaw CLI) |
| `docker-compose.yml` | Synology: bot + Caddy |
| `Caddyfile` | HTTPS reverse proxy |
