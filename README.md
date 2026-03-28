# Fish S2 Pro Voice Agent — Self-Hosted, Aris-Connected

Browser-based voice AI that talks to **Aris** (OpenClaw), not a random LLM.

**What it does:**
- Open a URL on any device → talk to Aris with voice
- Aris can also talk to you (proactive voice push via `/speak` endpoint)
- Self-hosted TTS (Fish Speech S2 Pro) — zero per-character fees
- GPU only runs on-demand via Modal — pay only when talking

**Architecture:**
```
┌──────────┐     ┌───────────────────────┐     ┌──────────────────┐
│  Browser │────▶│  Pipecat Bot          │────▶│  OpenClaw (Aris) │
│  (phone) │◀────│  (Synology Docker)    │◀────│  (this server)   │
└──────────┘     └───────────────────────┘     └──────────────────┘
       WebRTC           │         │
                        │  HTTP   │  HTTP
                        ▼         ▼
                 ┌────────────┐ ┌────────────────┐
                 │  Modal GPU │ │  Modal GPU     │
                 │  Whisper   │ │  Fish Speech   │
                 │  (STT)     │ │  S2 Pro (TTS)  │
                 └────────────┘ └────────────────┘
```

## SETUP

### Prerequisites
- Synology with Docker (or any Linux server)
- Modal account (free to sign up: https://modal.com)
- OpenRouter API key
- Domain with HTTPS (or use Synology's built-in reverse proxy)

### 1. Deploy GPU services on Modal

```bash
pip install modal
modal setup

# Deploy Whisper STT server
modal deploy stt_server.py

# Deploy Fish Speech TTS server
modal deploy tts_server.py
```

After deploy, you'll get URLs like:
- `https://your-org--whisper-stt-server.modal.run`
- `https://your-org--fish-tts-server.modal.run`

### 2. Configure the bot

```bash
cp .env.example .env
```

Fill in:
```
OPENROUTER_API_KEY=sk-or-v1-...
WHISPER_STT_URL=https://your-org--whisper-stt-server.modal.run
FISH_TTS_URL=https://your-org--fish-tts-server.modal.run
OPENCLAW_GATEWAY_URL=http://your-server:3117
BOT_SECRET=your-secret-for-speak-endpoint
```

### 3. Run on Synology

```bash
docker compose up -d
```

Or run directly:
```bash
pip install -e .
python bot.py
```

### 4. Access from anywhere

Open `https://your-domain.com` in your browser (phone, laptop, anything).
Click Connect → talk to Aris.

## SPEAKING TO YOU (Aris-initiated voice)

Aris can push voice to a connected browser via the `/speak` endpoint:

```bash
curl -X POST http://localhost:7860/speak \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret" \
  -d '{"text": "Morning John. Dominius deadline is in 2 days. How are we looking?"}'
```

The bot synthesizes via Fish Speech on Modal and plays it through the connected browser.

## COST

- Modal A10G GPU: ~$0.80/hour, **only when active**
- `scaledown_window=15`: GPU shuts down 15 sec after last request
- Typical usage (30 min/day of conversation): ~$0.40/day
- Cold start: ~10-15 sec for first request after idle (model loading)

## TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| 15 sec delay on first message | Normal — Modal loading models. Subsequent messages instant |
| No audio in browser | Check mic/speaker permissions. Use HTTPS (required for WebRTC) |
| `/speak` not working | Check `BOT_SECRET` matches between .env and curl command |
| Modal timeout | First deploy downloads models (~8GB). Wait 5 min. |
