FROM node:22-slim AS openclaw
RUN npm install -g openclaw@latest

FROM python:3.12-slim

WORKDIR /app

# Copy OpenClaw CLI from node stage
COPY --from=openclaw /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=openclaw /usr/local/bin/openclaw /usr/local/bin/openclaw

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libxcb1 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (+ modal for auto-deploy)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt modal

# Copy application
COPY . .
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["bash", "./entrypoint.sh"]
