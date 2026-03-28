.PHONY: setup run docker-build docker-up docker-down deploy deploy-local

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e .
	@echo ""
	@echo "Done. Copy .env.example to .env and fill in your keys."
	@echo "  source .venv/bin/activate"
	@echo "  make run"

run:
	python bot.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

deploy:
	modal deploy server.py
	@echo ""
	@echo "Done. Update VOICE_SERVER_URL in .env with the Modal URL."

deploy-local:
	python server.py
