.PHONY: setup dev docker-up docker-down test test-all lint format clean

setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt
	@[ -f .env ] || cp .env.example .env
	@echo "Setup complete. Activate with: source .venv/bin/activate"

dev:
	uvicorn app.main:app --reload --port 8000

docker-up:
	docker compose -f docker-compose.rag.yml up --build -d

docker-down:
	docker compose -f docker-compose.rag.yml down

test:
	pytest tests/unit/ -x -q

test-all:
	pytest tests/

lint:
	black --check app/ tests/
	flake8 app/ tests/ --max-line-length=100
	mypy app/ --ignore-missing-imports

format:
	black app/ tests/

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
