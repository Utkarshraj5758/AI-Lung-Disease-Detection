# Common tasks. Pass extra options with ARGS, e.g. make train ARGS="--set train.epochs_stage2=5".
# For a CPU-only install: make setup TORCH_INDEX=https://download.pytorch.org/whl/cpu

PYTHON ?= python
PIP := $(PYTHON) -m pip
IMAGE ?= lungxray
ARGS ?=

.PHONY: help setup lint format test train evaluate docs app api docker docker-api clean

help:  ## list the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-11s %s\n", $$1, $$2}'

setup:  ## install the package with dev, api and train extras (editable)
ifdef TORCH_INDEX
	$(PIP) install torch torchvision --index-url $(TORCH_INDEX)
endif
	$(PIP) install -e ".[dev,api,train]"

lint:  ## ruff lint and format check
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:  ## apply ruff formatting and safe fixes
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

test:  ## run the tests with coverage (no dataset, GPU or network needed)
	$(PYTHON) -m pytest --cov=lungxray --cov=api --cov-report=term-missing --cov-fail-under=80 $(ARGS)

train:  ## two-stage fine-tune + calibration -> artifacts/best.pt (needs dataset/)
	$(PYTHON) -m lungxray.train $(ARGS)

evaluate:  ## test-set metrics with CIs and plots -> reports/, then refresh the docs
	$(PYTHON) -m lungxray.evaluate $(ARGS)
	$(PYTHON) scripts/render_results.py

docs:  ## rewrite the results in README.md and docs/MODEL_CARD.md from reports/metrics.json
	$(PYTHON) scripts/render_results.py

app:  ## Streamlit app on http://localhost:8501
	$(PYTHON) -m streamlit run app/streamlit_app.py $(ARGS)

api:  ## FastAPI service on http://localhost:8000 (docs at /docs)
	$(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8000 $(ARGS)

docker:  ## build the Streamlit image
	docker build -t $(IMAGE) .

docker-api:  ## build the API image
	docker build --build-arg SERVICE=api -t $(IMAGE)-api .

clean:  ## remove caches and build files (keeps runs/, artifacts/, reports/)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
