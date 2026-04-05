.PHONY: install collect collect-synthetic features train evaluate all all-synthetic \
        test clean serve-local prepare-deploy prepare-deploy-synthetic

# Interpreter resolution order (override anytime with `make PYTHON=/path/to/python <target>`):
# 1) project-local virtualenv, 2) python3 on PATH, 3) python on PATH.
ifneq ("$(wildcard .venv/bin/python)","")
PYTHON ?= .venv/bin/python
else ifneq ("$(shell command -v python3 2>/dev/null)","")
PYTHON ?= python3
else
PYTHON ?= python
endif

# ── Core pipeline ──────────────────────────────────────────────────────────────
install:
	$(PYTHON) -m pip install -e ".[dev]"

collect:
	$(PYTHON) -m src.pipeline.run_collect --config config/default.yaml

collect-synthetic:
	$(PYTHON) -m src.pipeline.run_collect --config config/default.yaml --synthetic

features:
	$(PYTHON) -m src.pipeline.run_features --config config/default.yaml

train:
	$(PYTHON) -m src.pipeline.run_train --config config/default.yaml

evaluate:
	$(PYTHON) -m src.pipeline.run_evaluate --config config/default.yaml

all:
	$(PYTHON) -m src.pipeline.run_all --config config/default.yaml

all-synthetic:
	# $(PYTHON) -m src.pipeline.run_all --config config/default.yaml --synthetic
	$(PYTHON) -m src.pipeline.run_all --config config/default.yaml --synthetic --skip-topics --skip-search --force

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	$(PYTHON) -c "import pathlib, shutil; dirs=['data/raw','data/processed','data/features','data/models','data/reports']; [pathlib.Path(d).mkdir(parents=True, exist_ok=True) for d in dirs]; [shutil.rmtree(p) if p.is_dir() else p.unlink() for d in dirs for p in pathlib.Path(d).iterdir()]"

# ── Deployment helpers ─────────────────────────────────────────────────────────

## Start the FastAPI inference service locally at http://localhost:8000
## Visit http://localhost:8000/docs for interactive API docs.
serve-local:
	cd serving && uvicorn main:app --reload --port 8000

## Run full pipeline with REAL data. After this, run:
##   git add . && git commit -m "Update model artifacts" && git push
## Render.com and Streamlit Cloud auto-redeploy on push.
prepare-deploy: all
	@echo ""
	@echo "=========================================="
	@echo "Deploy artifacts ready."
	@echo "Next steps:"
	@echo "  git add ."
	@echo "  git commit -m 'Update model artifacts'"
	@echo "  git push"
	@echo "Render.com and Streamlit Cloud auto-redeploy on push."
	@echo "=========================================="

## Same as prepare-deploy but uses synthetic data (no Zenodo download needed).
## Useful for quick CI rebuilds or if you don't have the Zenodo cache locally.
prepare-deploy-synthetic: all-synthetic
	@echo ""
	@echo "=========================================="
	@echo "Synthetic deploy artifacts ready."
	@echo "Next steps:"
	@echo "  git add ."
	@echo "  git commit -m 'Update model artifacts (synthetic)'"
	@echo "  git push"
	@echo "=========================================="
