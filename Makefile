.PHONY: install collect collect-synthetic features train evaluate all all-synthetic \
        test clean copy-models serve-local prepare-deploy prepare-deploy-synthetic

# ── Core pipeline ──────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

collect:
	python -m src.pipeline.run_collect --config config/default.yaml

collect-synthetic:
	python -m src.pipeline.run_collect --config config/default.yaml --synthetic

features:
	python -m src.pipeline.run_features --config config/default.yaml

train:
	python -m src.pipeline.run_train --config config/default.yaml

evaluate:
	python -m src.pipeline.run_evaluate --config config/default.yaml

all:
	python -m src.pipeline.run_all --config config/default.yaml

all-synthetic:
	python -m src.pipeline.run_all --config config/default.yaml --synthetic

test:
	pytest tests/ -v

clean:
	rm -rf data/raw/* data/processed/* data/features/* data/models/* data/reports/*

# ── Deployment helpers ─────────────────────────────────────────────────────────

## Copy trained model artifacts from data/models/ into serving/models/.
## Run this after `make train` + `make evaluate` to prepare for git push + deploy.
copy-models:
	mkdir -p serving/models
	@echo "Copying XGB models..."
	@echo "Copying XGB models (lowercase names for Linux/Render compatibility)..."
	@for sub in depression anxiety suicidewatch lonely mentalhealth; do \
	  src_lower="data/models/$${sub}_xgb.pkl"; \
	  src_orig="data/models/SuicideWatch_xgb.pkl"; \
	  if [ -f "$$src_lower" ]; then \
	    cp "$$src_lower" "serving/models/$${sub}_xgb.pkl" && echo "  copied $$src_lower"; \
	  elif [ "$$sub" = "suicidewatch" ] && [ -f "$$src_orig" ]; then \
	    cp "$$src_orig" "serving/models/$${sub}_xgb.pkl" && echo "  copied $$src_orig as $$sub"; \
	  else echo "  skip (not found): $$src_lower"; fi; \
	done
	@echo "Copying LSTM models..."
	@for sub in depression anxiety suicidewatch lonely mentalhealth; do \
	  src_lower="data/models/$${sub}_lstm.pt"; \
	  src_orig="data/models/SuicideWatch_lstm.pt"; \
	  if [ -f "$$src_lower" ]; then \
	    cp "$$src_lower" "serving/models/$${sub}_lstm.pt" && echo "  copied $$src_lower"; \
	  elif [ "$$sub" = "suicidewatch" ] && [ -f "$$src_orig" ]; then \
	    cp "$$src_orig" "serving/models/$${sub}_lstm.pt" && echo "  copied $$src_orig as $$sub"; \
	  else echo "  skip (not found): $$src_lower"; fi; \
	done
	@echo "Copying feature stats..."
	@for sub in depression anxiety suicidewatch lonely mentalhealth; do \
	  src_lower="data/models/$${sub}_feature_stats.json"; \
	  src_orig="data/models/SuicideWatch_feature_stats.json"; \
	  if [ -f "$$src_lower" ]; then \
	    cp "$$src_lower" "serving/models/$${sub}_feature_stats.json" && echo "  copied $$src_lower"; \
	  elif [ "$$sub" = "suicidewatch" ] && [ -f "$$src_orig" ]; then \
	    cp "$$src_orig" "serving/models/$${sub}_feature_stats.json" && echo "  copied $$src_orig as $$sub"; \
	  else echo "  skip (not found): $$src_lower"; fi; \
	done
	@echo "Copying eval_results.json..."
	@[ -f "data/models/eval_results.json" ] && cp data/models/eval_results.json serving/models/eval_results.json && echo "  copied eval_results.json" || echo "  skip (not found): data/models/eval_results.json"
	@echo "Copying SHAP CSVs..."
	@for sub in depression anxiety suicidewatch lonely mentalhealth; do \
	  src_lower="data/reports/$${sub}/shap.csv"; \
	  src_orig="data/reports/SuicideWatch/shap.csv"; \
	  if [ -f "$$src_lower" ]; then \
	    cp "$$src_lower" "serving/models/$${sub}_shap.csv" && echo "  copied $$src_lower"; \
	  elif [ "$$sub" = "suicidewatch" ] && [ -f "$$src_orig" ]; then \
	    cp "$$src_orig" "serving/models/$${sub}_shap.csv" && echo "  copied $$src_orig as $$sub"; \
	  else echo "  skip (not found): $$src_lower"; fi; \
	done
	@echo "copy-models done."

## Start the FastAPI inference service locally at http://localhost:8000
## Visit http://localhost:8000/docs for interactive API docs.
serve-local:
	cd serving && uvicorn main:app --reload --port 8000

## Run full pipeline with REAL data then copy artifacts. After this, run:
##   git add . && git commit -m "Update model artifacts" && git push
## Render.com and Streamlit Cloud auto-redeploy on push.
prepare-deploy: all copy-models
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
prepare-deploy-synthetic: all-synthetic copy-models
	@echo ""
	@echo "=========================================="
	@echo "Synthetic deploy artifacts ready."
	@echo "Next steps:"
	@echo "  git add ."
	@echo "  git commit -m 'Update model artifacts (synthetic)'"
	@echo "  git push"
	@echo "=========================================="
