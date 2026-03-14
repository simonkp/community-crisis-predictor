.PHONY: install collect collect-synthetic features train evaluate all test clean

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
