# P.O.W.E.R. task runner. On Windows without `make`, run the `uv run ...` lines directly.
SRC ?= palo_alto

.PHONY: sync test leakage-test lint data features weather panel ingest resample report

sync:
	uv sync

test:
	uv run pytest -q

leakage-test:
	uv run pytest -q -m leakage

lint:
	uv run ruff check .

ingest:
	uv run power-ml ingest --source $(SRC)

resample:
	uv run power-ml resample --source $(SRC)

weather:
	uv run power-ml weather --source $(SRC)

panel:
	uv run power-ml panel --source $(SRC)

features:
	uv run power-ml features --source $(SRC)

# Full data pipeline once data/raw/<source file> is in place.
data: ingest resample weather panel features

report:
	cd scripts/report && npm install --silent && npm run report
