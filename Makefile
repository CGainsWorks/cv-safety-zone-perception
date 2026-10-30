# Safety-zone perception.  NOT A CERTIFIED SAFETY FUNCTION -- read
# docs/SAFETY_DISCLAIMER.md before anything else.
#
# Three classes of target and the distinction matters when reading the README:
#
#   FAST      seconds to a couple of minutes, no dataset and no GPU.  Everything
#             CI runs, plus the synthetic demo.
#               make check  make test  make lint  make demo  make zones
#
#   HEAVY     needs the detector weights and for the detection metrics the COCO
#             val2017 images.  `make experiments` is the target that regenerates
#             every number the README quotes; roughly five minutes on eight CPU
#             cores, most of it inference.
#               make model  make data  make experiments
#
#   GPU       needs an NVIDIA GPU.  Not run by CI, which has no GPU.
#             `make tensorrt-desktop` builds the FP32, FP16 and INT8 engines in
#             NVIDIA's TensorRT container, scores them on COCO, times the
#             pipeline and writes results/published/tensorrt.json; it ran on a
#             desktop RTX 4060, never on a Jetson.  See docs/tensorrt.md.
#               make tensorrt-desktop  make engine  make trt-check
#
# `make check` is exactly what CI runs.

SHELL := /bin/sh
PYTHON ?= python
VENV ?= .venv
VENV_PY := $(VENV)/bin/python
ifeq ($(wildcard $(VENV_PY)),)
VENV_PY := $(VENV)/Scripts/python.exe
endif
ifeq ($(wildcard $(VENV_PY)),)
VENV_PY := $(PYTHON)
endif

RESULTS ?= results
PUBLISHED := $(RESULTS)/published
FIGURES := docs/figures
CAN ?= auto
DURATION ?= 60
REPEATS ?= 30

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  python: $(VENV_PY)"

# ---------------------------------------------------------------------------
# Fast
# ---------------------------------------------------------------------------

.PHONY: check
check: lint test demo-fast fmea docs-check  ## Everything CI runs

.PHONY: lint
lint:  ## flake8 over the whole repository
	$(VENV_PY) -m flake8 src/ tests/ scripts/ analysis/

.PHONY: test
test:  ## The test suite
	$(VENV_PY) -m pytest tests/ -q --durations=10

.PHONY: demo
demo:  ## The whole pipeline over the synthetic cell, on a CAN bus
	@mkdir -p $(RESULTS)
	$(VENV_PY) scripts/run_demo.py --can $(CAN) \
	  --out $(RESULTS)/demo.json --timeline $(RESULTS)/demo_timeline.csv

.PHONY: demo-fast
demo-fast:  ## A short demo, for CI
	@mkdir -p $(RESULTS)
	$(VENV_PY) scripts/run_demo.py --frames 90 --can $(CAN) \
	  --out $(RESULTS)/demo_short.json --quiet

.PHONY: zones
zones:  ## Print and plot the cell, to check a zone file by eye
	@mkdir -p $(FIGURES)
	$(VENV_PY) scripts/show_zones.py --plot $(FIGURES)/zones.png

.PHONY: calibrate
calibrate:  ## Calibration self-check: recover a known transform
	@mkdir -p $(RESULTS)
	$(VENV_PY) scripts/calibrate.py --self-check --json $(RESULTS)/calibration_self_check.json

.PHONY: fmea
fmea:  ## Check the FMEA table is self-consistent and emit it as JSON
	@mkdir -p $(RESULTS)
	$(VENV_PY) analysis/make_fmea_summary.py --out $(RESULTS)/fmea.json --quiet

.PHONY: docs-check
docs-check:  ## Every number in the README must come from a published result
	$(VENV_PY) scripts/check_docs.py

.PHONY: benchmark
benchmark:  ## Time the individual stages on this machine
	@mkdir -p $(RESULTS)
	$(VENV_PY) scripts/benchmark.py --out $(RESULTS)/benchmark.json

# ---------------------------------------------------------------------------
# Heavy
# ---------------------------------------------------------------------------

.PHONY: install
install:  ## Create .venv and install the runtime requirements
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt

.PHONY: model
model:  ## Download YOLOX-Tiny and export it to ONNX (installs torch and yolox)
	$(VENV_PY) -m pip install -r requirements-export.txt
	$(VENV_PY) -m pip install --no-deps --no-build-isolation -r requirements-yolox.txt
	$(VENV_PY) scripts/fetch_model.py

.PHONY: data
data:  ## Download COCO val2017 (about 1.1 GB)
	$(VENV_PY) scripts/fetch_data.py

.PHONY: experiments
experiments: detection geometry latency fmea  ## Regenerate every published number
	@echo
	@echo "published results are in $(PUBLISHED)/"

.PHONY: detection
detection:  ## Person-detection recall and miss rate on COCO val2017
	@mkdir -p $(PUBLISHED)
	$(VENV_PY) analysis/detection_metrics.py --out $(PUBLISHED)/detection_val2017.json

.PHONY: int8
int8:  ## FP16 and INT8 ONNX graphs, scored on COCO val2017 (export toolchain, a GPU for FP16)
	@mkdir -p $(PUBLISHED)
	$(VENV_PY) scripts/fetch_model.py --fp16
	$(VENV_PY) scripts/fetch_calibration.py
	$(VENV_PY) scripts/quantize_onnx.py
	cp models/quantisation.json $(PUBLISHED)/quantisation.json
	$(VENV_PY) analysis/detection_metrics.py --backend onnx --model models/yolox_tiny_fp16.onnx \
	  --out $(PUBLISHED)/detection_val2017_fp16.json
	$(VENV_PY) analysis/detection_metrics.py --backend onnx --model models/yolox_tiny_int8.onnx \
	  --out $(PUBLISHED)/detection_val2017_int8.json

.PHONY: compare
compare:  ## Put the published numbers beside the YOLOv8n ones they replaced
	$(VENV_PY) analysis/compare_detectors.py

.PHONY: geometry
geometry:  ## Fusion accuracy and zone confusion on the synthetic cell
	@mkdir -p $(PUBLISHED)
	$(VENV_PY) analysis/geometry_accuracy.py --duration $(DURATION) \
	  --out $(PUBLISHED)/geometry_accuracy.json

.PHONY: latency
latency:  ## End-to-end latency budget, with the real model and without it
	@mkdir -p $(PUBLISHED)
	$(VENV_PY) analysis/latency_budget.py --detector onnx_timing --repeats $(REPEATS) \
	  --out $(PUBLISHED)/latency_onnx.json
	$(VENV_PY) analysis/latency_budget.py --detector simulated --repeats $(REPEATS) \
	  --out $(PUBLISHED)/latency_simulated.json

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

TRT_TAG ?= 25.06-py3
COCO_ROOT ?= $(HOME)/datasets/coco

.PHONY: tensorrt-desktop
tensorrt-desktop:  ## Build, score and time TensorRT engines on this GPU (Docker)
	docker build -f docker/Dockerfile.tensorrt --build-arg TRT_TAG=$(TRT_TAG) \
	  -t szp:tensorrt .
	docker run --rm --gpus all --network none \
	  -e TRT_IMAGE=nvcr.io/nvidia/tensorrt:$(TRT_TAG) -e SZP_COCO_ROOT=/coco \
	  -v "$(CURDIR):/work" -v "$(COCO_ROOT):/coco:ro" \
	  szp:tensorrt python3 scripts/measure_tensorrt.py
	$(VENV_PY) analysis/tensorrt_table.py

.PHONY: engine
engine:  ## Build a TensorRT engine (needs an NVIDIA GPU; never committed)
	$(VENV_PY) scripts/export_tensorrt.py --precision fp16

.PHONY: trt-check
trt-check:  ## Check the TensorRT engine agrees with ONNX Runtime
	$(VENV_PY) -m pytest tests/test_backends.py -q

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

.PHONY: docker
docker:  ## Build the runtime image
	docker build -f docker/Dockerfile -t szp:latest .

.PHONY: docker-demo
docker-demo:  ## Run the demo in Docker, on a vcan interface if one exists
	docker compose -f docker/docker-compose.yml up --build --abort-on-container-exit

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

.PHONY: clean
clean:  ## Remove run outputs, keeping results/published
	rm -rf $(RESULTS)/*.json $(RESULTS)/*.csv .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

.PHONY: distclean
distclean: clean  ## Also remove the venv and the model weights
	rm -rf $(VENV) models
