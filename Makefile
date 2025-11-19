MODEL_DIR ?= ./models/v1
DATA_DIR ?= ./data
EVAL_DATA ?= ./eval_data
TEST_DATA ?= ./test_data
ARTIFACTS_DIR ?= ./artifacts_1
IMAGE ?= burst-classifier-serve
TAG ?= main
PORT ?= 8000
HOST ?= 0.0.0.0

GATE_METRIC ?= accuracy
GATE_THRESHOLD ?= 0.65

DEVICE ?= cpu
BATCH_SIZE ?= 64
LEARNING_RATE ?= 0.001
EPOCHS ?= 1

TRACKER ?= mlflow
TRACKER_PROJECT ?= Burst_Classifier_POC

.PHONY: all train evaluate gate prelabel label-ui label-ui-stop snapshot-labels retrain mlflow-start mlflow-stop docker-build docker-run docker-stop docker-push docker-compose-up docker-compose-down serve demo clean help

all: train evaluate

# --- Training / evaluation / gating ---
train:
	@echo "=== TRAIN ==="
	python3 src/cli.py train --data-dir $(DATA_DIR) --artifacts-dir $(ARTIFACTS_DIR) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --lr $(LEARNING_RATE) --tracker $(TRACKER) --tracker-project $(TRACKER_PROJECT)

evaluate:
	@echo "=== EVALUATE ==="
	python3 src/cli.py evaluate --model $(ARTIFACTS_DIR) --data-dir $(EVAL_DATA) --out-dir $(ARTIFACTS_DIR)/eval --tracker $(TRACKER) --tracker-project $(TRACKER_PROJECT)

gate: evaluate
	@echo "=== GATE (threshold: $(GATE_THRESHOLD)) ==="
	python3 src/gate.py --eval-dir $(ARTIFACTS_DIR)/eval/ --metric $(GATE_METRIC) --threshold $(GATE_THRESHOLD)

# --- Prelabel / Annotator UI ---
prelabel:
	@echo "=== PRELABEL (batch) ==="
	python3 src/cli.py predict --model $(ARTIFACTS_DIR) --audio $(TEST_DATA) --out $(ARTIFACTS_DIR)/prelabels.json --device $(DEVICE) --tracker $(TRACKER) --tracker-project $(TRACKER_PROJECT) --tracker-task prelabeling-snapshots --batch

label-ui:
	@echo "=== START LABEL UI ==="
	docker compose -f docker-compose.streamlit.yml up -d

label-ui-stop:
	@echo "=== STOP LABEL UI ==="
	docker compose -f docker-compose.streamlit.yml down

# Mlflow, use only if tracker set to mlflow
mlflow-start:
	docker compose -f docker-compose.mlflow.yml up -d

mlflow-stop:
	docker compose -f docker-compose.mlflow.yml down

# --- Docker / serve ---
docker-build:
	@echo "=== DOCKER BUILD ==="
	docker build -t $(IMAGE):$(TAG) .

docker-run: docker-build
	@echo "=== DOCKER RUN (mount model dir) ==="
	mkdir -p $(MODEL_DIR)
	docker run --rm -d --name $(IMAGE)-run -p $(PORT):8000 -v $(PWD)/$(MODEL_DIR):/models:ro $(IMAGE):$(TAG) --model /models --device $(DEVICE) --port $(PORT)

docker-stop:
	@echo "=== DOCKER STOP ==="
	-docker stop $(IMAGE)-run || true
	-docker rm $(IMAGE)-run || true

docker-compose-up:
	@echo "=== DOCKER-COMPOSE UP ==="
	docker compose up -d

docker-compose-down:
	@echo "=== DOCKER-COMPOSE DOWN ==="
	docker compose down

serve:
	@echo "=== RUN LOCAL PYTHON SERVER ==="
	python3 src/cli.py serve --model $(MODEL_DIR) --device $(DEVICE) --host $(HOST) --port $(PORT)

help:
	@echo "Makefile targets:"
	@echo "  make train              # run training (POC)"
	@echo "  make evaluate           # evaluate model on EVAL_DATA"
	@echo "  make gate               # evaluate + gate using metrics JSON"
	@echo "  make prelabel           # run bulk prelabeling on TEST_DATA"
	@echo "  make label-ui           # start a dockerized streamlit annotator UI"
	@echo "  make label-ui-stop      # stop a dockerized streamlit annotator UI app"
	@echo "  make snapshot-labels    # create local labels snapshot"
	@echo "  make mlflow-start       # start mlflow server if tracker set to mlflow"
	@echo "  make mlflow-stop        # stop mlflow server if tracker set to mlflow"
	@echo "  make docker-compose-up  # run docker-compose up for containerized serve API"
	@echo "  make docker-compose-up  # run docker-compose down for containerized serve API"
	@echo "  make docker-build       # build docker image"
	@echo "  make docker-run         # run docker image mounting model dir"
	@echo "  make docker-stop        # stop docker container"
	@echo "  make serve              # run python server locally"
