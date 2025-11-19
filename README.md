# Burst Classifier POC
This repository implements a proof-of-concept machine learning pipeline for classifying burst-type patterns in audio segments into three classes:

- **b**: single burst

- **mb**: multiple burst

- **h**: harmonic

The goal is not to build the optimal model, but to show a complete ML **training → evaluation → deployment → feedback** loop pipeline, similar to real-world production MLOps systems.

## Quick Summary
Proof of Concept repository with CLI tools for train, evaluate, predict, serve and a Streamlit annotator UI. Training writes `artifacts/` with `metadata.json`, model checkpoints, `dataset_manifest_*.json`, evaluation reports and plots. A `Makefile` orchestrates common flows.

This repository provides:

- **Training pipeline**: CNN + spectrogram preprocessing

- **Evaluation pipeline**: metrics, confusion matrix, PR curves, calibration

- **Batch inference**: used for pre-labelling

- **Model server**: FastAPI wrapper

- **Annotation UI**: Streamlit app for human-in-the-loop corrections

- **MLOps tracking**: MLflow / ClearML / local JSON tracker

- **Reproducibility**: Makefile, Docker, dataset manifests, metadata

# Project Structure
- `src/`: main python code

    - `cli.py`: single CLI entrypoint wrapping train / eval / predict / serve

    - `train.py`: training code (kfold / lofo / none)

    - `dataset.py`: label parsing and SegmentDataset

    - `models/`: includes models to be used for training (eg: SmallCNN model)

    - `predict.py`: inference & batch predict

    - `eval.py`: metrics, plots, PR/ECE utilities

    - `tracker.py`: small wrapper: none | mlflow | clearml

    - `serve.py`:FastAPI serve wrapper

    - `label_ui.py`: Streamlit annotator UI

- `data/`:audio files + .txt labels (format: start end label whitespace-separated)

- `Makefile`: convenience targets (train, evaluate, prelabel, label-ui, serve, docker)

- `docker-compose.*.yml`: examples (mlflow, streamlit, serve)

- `mlflow/:` optional docker image for mlflow server

# Quickstart
1- Create virtualenv and install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2- Train (quick demo - 1 epoch):

```bash
make train DATA_DIR=./data ARTIFACTS_DIR=./artifacts BATCH_SIZE=32 EPOCHS=1
# or:
python3 src/cli.py train --data-dir ./data --artifacts-dir ./artifacts --epochs 1 --batch-size 32
```
3- Evaluate
```bash
make evaluate EVAL_DATA=./eval_data ARTIFACTS_DIR=./artifacts
# or:
python3 src/cli.py evaluate --model ./artifacts --data-dir ./eval_data --out-dir ./artifacts/eval
```
4- Prelabel a folder (batch predict):
```bash
make prelabel TEST_DATA=./test-data ARTIFACTS_DIR=./artifacts_1
# or:
python3 src/cli.py predict --model ./artifacts_1 --audio ./test-data --out ./artifacts/prelabels.json --batch --device cpu # (or cuda)
```
5- Start Streamlit UI for annotators:
```bash
make label-ui # to run a dockerized UI
# or
streamlit run src/label_ui.py
```
6- Serve the model locally
```bash
make serve MODEL_DIR=./artifacts_1 DEVICE=cpu
# or run the serve CLI:
python3 src/cli.py serve --model ./artifacts_1 --device cpu --port 8000
```

7- Docker (build & run):
```bash
make docker-build
make docker-run MODEL_DIR=./artifacts_1 PORT=8000 HOST=0.0.0.0
```
# Evaluation & Gating
Model evaluation is run with make evaluate or the equivalent CLI command (cli.py evaluate). This produces eval_report.json along with plots such as the confusion matrix, PR curves, and calibration curves. Metrics include per-class precision/recall/F1, macro F1, and ECE.

Gating is handled by a separate command, make gate, which reads the produced evaluation report and applies a simple single-metric threshold. If the metric falls below the configured threshold, the gate exits with a non-zero status. In future versions, the gate may become richer and support multi-metric rules, weighted conditions, or class-specific constraints.

# Deployment & Traceability
A FastAPI server exposes ``/predict``, ``batch_predict`` and ``/health`` endpoints and is container-ready. Each training run writes ``metadata.json`` with git commit, dataset manifest hashes, environment setup, hyperparameters, epoch logs, and evaluation metrics. When **MLflow/ClearML** is enabled, runs and artifacts are logged to a registry.

# Semi-automated Labelling & Feedback Loop
Batch prelabeling produces ``prelabels.json`` with start_seconds/end_seconds, predicted class, and probability. The Streamlit annotator shows waveform + suggestions; annotators can accept or edit class and boundaries. Edits are saved with provenance (annotator id, timestamp, original suggestion). Corrected labels are collected into new dataset snapshots.

# Additional considerations
This section outlines future extensions that are not yet implemented but serve as guidance for further enhancements.
## Parallel GPUs & multi-GPU training / inference
Future expansions may include support for parallel GPU usage and multi‑GPU workflows. Introducing distributed training approaches, such as Data Parallelism or integration with frameworks like Accelerate or DeepSpeed—would significantly shorten training times for large models. The goal is to eventually allow configurable GPU counts, backend settings, precision modes, and resource-aware training strategies, improving both performance and flexibility.
## Support for Additional Parameters
Configurable parameters will be expanded in the future to give users more granular control over training, inference, and deployment. This would include core hyperparameters such as weight decay, schedulers and options for precision modes like FP16 or BF16. Infrastructure‑related parameters—such as worker counts, memory pinning, distributed settings, and checkpoint intervals—could also be introduced. Centralizing these in a config file with CLI overrides would enhance reproducibility and simplify experimentation.
## ONNX & TorchScript support
The project may eventually include support for exporting models to ONNX and TorchScript, enabling portability across different runtimes and deployment environments. Providing export utilities, validation steps, and compatibility checks would allow users to move models from research environments into production pipelines with greater ease.
## Hyperparameter optimization
o improve model performance systematically, hyperparameter optimization tools such as Optuna, Ray Tune, or Weights & Biases Sweeps could be integrated in the future. Rather than relying on manual trial‑and‑error, automated search spaces and guided optimization would make experimentation more efficient. As part of this effort, example studies, visualizations, and metric‑logging integrations may be added to simplify the tuning workflow.
## Authentication & RBAC (Streamlit, Metaflow and Serve API)
Looking forward, authentication and role‑based access control (RBAC) could be incorporated into the Streamlit UI, Metaflow Server, and Serve API to enhance security. Adding support for JWT or OAuth2 authentication, along with structured roles (e.g., admin, developer, viewer), would protect sensitive operations and datasets.
## Data Versioning (DVC)
Although dataset manifests currently exist, future integration of DVC would allow the project to track large datasets, audio files, and artifacts more comprehensively. DVC would provide reproducible dataset snapshots, remote storage support, and consistent versioning of data used during training and evaluation. This would strengthen the overall reproducibility and traceability of the project.
## Weight & Biases Integration
Although ClearML and MLflow are currently supported, future integration with Weights & Biases (W&B) would offer an additional experiment‑tracking option. Making W&B an optional logging backend would give users more flexibility in how they monitor and analyze experiments.
## Alternative Container Runtimes (podman, nerdctl, colima)
To accommodate users working in different development environments, future work may include documentation and compatibility testing for alternative container runtimes such as Podman, Nerdctl, and Colima. Providing equivalent commands, troubleshooting guidance, and runtime‑specific notes would improve accessibility for developers who do not use Docker.

In addition, Docker presents a few limitations that motivate exploring alternatives. Docker relies on a long‑running background daemon, which can cause performance issues, increased resource usage, or permission‑related complications in certain environments. Furthermore, modern Kubernetes distributions no longer use the Docker daemon as their underlying container engine; instead, they rely on containerd (through CRI‑O or other CRI‑compliant runtimes). Aligning with these runtimes in the future would improve compatibility with Kubernetes-native workflows, reduce unnecessary abstraction layers, and simplify deployments in cloud‑native infrastructures.
