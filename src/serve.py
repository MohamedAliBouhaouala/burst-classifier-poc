import os
import tempfile
import shutil
import argparse
import logging
import traceback
from typing import List, Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import json
import uuid
import sys
import time
import torch

from helpers.constants import LABEL_MAP, INV_LABEL_MAP

from helpers.common import load_model
from utils.utils_predict import predict

logger = logging.getLogger("serve")
logging.basicConfig(level=logging.INFO)


def _safe_save_upload(upload: UploadFile, suffix=".wav") -> str:
    """
    Save UploadFile to a temp file and return path.
    Caller must unlink the file.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as out_f:
        shutil.copyfileobj(upload.file, out_f)
    return tmp_path


def _format_prediction(
    audio_path: str, win: Dict[str, Any], top_k: int = 1
) -> Dict[str, Any]:
    """
    Convert window prediction (win) to desired output fields.
    Uses LABEL_MAP & INV_LABEL_MAP for consistent ordering/mapping.
    Expected win keys: 'start','end','label','probs' (list of floats in the same order as LABEL_MAP)
    Produces:
      audio_file, start_seconds, end_seconds, label, probability, b_probability, h_probability, mb_probability
    plus top_predictions when top_k>1.
    """
    probs = win.get("probabilities", None)
    C = len(LABEL_MAP)
    if probs is None:
        probs_arr = np.zeros(C, dtype=float)
    else:
        probs_arr = np.array(probs, dtype=float)

    # pad/trim to C
    if probs_arr.size < C:
        probs_arr = np.pad(
            probs_arr, (0, C - probs_arr.size), "constant", constant_values=0.0
        )
    elif probs_arr.size > C:
        probs_arr = probs_arr[:C]

    pred_idx = int(np.argmax(probs_arr)) if probs_arr.size > 0 else 0
    predicted_label = INV_LABEL_MAP.get(pred_idx, win.get("label", ""))

    top_pred_conf = float(probs_arr[pred_idx]) if probs_arr.size > 0 else 0.0

    def prob_for(name: str) -> float:
        idx = LABEL_MAP.get(name)
        if idx is None or idx >= probs_arr.size:
            return 0.0
        return float(probs_arr[idx])

    out = {
        "audio_file": os.path.basename(audio_path),
        "start_seconds": float(win.get("start_seconds", 0.0)),
        "end_seconds": float(win.get("end_seconds", 0.0)),
        "label": predicted_label,
        "probability": top_pred_conf,
        "b_probability": prob_for("b"),
        "h_probability": prob_for("h"),
        "mb_probability": prob_for("mb"),
    }

    if top_k and top_k > 1:
        pairs = [
            (INV_LABEL_MAP.get(i, str(i)), float(p))
            for i, p in enumerate(probs_arr.tolist())
        ]
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k]
        out["top_predictions"] = [
            {"label": p[0], "probability": p[1]} for p in pairs_sorted
        ]

    return out


def server(
    model: torch.nn.Module,
    meta: Dict[str, Any],
    allowed_origins: Optional[List[str]] = None,
    default_device: str = "cpu",
) -> FastAPI:
    middleware = (
        [Middleware(CORSMiddleware, allow_origins=allowed_origins)]
        if allowed_origins
        else None
    )
    app = FastAPI(title="Burst Classifier POC - Serve", middleware=middleware)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    async def predict_endpoint(file: UploadFile = File(...)):
        tmp_path = None
        request_id = str(uuid.uuid4())
        start_ts = time.time()
        try:
            tmp_path = _safe_save_upload(
                file, suffix=os.path.splitext(file.filename)[1] or ".wav"
            )

            # call predict helper. Be robust if predict signature changed: try with top_k then without.
            preds = predict(model=model, metadata=meta, audio_path=tmp_path)

            # preds: list of windows dicts
            out = []
            for w in preds:
                out.append(_format_prediction(tmp_path, w, top_k=1))

            latency = time.time() - start_ts
            logger.info(
                json.dumps(
                    {
                        "request_id": request_id,
                        "audio_file": os.path.basename(tmp_path),
                        "n_predictions": len(out),
                        "latency_seconds": latency,
                    }
                )
            )

            return {"predictions": out, "n_predictions": len(out)}
        except Exception as e:
            logger.error(f"Prediction failed: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    @app.post("/batch_predict")
    async def batch_predict_endpoint(files: List[UploadFile] = File(...)):
        tmp_paths: List[str] = []
        all_results: List[Dict[str, Any]] = []
        request_id = str(uuid.uuid4())
        start_ts = time.time()
        try:
            # save all uploads temporarily
            for f in files:
                p = _safe_save_upload(
                    f, suffix=os.path.splitext(f.filename)[1] or ".wav"
                )
                tmp_paths.append(p)

            # sequentially predict per file to bound memory
            for p in tmp_paths:
                preds = predict(model=model, metadata=meta, audio_path=p)
                for w in preds:
                    all_results.append(_format_prediction(p, w, top_k=1))

            latency = time.time() - start_ts
            logger.info(
                json.dumps(
                    {
                        "request_id": request_id,
                        "n_files": len(tmp_paths),
                        "n_predictions": len(all_results),
                        "latency_seconds": latency,
                    }
                )
            )

            return {"predictions": all_results, "n_predictions": len(all_results)}
        finally:
            # cleanup temp files
            for p in tmp_paths:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

    return app


def run_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    allowed_origins: Optional[List[str]] = None,
    default_device: str = "cpu",
):
    model, meta = load_model(model_path, default_device)
    app = server(
        model, meta, allowed_origins=allowed_origins, default_device=default_device
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="Serve a pretrained model", prog="serve", usage="%(prog)s [options]"
    )
    parser.add_argument(
        "-m", "--model-path", help="model directory or name to load", required=True
    )
    parser.add_argument(
        "--device", default="cpu", help="default device for inference ('cpu' or 'cuda')"
    )
    parser.add_argument(
        "-p", "--port", help="port for server (default: 8000)", default=8000, type=int
    )
    parser.add_argument(
        "-H", "--host", help="host for server (default: 0.0.0.0)", default="0.0.0.0"
    )
    parser.add_argument(
        "-ao",
        "--allowed-origins",
        nargs="*",
        help="CORS allowed origins (default '*')",
        default=["*"],
    )
    parser.add_argument(
        "-l",
        "--logging-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "notset"],
    )

    args = parser.parse_args(sys_argv)

    level = getattr(logging, args.logging_level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)

    run_server(
        args.model_path,
        host=args.host,
        port=args.port,
        allowed_origins=args.allowed_origins,
        default_device=args.device,
    )


if __name__ == "__main__":
    cli(sys.argv[1:])
