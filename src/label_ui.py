"""
Streamlit prelabel review & correction UI.

Inputs must already use canonical column names:
  audio_file, start_seconds, end_seconds, label, probability (optional)

Features:
 - low-confidence indicator (active-learning)
 - filtering by filename substring and by label
 - playback of audio segments, save/append corrections, download
 - export: zip with audio segments + label .txt files (no header, tab-separated)
"""

import streamlit as st
import pandas as pd
import soundfile as sf
import torchaudio
import io
import os
import time
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import tempfile
import shutil
import zipfile
import getpass

from helpers.constants import LABELS, REQUIRED_COLS

st.set_page_config(layout="wide", page_title="Prelabel Review (Simple)")
st.title("Prelabel Review & Correction")


def load_csv_from_uploaded(uploaded) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded)
        return df
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        return pd.DataFrame()


def load_json_from_uploaded(uploaded) -> pd.DataFrame:
    """
    Accepts:
      - a JSON object with key "predictions": [ {...}, ... ] where each dict uses canonical column names, or
      - a JSON list [ {...}, {...} ] where each dict uses canonical column names.
    """
    try:
        raw = uploaded.read()
        if isinstance(raw, bytes):
            text = raw.decode("utf-8")
        else:
            text = str(raw)
    except Exception as e:
        st.error(f"Failed to read uploaded JSON: {e}")
        return pd.DataFrame()

    try:
        doc = json.loads(text)
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        return pd.DataFrame()

    if (
        isinstance(doc, dict)
        and "predictions" in doc
        and isinstance(doc["predictions"], list)
    ):
        df = pd.DataFrame(doc["predictions"])
        return df
    if isinstance(doc, list):
        df = pd.DataFrame(doc)
        return df

    st.error(
        "JSON must be a list of prediction objects or an object with a 'predictions' list. Each item must already contain canonical fields."
    )
    return pd.DataFrame()


def load_from_dir(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        st.error(f"Folder not found: {path}")
        return pd.DataFrame()
    dfs = []
    for f in sorted(p.glob("*.csv")):
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            st.warning(f"Skipping {f.name}: {e}")
    for f in sorted(p.glob("*.json")):
        try:
            with f.open("r", encoding="utf-8") as fh:
                doc = json.load(fh)
            if (
                isinstance(doc, dict)
                and "predictions" in doc
                and isinstance(doc["predictions"], list)
            ):
                dfs.append(pd.DataFrame(doc["predictions"]))
            elif isinstance(doc, list):
                dfs.append(pd.DataFrame(doc))
            else:
                st.warning(
                    f"Skipping {f.name}: JSON doesn't contain 'predictions' list or top-level list"
                )
        except Exception as e:
            st.warning(f"Skipping {f.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, sort=False)


def validate_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal validation:
      - required columns must be present (exact names)
      - start_seconds and end_seconds coerced to numeric; rows with invalid times are dropped
      - probability coerced to numeric if present
    """
    if df is None or df.empty:
        return pd.DataFrame()
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        st.error(f"Input missing required columns (must be exact names): {missing}")
        return pd.DataFrame()
    df = df.copy()
    df["start_seconds"] = pd.to_numeric(df["start_seconds"], errors="coerce")
    df["end_seconds"] = pd.to_numeric(df["end_seconds"], errors="coerce")
    df = df.dropna(subset=["start_seconds", "end_seconds"])
    df = df[df["end_seconds"] > df["start_seconds"]]
    if "probability" in df.columns:
        df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    else:
        df["probability"] = np.nan
    df["audio_file"] = df["audio_file"].astype(str)
    df["label"] = df["label"].astype(str)
    df = df.reset_index(drop=True)
    return df


def save_corrections_to_disk(
    corrections: pd.DataFrame, annotator: str, out_dir: str = "."
):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"annotated_{annotator}_{ts}.csv"
    out_path = Path(out_dir) / fname
    corrections.to_csv(out_path, index=False)
    return str(out_path)


def export_corrections_package(
    corrections_df: pd.DataFrame, audio_root: str, annotator: str
) -> str | None:
    """
    Create a zip archive with:
      - audio_segments/<segment_wav_files>
      - labels/<original_audio_basename>.txt  (tab-separated lines: start_seconds \t end_seconds \t label)
      - metadata.json (annotator + timestamp + num_items)
    Returns path to zip file, or None on error.
    """
    if corrections_df is None or corrections_df.empty:
        return None

    audio_root = str(Path(audio_root))
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    tmpdir = Path(tempfile.mkdtemp(prefix=f"export_{annotator}_{ts}_"))

    # Keep track of labels per original audio (basename)
    labels_map = {}
    missing_audio = []
    written_count = 0

    for i, row in corrections_df.reset_index(drop=True).iterrows():
        orig_rel = str(row.get("audio_file", ""))
        start = float(row.get("start_seconds", 0.0))
        end = float(row.get("end_seconds", 0.0))
        label = str(row.get("corrected_label", row.get("label", ""))).strip()
        orig_path = os.path.join(audio_root, orig_rel)

        audio_basename = Path(orig_rel).stem
        # ensure list exists
        labels_map.setdefault(audio_basename, []).append((start, end, label))

        if not os.path.exists(orig_path):
            missing_audio.append(orig_rel)
            continue

        try:
            waveform, sr = torchaudio.load(orig_path)  # [channels, T]
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            s_frame = int(max(0, round(start * sr)))
            e_frame = int(min(round(end * sr), waveform.shape[1]))
            seg = (
                waveform[:, s_frame:e_frame]
                if e_frame > s_frame
                else waveform[:, s_frame : s_frame + 1]
            )
            arr = seg.numpy()
            arr = np.transpose(arr)  # (T, channels)

            seg_fname = f"{audio_basename}_{i:05d}.wav"
            seg_path = tmpdir / seg_fname
            sf.write(str(seg_path), arr, sr, format="WAV")
            written_count += 1
        except Exception:
            missing_audio.append(orig_rel)
            continue

    # write label txt files (no header, tab-separated)
    for audio_basename, segs in labels_map.items():
        txt_path = tmpdir / f"{audio_basename}.txt"
        with open(txt_path, "w", encoding="utf-8") as fh:
            for s, e, lab in segs:
                fh.write(f"{s}\t{e}\t{lab}\n")

    # write metadata
    meta = {
        "annotator": annotator,
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_items": int(len(corrections_df)),
        "n_segments_written": int(written_count),
        "missing_audio_files": missing_audio[:50],
    }
    with open(tmpdir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    # zip it
    out_zip = Path(tempfile.gettempdir()) / f"export_{annotator}_{ts}.zip"
    try:
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(tmpdir):
                for fname in files:
                    full = Path(root) / fname
                    arcname = str(full.relative_to(tmpdir))
                    zf.write(full, arcname)
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None

    shutil.rmtree(tmpdir, ignore_errors=True)
    return str(out_zip)


# ------------------------
# UI Inputs
# ------------------------
left, right = st.columns([2, 1])

with left:
    uploaded = st.file_uploader("Upload prelabels (CSV or JSON)", type=["csv", "json"])
    audio_dir = st.text_input("Path to audio folder", value=str(Path.cwd() / "data"))
    load_folder = st.checkbox(
        "Load CSV/JSON files from folder instead of upload", value=False
    )
    if load_folder:
        folder = st.text_input(
            "Prelabels folder (contains .csv/.json)",
            value=str(Path.cwd() / "prelabels"),
        )
    default_annot = getpass.getuser() or "annotator_1"
    annotator = st.text_input("Annotator ID", value=f"annotator_{default_annot}")
    auto_advance = st.checkbox("Auto-advance to next row after Save", value=True)

with right:
    st.markdown("### Filters & Active-Learning")
    low_conf_threshold = st.slider(
        "Low-confidence threshold (probability <=)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.01,
    )
    show_only_low = st.checkbox("Show only low-confidence rows", value=False)
    filename_filter = st.text_input("Filename substring filter (leave empty for all)")
    label_options = st.multiselect(
        "Filter by label (multi-select)", options=LABELS, default=LABELS
    )
    sort_by_prob = st.checkbox("Sort by probability ascending (low first)", value=True)

# ------------------------
# Session state init
# ------------------------
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
    st.session_state.corrections = []
    st.session_state.index = 0

# Load / parse input into session_state.df
if uploaded is not None and not load_folder:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".csv":
        df = load_csv_from_uploaded(uploaded)
    else:
        df = load_json_from_uploaded(uploaded)
    df = validate_and_normalize(df)
    st.session_state.df = df
    st.session_state.index = 0

if load_folder:
    df = load_from_dir(folder)
    df = validate_and_normalize(df)
    st.session_state.df = df
    st.session_state.index = 0

# short-circuit if nothing loaded
if st.session_state.df.empty:
    st.info(
        "Upload a CSV/JSON or load from folder to begin annotation (must contain exact canonical columns)."
    )
    st.stop()

# compute low-confidence flag
df_all = st.session_state.df.copy()
df_all["probability"] = pd.to_numeric(
    df_all.get("probability", np.nan), errors="coerce"
)
df_all["low_confidence"] = df_all["probability"].fillna(0.0) <= float(
    low_conf_threshold
)

# apply filters
df_filtered = df_all.copy()
if filename_filter:
    df_filtered = df_filtered[
        df_filtered["audio_file"].str.contains(filename_filter, na=False)
    ]
if label_options:
    df_filtered = df_filtered[df_filtered["label"].isin(label_options)]
if show_only_low:
    df_filtered = df_filtered[df_filtered["low_confidence"]]
if sort_by_prob:
    df_filtered = df_filtered.sort_values(
        by=["probability"], ascending=True, na_position="last"
    )
else:
    df_filtered = df_filtered.sort_index()
df_filtered = df_filtered.reset_index(drop=True)

# navigation controls
n = len(df_filtered)
st.sidebar.markdown(f"**Rows (filtered):** {n}  |  **Total:** {len(df_all)}")

if n == 0:
    st.warning("No rows match your filters.")
    st.stop()

nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 6])
with nav_col1:
    if st.button("Previous"):
        st.session_state.index = max(0, st.session_state.index - 1)
with nav_col2:
    if st.button("Next"):
        st.session_state.index = min(n - 1, st.session_state.index + 1)
with nav_col3:
    idx = st.slider(
        "Index",
        0,
        max(0, n - 1),
        value=min(st.session_state.index, n - 1),
        key="index_slider",
    )
    st.session_state.index = int(idx)

i = int(st.session_state.index)
row = df_filtered.loc[i]

st.markdown(f"### Item {i+1} / {n}")
# Visual low-confidence indicator
if bool(row.get("low_confidence", False)):
    st.warning(f"LOW CONFIDENCE — probability={row.get('probability')}")
else:
    st.success(f"Predicted label: {row.get('label')} (prob={row.get('probability')})")

st.write(
    {
        "audio_file": row.get("audio_file"),
        "start_seconds": float(row.get("start_seconds", 0.0)),
        "end_seconds": float(row.get("end_seconds", 0.0)),
        "label": row.get("label"),
        "probability": row.get("probability"),
    }
)

# playback (use soundfile to write BytesIO)
audio_path = os.path.join(audio_dir, row["audio_file"])
if os.path.exists(audio_path):
    try:
        waveform, sr = torchaudio.load(audio_path)  # waveform: [channels, T]
        s = int(max(0, round(row["start_seconds"] * sr)))
        e = int(min(round(row["end_seconds"] * sr), waveform.shape[1]))
        seg = (
            waveform[:, s:e] if e > s else waveform[:, s : s + 1]
        )  # shape [channels, T]

        # ensure there is at least 1 frame
        if seg.shape[1] == 0:
            st.warning("Segment length is zero (start==end or out of bounds).")
        else:
            buf = io.BytesIO()
            arr = seg.numpy()
            arr = np.transpose(arr)  # (T, channels)
            sf.write(buf, arr, sr, format="WAV")
            buf.seek(0)
            st.audio(buf.read(), format="audio/wav")

            # optional waveform preview
            fig, ax = plt.subplots(figsize=(8, 2))
            arr_mono = arr[:, 0] if arr.ndim > 1 else arr
            ax.plot(arr_mono)
            ax.set_title(
                f"{row['audio_file']} [{row['start_seconds']:.3f} - {row['end_seconds']:.3f}]"
            )
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to load/play audio: {e}")
else:
    st.warning(f"Audio file not found: {audio_path}")

# Correction controls
col_a, col_b, col_c = st.columns([3, 2, 3])
with col_a:
    try:
        cur_idx = LABELS.index(row["label"]) if row["label"] in LABELS else 0
    except Exception:
        cur_idx = 0
    new_label = st.selectbox("Correct label", options=LABELS, index=cur_idx)
with col_b:
    action = st.selectbox(
        "Action", options=["corrected", "unchanged", "skip", "uncertain"]
    )
with col_c:
    comment = st.text_input("Comment (optional)", value="")

save_btn = st.button("Save correction for this row")
if save_btn:
    out = row.to_dict()
    out.update(
        {
            "corrected_label": new_label,
            "annotator_id": annotator,
            "action": action,
            "comment": comment,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    # append to session corrections
    st.session_state.corrections.append(out)
    # append to disk file as well (persist)
    out_df = pd.DataFrame([out])
    out_path = f"annotated_{annotator}.csv"
    if os.path.exists(out_path):
        out_df.to_csv(out_path, mode="a", header=False, index=False)
    else:
        out_df.to_csv(out_path, index=False)
    st.success(f"Saved to {out_path} (and session)")

    # auto-advance: update index and request a rerun in a robust way
    if auto_advance:
        st.session_state.index = min(n - 1, st.session_state.index + 1)
        # st.stop will halt the run; UI will rerun on next interaction - that is sufficient for most setups
        st.stop()

# Session corrections table + download/save/export
if st.session_state.corrections:
    corrections_df = pd.DataFrame(st.session_state.corrections)
    st.markdown("#### Corrections (this session)")
    st.dataframe(corrections_df.head(50))
    csv = corrections_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download corrections CSV",
        data=csv,
        file_name=f"corrections_{annotator}_{int(time.time())}.csv",
        mime="text/csv",
    )

    if st.button("Save corrections to disk (append)"):
        out_path = save_corrections_to_disk(corrections_df, annotator, out_dir=".")
        st.success(f"Wrote {out_path}")

    st.markdown("---")
    st.markdown("### Export corrected segments + label files (ZIP)")

    total_rows = len(corrections_df)
    if total_rows > 500:
        st.warning(
            f"You are about to export {total_rows} items — this may create a large zip file. Proceed if you know the size is OK."
        )

    if st.button("Create export package (zip)"):
        with st.spinner("Creating package..."):
            zip_path = export_corrections_package(corrections_df, audio_dir, annotator)
        if zip_path:
            st.success("Package created")
            with open(zip_path, "rb") as fh:
                data = fh.read()
            st.download_button(
                "Download export zip",
                data=data,
                file_name=os.path.basename(zip_path),
                mime="application/zip",
            )
            try:
                os.remove(zip_path)
            except Exception:
                pass
        else:
            st.error("Failed to create export package")

st.markdown("---")
st.write(
    "Note: Inputs must already use canonical columns: audio_file,start_seconds,end_seconds,label,probability (probability optional). This UI intentionally does NOT support NDJSON or automatic key remapping."
)
