# ============================================================
# CONTROLLED, CRASH-SAFE EEG ABLATION STUDY
#
# Key design features:
#   * Subject-separated evaluation (Stratified Group 5-Fold + LOSO)
#   * Fixed fold-wise seed schedule shared across ALL configurations/models
#   * TensorFlow deterministic operations enabled where supported
#   * Every completed fold is saved to disk BEFORE it is marked DONE
#   * True automatic resume: completed folds are loaded, incomplete folds rerun
#   * Per-fold predictions, history, metrics, weights and plots are preserved
#   * Per-subject out-of-fold summaries are preserved for later bootstrap CI
#   * No hyperparameter is selected from test performance inside this script
#
# IMPORTANT:
#   Bootstrap confidence intervals are intentionally NOT calculated here.
#   They should be calculated later from the saved subject-level predictions.
# ============================================================

import os

# Set deterministic-related environment flags BEFORE TensorFlow is imported.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import csv
import json
import math
import time
import glob
import random
import traceback
import platform
import datetime
from pathlib import Path
from importlib import metadata as importlib_metadata

import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping

mne.set_log_level("WARNING")

try:
    tf.config.experimental.enable_op_determinism()
    DETERMINISM_STATUS = "TensorFlow deterministic operations enabled"
except Exception as _det_err:
    DETERMINISM_STATUS = f"Determinism request could not be enabled: {_det_err}"

from tensorflow.keras.layers import DepthwiseConv2D, SeparableConv2D, AveragePooling2D, Activation
from tensorflow.keras.constraints import max_norm

# ============================================================
# ================ EDITABLE CONFIGURATION ====================
# ============================================================
MODEL_NAME = "EEGNet"
RUN_VERSION = "CONTROLLED_ABLATION_V1_2026-08-21"

DATA_FOLDER = r"D:\EEG DATA\Dataset\dataverse_files"
OUTPUT_FOLDER = r"D:\Manisha Phd\3rd Major revision-Paper#2\Output_EEDNet"

BASE_SEED = 42
PROTOCOLS_TO_RUN = ["stratkfold", "loso"]

SFREQ = 250
EPOCH_DUR = 1.0
LOWPASS = 0.5
HIGHPASS = 40.0

BATCH_SIZE = 32
MAX_EPOCHS = 150
PATIENCE = 20
PRINT_EVERY_N_EPOCHS = 5

LR_GRID = [0.005, 0.0005, 0.00005]

AUTO_RESUME = True
FORCE_RERUN = False
SAVE_MODEL_WEIGHTS = True
PREDICTION_THRESHOLD = 0.5
CSV_WRITE_RETRIES = 30
CSV_RETRY_SECONDS = 1.0
# ============================================================
# ============== END EDITABLE CONFIGURATION ==================
# ============================================================

# ============================================================
# OUTPUT PATHS
# ============================================================
OUTPUT_PATH = Path(OUTPUT_FOLDER)
CHECKPOINT_ROOT = OUTPUT_PATH / "fold_checkpoints"
CONFIG_ROOT = OUTPUT_PATH / "configuration_summaries"
GLOBAL_PLOTS = OUTPUT_PATH / "investigation_summary"

for _p in (OUTPUT_PATH, CHECKPOINT_ROOT, CONFIG_ROOT, GLOBAL_PLOTS):
    _p.mkdir(parents=True, exist_ok=True)

RUN_LOG = OUTPUT_PATH / "run_log.txt"
RUN_STATUS = OUTPUT_PATH / "run_status.json"
MANIFEST_JSON = OUTPUT_PATH / "experiment_manifest.json"
FOLD_DEFINITIONS_CSV = OUTPUT_PATH / "fold_definitions.csv"
SIGNAL_QUALITY_CSV = OUTPUT_PATH / "subject_signal_quality.csv"
ALL_FOLD_RESULTS_CSV = OUTPUT_PATH / "ALL_fold_results.csv"
ALL_CONFIG_RESULTS_CSV = OUTPUT_PATH / "ALL_configuration_results.csv"
ALL_SUBJECT_RESULTS_CSV = OUTPUT_PATH / "ALL_subject_predictions.csv"


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def log(msg=""):
    text = str(msg)
    print(text, flush=True)
    for attempt in range(CSV_WRITE_RETRIES):
        try:
            with open(RUN_LOG, "a", encoding="utf-8") as f:
                f.write(f"[{now_iso()}] {text}\n")
            return
        except PermissionError:
            time.sleep(CSV_RETRY_SECONDS)


def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=True)
    os.replace(tmp, path)


def atomic_write_npz(path, **arrays):
    path = Path(path)
    tmp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def safe_write_rows_csv(path, fieldnames, rows):
    """Rewrite a CSV atomically. Used for authoritative summary tables."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(CSV_WRITE_RETRIES):
        try:
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                for row in rows:
                    w.writerow(row)
            os.replace(tmp, path)
            return True
        except PermissionError:
            if attempt < CSV_WRITE_RETRIES - 1:
                time.sleep(CSV_RETRY_SECONDS)
    log(f"WARNING: Could not write {path} after {CSV_WRITE_RETRIES} attempts.")
    return False


def safe_append_csv(path, fieldnames, row, unique_keys=None):
    """Append with retries. This file is secondary; checkpoints remain authoritative."""
    path = Path(path)
    for attempt in range(CSV_WRITE_RETRIES):
        try:
            exists = path.exists()
            effective_fields = list(fieldnames)
            if exists:
                try:
                    with open(path, "r", newline="", encoding="utf-8") as _hf:
                        _reader = csv.reader(_hf)
                        _header = next(_reader, None)
                        if _header:
                            effective_fields = _header
                except Exception:
                    pass
            if unique_keys and exists:
                try:
                    with open(path, "r", newline="", encoding="utf-8") as f:
                        for old in csv.DictReader(f):
                            if all(str(old.get(k, "")) == str(row.get(k, "")) for k in unique_keys):
                                return True
                except (OSError, csv.Error):
                    pass
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=effective_fields, extrasaction="ignore")
                if not exists or path.stat().st_size == 0:
                    w.writeheader()
                w.writerow(row)
            return True
        except PermissionError:
            if attempt < CSV_WRITE_RETRIES - 1:
                log(f"CSV temporarily locked: {path.name}; retrying...")
                time.sleep(CSV_RETRY_SECONDS)
    log(f"WARNING: Could not append to {path}. The fold checkpoint is still preserved.")
    return False


def package_version(name):
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "unknown"


def write_run_status(status, **extra):
    payload = {
        "run_version": RUN_VERSION,
        "model": MODEL_NAME,
        "status": status,
        "timestamp": now_iso(),
    }
    payload.update(extra)
    atomic_write_json(RUN_STATUS, payload)


def fold_seed(fold_idx_zero_based):
    """Same fold number always receives the same seed in every config and model."""
    return int(BASE_SEED + fold_idx_zero_based)


def reset_random_state(seed):
    # Clear old graph/state before every fold, then reset all RNGs.
    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        tf.random.set_seed(seed)
    tf.random.set_seed(seed)


def subject_code(subject_id):
    subject_id = int(subject_id)
    if subject_id < 14:
        return f"H{subject_id + 1:02d}"
    return f"S{subject_id - 13:02d}"


def finite_or_nan(x):
    try:
        x = float(x)
    except Exception:
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def compute_metrics(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    f1 = f1_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred) if len(y_true) > 1 else float("nan")

    auc = float("nan")
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, np.asarray(y_prob, dtype=float))
        except Exception:
            auc = float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1),
        "kappa": float(kappa),
        "auc": float(auc),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def nan_summary(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "iqr": float("nan")}
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
    }


def aggregate_subject_predictions(y_true, y_pred, y_prob, subject_ids):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    subject_ids = np.asarray(subject_ids).astype(int)

    rows = []
    for sid in sorted(np.unique(subject_ids)):
        mask = subject_ids == sid
        true_vals = np.unique(y_true[mask])
        if len(true_vals) != 1:
            raise RuntimeError(f"Subject {sid} has more than one true class in test predictions.")
        true_label = int(true_vals[0])
        probs = y_prob[mask]
        preds = y_pred[mask]
        mean_prob = float(np.mean(probs))
        median_prob = float(np.median(probs))
        subject_pred = int(mean_prob >= PREDICTION_THRESHOLD)
        epoch_acc = float(np.mean(preds == true_label))
        rows.append({
            "subject_id": int(sid),
            "subject_code": subject_code(sid),
            "true_label": true_label,
            "n_test_epochs": int(mask.sum()),
            "epoch_accuracy": epoch_acc,
            "mean_probability_schizophrenia": mean_prob,
            "median_probability_schizophrenia": median_prob,
            "subject_predicted_label": subject_pred,
            "subject_correct": int(subject_pred == true_label),
            "mean_prediction_margin_from_0.5": float(abs(mean_prob - 0.5)),
        })
    return rows


def subject_level_metrics(subject_rows):
    yt = np.array([r["true_label"] for r in subject_rows], dtype=int)
    yp = np.array([r["subject_predicted_label"] for r in subject_rows], dtype=int)
    pr = np.array([r["mean_probability_schizophrenia"] for r in subject_rows], dtype=float)
    return compute_metrics(yt, yp, pr)


def save_confusion_plot(cm, path, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Healthy(0)", "Schiz(1)"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Healthy(0)", "Schiz(1)"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def save_roc_plot(y_true, y_prob, path, title):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        Path(str(path).replace(".png", "_NOT_APPLICABLE.txt")).write_text(
            "ROC-AUC is not defined because this test fold contains only one true class.\n"
            "This is expected for an individual LOSO fold. The pooled LOSO ROC is generated after all subjects are complete.\n",
            encoding="utf-8"
        )
        return float("nan")
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return float(auc)


def save_training_plots(history_dict, folder, title_prefix):
    epochs = np.arange(1, len(history_dict.get("loss", [])) + 1)
    if len(epochs) == 0:
        return

    fig = plt.figure(figsize=(6, 4))
    plt.plot(epochs, history_dict.get("loss", []), label="Train loss")
    if "val_loss" in history_dict:
        plt.plot(epochs, history_dict["val_loss"], label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{title_prefix} - loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(Path(folder) / "training_loss.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 4))
    plt.plot(epochs, history_dict.get("accuracy", []), label="Train accuracy")
    if "val_accuracy" in history_dict:
        plt.plot(epochs, history_dict["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{title_prefix} - accuracy")
    plt.legend(); plt.tight_layout()
    plt.savefig(Path(folder) / "training_accuracy.png", dpi=160)
    plt.close(fig)


def save_probability_plot(y_true, y_prob, path, title):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    fig = plt.figure(figsize=(6, 4))
    for cls, label in [(0, "Healthy true class"), (1, "Schizophrenia true class")]:
        vals = y_prob[y_true == cls]
        if len(vals):
            plt.hist(vals, bins=20, alpha=0.55, label=label)
    plt.axvline(PREDICTION_THRESHOLD, linestyle="--", linewidth=1)
    plt.xlabel("Predicted probability of schizophrenia")
    plt.ylabel("Epoch count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def write_history_csv(history_dict, path):
    keys = list(history_dict.keys())
    n = max([len(history_dict[k]) for k in keys], default=0)
    rows = []
    for i in range(n):
        row = {"epoch": i + 1}
        for k in keys:
            vals = history_dict[k]
            row[k] = float(vals[i]) if i < len(vals) else ""
        rows.append(row)
    safe_write_rows_csv(path, ["epoch"] + keys, rows)


def best_epoch_from_history(history_dict):
    vals = history_dict.get("val_loss", [])
    if not vals:
        return None
    return int(np.argmin(np.asarray(vals, dtype=float)) + 1)


def fold_folder(config_tag, fold_number):
    return CHECKPOINT_ROOT / config_tag / f"fold_{fold_number:03d}"


def done_marker_valid(config_tag, fold_number, expected_seed, protocol, extra_expected=None):
    folder = fold_folder(config_tag, fold_number)
    done_path = folder / "DONE.json"
    summary_path = folder / "fold_summary.json"
    pred_path = folder / "predictions.npz"
    if not (done_path.exists() and summary_path.exists() and pred_path.exists()):
        return False
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("run_version") != RUN_VERSION:
            return False
        if done.get("model") != MODEL_NAME:
            return False
        if done.get("config_tag") != config_tag:
            return False
        if int(done.get("fold")) != int(fold_number):
            return False
        if int(done.get("seed")) != int(expected_seed):
            return False
        if done.get("protocol") != protocol:
            return False
        if extra_expected:
            for k, v in extra_expected.items():
                if str(done.get(k)) != str(v):
                    return False
        return True
    except Exception:
        return False


def load_saved_fold(config_tag, fold_number):
    folder = fold_folder(config_tag, fold_number)
    summary = json.loads((folder / "fold_summary.json").read_text(encoding="utf-8"))
    with np.load(folder / "predictions.npz", allow_pickle=False) as z:
        return {
            "summary": summary,
            "y_true": z["y_true"].copy(),
            "y_pred": z["y_pred"].copy(),
            "y_prob": z["y_prob"].copy(),
            "subject_ids": z["subject_ids"].copy(),
        }


def make_signal_quality_row(X_epochs, subject_id, label):
    # Descriptive signal-quality indicators only. They do NOT prove that a recording is an artifact.
    x_uv = np.asarray(X_epochs, dtype=np.float32) * 1e6
    ptp = np.ptp(x_uv, axis=2)
    rms = np.sqrt(np.mean(np.square(x_uv), axis=2))
    ch_std = np.std(x_uv, axis=(0, 2))
    abs_x = np.abs(x_uv)
    return {
        "subject_id": int(subject_id),
        "subject_code": subject_code(subject_id),
        "true_label": int(label),
        "n_epochs": int(x_uv.shape[0]),
        "n_channels": int(x_uv.shape[1]),
        "median_abs_amplitude_uV": float(np.median(abs_x)),
        "p95_abs_amplitude_uV": float(np.percentile(abs_x, 95)),
        "median_epoch_channel_ptp_uV": float(np.median(ptp)),
        "p95_epoch_channel_ptp_uV": float(np.percentile(ptp, 95)),
        "median_epoch_channel_rms_uV": float(np.median(rms)),
        "min_channel_std_uV": float(np.min(ch_std)),
        "max_channel_std_uV": float(np.max(ch_std)),
    }


class LiveEpochTimer(tf.keras.callbacks.Callback):
    def __init__(self, tag, fold_label, print_every=PRINT_EVERY_N_EPOCHS):
        super().__init__()
        self.tag = tag
        self.fold_label = fold_label
        self.print_every = print_every
        self.t0 = None

    def on_train_begin(self, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_every == 0:
            logs = logs or {}
            elapsed_min = (time.time() - self.t0) / 60
            vl = logs.get("val_loss", float("nan"))
            va = logs.get("val_accuracy", float("nan"))
            log(f"      [{self.tag} | {self.fold_label}] epoch {epoch+1:>3}  "
                f"elapsed {elapsed_min:5.1f} min  val_loss={vl:.3f}  val_acc={va:.3f}")


def save_fold_definitions(stratk_folds, loso_folds, subj_all, subject_label):
    rows = []
    for protocol, folds in [("stratkfold", stratk_folds), ("loso", loso_folds)]:
        for i, fd in enumerate(folds, 1):
            for role, mask_key in [("train", "train_mask"), ("validation", "val_mask"), ("test", "test_mask")]:
                sids = sorted(np.unique(subj_all[fd[mask_key]]).astype(int))
                for sid in sids:
                    rows.append({
                        "protocol": protocol,
                        "fold": i,
                        "fold_label": fd["label"],
                        "role": role,
                        "subject_id": sid,
                        "subject_code": subject_code(sid),
                        "class_label": int(subject_label[sid]),
                        "seed_for_this_fold": fold_seed(i - 1),
                    })
    safe_write_rows_csv(
        FOLD_DEFINITIONS_CSV,
        ["protocol", "fold", "fold_label", "role", "subject_id", "subject_code", "class_label", "seed_for_this_fold"],
        rows,
    )


def write_manifest(extra_manifest):
    payload = {
        "run_version": RUN_VERSION,
        "model": MODEL_NAME,
        "created": now_iso(),
        "data_folder": DATA_FOLDER,
        "output_folder": OUTPUT_FOLDER,
        "base_seed": BASE_SEED,
        "seed_policy": "fold seed = BASE_SEED + zero_based_fold_index; reset before every fold; same schedule across configs/models",
        "determinism_status": DETERMINISM_STATUS,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "mne": mne.__version__,
        "scikit_learn": package_version("scikit-learn"),
        "scipy": package_version("scipy"),
        "matplotlib": matplotlib.__version__,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "auto_resume": AUTO_RESUME,
        "save_model_weights": SAVE_MODEL_WEIGHTS,
    }
    payload.update(extra_manifest)
    atomic_write_json(MANIFEST_JSON, payload)


def rebuild_global_tables():
    fold_rows, config_rows, subject_rows = [], [], []
    for p in sorted(CHECKPOINT_ROOT.glob("*/fold_*/fold_summary.json")):
        try:
            fold_rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    for p in sorted(CONFIG_ROOT.glob("*/config_summary.json")):
        try:
            config_rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    for p in sorted(CONFIG_ROOT.glob("*/subject_predictions.csv")):
        try:
            with open(p, "r", newline="", encoding="utf-8") as f:
                subject_rows.extend(list(csv.DictReader(f)))
        except Exception:
            pass

    if fold_rows:
        fields = sorted({k for r in fold_rows for k in r.keys()})
        safe_write_rows_csv(ALL_FOLD_RESULTS_CSV, fields, fold_rows)
    if config_rows:
        fields = sorted({k for r in config_rows for k in r.keys()})
        safe_write_rows_csv(ALL_CONFIG_RESULTS_CSV, fields, config_rows)
    if subject_rows:
        fields = sorted({k for r in subject_rows for k in r.keys()})
        safe_write_rows_csv(ALL_SUBJECT_RESULTS_CSV, fields, subject_rows)


def save_fold_bar_plot(fold_summaries, path, metric, title):
    xs = [int(r["fold"]) for r in fold_summaries]
    ys = [finite_or_nan(r.get(metric)) for r in fold_summaries]
    fig = plt.figure(figsize=(8, 4))
    plt.bar([str(x) for x in xs], ys)
    plt.xlabel("Fold")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def save_subject_bar_plot(subject_rows, path, title):
    rows = sorted(subject_rows, key=lambda r: r["subject_id"])
    labels = [r["subject_code"] for r in rows]
    vals = [r["epoch_accuracy"] for r in rows]
    fig = plt.figure(figsize=(11, 4))
    plt.bar(labels, vals)
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.ylabel("Epoch accuracy for subject")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def save_subject_difficulty_heatmap(protocol, config_tags, title_prefix):
    per_config = {}
    all_sids = set()
    for tag in config_tags:
        p = CONFIG_ROOT / tag / "subject_predictions.csv"
        if not p.exists():
            continue
        rows = []
        with open(p, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        d = {int(r["subject_id"]): float(r["epoch_accuracy"]) for r in rows}
        per_config[tag] = d
        all_sids.update(d.keys())
    if not per_config:
        return
    sids = sorted(all_sids)
    tags = [t for t in config_tags if t in per_config]
    mat = np.full((len(sids), len(tags)), np.nan, dtype=float)
    for j, tag in enumerate(tags):
        for i, sid in enumerate(sids):
            if sid in per_config[tag]:
                mat[i, j] = per_config[tag][sid]

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(tags)), 9))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_yticks(range(len(sids)))
    ax.set_yticklabels([subject_code(s) for s in sids])
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=90, fontsize=7)
    ax.set_title(f"{title_prefix} - subject difficulty ({protocol})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Epoch accuracy for each held-out subject")
    plt.tight_layout()
    plt.savefig(GLOBAL_PLOTS / f"{MODEL_NAME}_{protocol}_subject_difficulty_heatmap.png", dpi=180)
    plt.close(fig)


def save_readme():
    text = f"{MODEL_NAME} CONTROLLED ABLATION OUTPUT\n\n"
    text += "Authoritative resume record:\n  fold_checkpoints/<config>/fold_XXX/DONE.json\n\n"
    text += "A fold is considered complete ONLY when DONE.json exists and matches the current run version/seed/config.\n"
    text += "If execution stops during a fold, that incomplete fold is retrained on restart; earlier completed folds are loaded from disk.\n\n"
    text += "Each completed fold contains:\n"
    text += "  fold_summary.json, predictions.npz, history.csv, model weights (if enabled), confusion matrix, training curves, probability plot, and ROC plot when mathematically valid.\n\n"
    text += "LOSO note:\n  An individual LOSO test fold contains only one subject and therefore one true class. Individual-fold ROC-AUC is undefined and is intentionally NOT plotted. Pooled LOSO ROC-AUC is generated after all 28 subjects are complete.\n\n"
    text += "Subject-level predictions:\n  configuration_summaries/<config>/subject_predictions.csv contains one row per held-out subject and is intended for later subject-level bootstrap confidence-interval analysis.\n\n"
    text += "Signal-quality file:\n  subject_signal_quality.csv contains descriptive amplitude statistics only. It does not prove that any subject has an artifact.\n"
    (OUTPUT_PATH / "README_OUTPUT.txt").write_text(text, encoding="utf-8")

# ============================================================
# LABELS + LOAD/PREPROCESS
# ============================================================
def get_label_and_subject_id(filepath):
    fname = os.path.basename(filepath).lower()
    name = os.path.splitext(fname)[0]
    if name.startswith("h"):
        label = 0
        try:
            number = int(name[1:])
        except ValueError:
            return None, None
        subject_id = number - 1
    elif name.startswith("s"):
        label = 1
        try:
            number = int(name[1:])
        except ValueError:
            return None, None
        subject_id = 14 + (number - 1)
    else:
        return None, None
    return label, subject_id


def load_and_preprocess(filepath):
    label, subject_id = get_label_and_subject_id(filepath)
    if label is None:
        log(f"SKIPPING unknown file: {filepath}")
        return None, None, None, None
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    raw.pick_types(eeg=True)
    raw.filter(LOWPASS, HIGHPASS, fir_design="firwin", verbose=False)
    raw.set_eeg_reference(ref_channels="average", verbose=False)
    from mne import make_fixed_length_epochs
    epochs = make_fixed_length_epochs(
        raw, duration=EPOCH_DUR, overlap=0.0, preload=True, verbose=False
    )
    X = epochs.get_data()
    y = np.full(len(epochs), label, dtype=np.int32)
    quality = make_signal_quality_row(X, subject_id, label)
    return X, y, subject_id, quality


log("=" * 70)
log(f"LOADING AND PREPROCESSING EDF FILES ({MODEL_NAME})")
log("=" * 70)
t_load_start = time.time()

edf_paths = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.edf")))
if not edf_paths:
    raise FileNotFoundError(f"No EDF files found in DATA_FOLDER: {DATA_FOLDER}")
log(f"Found {len(edf_paths)} EDF files.")

X_list, y_list, subject_id_list, quality_rows = [], [], [], []
for path in edf_paths:
    X, y, sid, quality = load_and_preprocess(path)
    if X is None:
        continue
    X_list.append(X)
    y_list.append(y)
    subject_id_list.append(np.full(len(y), sid, dtype=np.int32))
    quality_rows.append(quality)
    log(f"  {os.path.basename(path)}: {len(y)} epochs, label={int(y[0])}, subject={subject_code(sid)}")

X_all = np.concatenate(X_list, axis=0)
y_all = np.concatenate(y_list, axis=0)
subj_all = np.concatenate(subject_id_list, axis=0)
log(f"Loaded in {(time.time()-t_load_start)/60:.1f} min. Total epochs: {len(y_all)}")

quality_fields = list(quality_rows[0].keys())
safe_write_rows_csv(SIGNAL_QUALITY_CSV, quality_fields, quality_rows)

# Preserve the supplied script's per-epoch z-score normalization.
X_1 = X_all.astype(np.float32)
mean = X_1.mean(axis=2, keepdims=True)
std = X_1.std(axis=2, keepdims=True)
std = np.where(std < 1e-8, 1e-8, std)
X_1 = (X_1 - mean) / std

unique_subjects = np.unique(subj_all)
subject_label = {int(s): int(y_all[subj_all == s][0]) for s in unique_subjects}
healthy_subjects = np.array([s for s in unique_subjects if subject_label[int(s)] == 0])
schiz_subjects = np.array([s for s in unique_subjects if subject_label[int(s)] == 1])
log(f"Healthy subjects ({len(healthy_subjects)}): {[subject_code(s) for s in healthy_subjects]}")
log(f"Schizophrenia subjects ({len(schiz_subjects)}): {[subject_code(s) for s in schiz_subjects]}")


def build_stratkfold_folds(k=5):
    # Same subject grouping logic as the supplied scripts.
    healthy_groups = np.array_split(healthy_subjects, k)
    schiz_groups = np.array_split(schiz_subjects, k)
    folds = []
    for fold in range(k):
        test_subjects = np.concatenate([healthy_groups[fold], schiz_groups[fold]])
        next_fold = (fold + 1) % k
        val_subjects = np.concatenate([healthy_groups[next_fold][:2], schiz_groups[next_fold][:2]])
        excluded = np.concatenate([test_subjects, val_subjects])
        train_subjects = np.array([s for s in unique_subjects if s not in excluded])
        folds.append({
            "label": f"fold{fold+1}of{k}",
            "test_mask": np.isin(subj_all, test_subjects),
            "val_mask": np.isin(subj_all, val_subjects),
            "train_mask": np.isin(subj_all, train_subjects),
        })
    return folds


def build_loso_folds():
    folds = []
    for i, test_subject in enumerate(unique_subjects):
        avail_healthy = [s for s in healthy_subjects if s != test_subject]
        avail_schiz = [s for s in schiz_subjects if s != test_subject]
        val_healthy = avail_healthy[i % len(avail_healthy)]
        val_schiz = avail_schiz[i % len(avail_schiz)]
        val_subjects = [val_healthy, val_schiz]
        test_mask = subj_all == test_subject
        val_mask = np.isin(subj_all, val_subjects)
        train_mask = ~test_mask & ~val_mask
        folds.append({
            "label": f"subj{int(test_subject)}_{subject_code(test_subject)}_fold{i+1}of{len(unique_subjects)}",
            "test_mask": test_mask,
            "val_mask": val_mask,
            "train_mask": train_mask,
        })
    return folds


STRATK_FOLDS = build_stratkfold_folds(k=5)
LOSO_FOLDS = build_loso_folds()
log(f"Stratified Group 5-Fold: {len(STRATK_FOLDS)} folds")
log(f"LOSO: {len(LOSO_FOLDS)} folds")
save_fold_definitions(STRATK_FOLDS, LOSO_FOLDS, subj_all, subject_label)

# ============================================================
# MODEL DEFINITION: EEGNet
# IMPORTANT: architecture is preserved from the supplied ablation_EEGNet script.
# ============================================================
def build_eegnet_model(input_shape, F1=8, D=2, kernel_length=125, dropout_rate=0.5):
    Chans, Samples, _ = input_shape
    F2 = F1 * D
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(F1, (1, kernel_length), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D,
                        depthwise_constraint=max_norm(1.))(x)
    x = layers.BatchNormalization()(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((1, 4))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = SeparableConv2D(F2, (1, 16), use_bias=False, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((1, 8))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(2, activation="softmax", kernel_constraint=max_norm(0.25))(x)
    return models.Model(inputs=inputs, outputs=outputs)


def build_model_for_config(input_shape, config_params):
    return build_eegnet_model(input_shape)

def run_one_config(protocol, config_tag, folds, config_params, X_data, input_shape):
    log("\n" + "=" * 78)
    log(f"CONFIG: {config_tag}")
    log(f"protocol={protocol} | params={config_params}")
    log("=" * 78)

    config_dir = CONFIG_ROOT / config_tag
    config_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []
    session_start = time.time()

    for fold_idx, fold_def in enumerate(folds):
        fold_number = fold_idx + 1
        seed = fold_seed(fold_idx)
        extra_expected = {k: v for k, v in config_params.items()}

        if AUTO_RESUME and not FORCE_RERUN and done_marker_valid(
            config_tag, fold_number, seed, protocol, extra_expected
        ):
            saved = load_saved_fold(config_tag, fold_number)
            fold_results.append(saved)
            log(f"  >>> AUTO-RESUME: {fold_def['label']} already complete; loaded from disk.")
            continue

        folder = fold_folder(config_tag, fold_number)
        folder.mkdir(parents=True, exist_ok=True)
        # Remove stale DONE marker before retraining an incomplete/forced fold.
        try:
            (folder / "DONE.json").unlink(missing_ok=True)
        except TypeError:
            if (folder / "DONE.json").exists():
                (folder / "DONE.json").unlink()

        fold_t0 = time.time()
        try:
            train_mask = fold_def["train_mask"]
            val_mask = fold_def["val_mask"]
            test_mask = fold_def["test_mask"]

            X_train, y_train = X_data[train_mask], y_all[train_mask]
            X_val, y_val = X_data[val_mask], y_all[val_mask]
            X_test, y_test = X_data[test_mask], y_all[test_mask]
            test_subject_ids = subj_all[test_mask]

            train_subjects = sorted(np.unique(subj_all[train_mask]).astype(int).tolist())
            val_subjects = sorted(np.unique(subj_all[val_mask]).astype(int).tolist())
            test_subjects = sorted(np.unique(test_subject_ids).astype(int).tolist())

            reset_random_state(seed)
            classes = np.unique(y_train)
            cw = compute_class_weight("balanced", classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes.astype(int), cw))

            model = build_model_for_config(input_shape, config_params)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=float(config_params["lr"])),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            es = EarlyStopping(
                monitor="val_loss", mode="min", patience=PATIENCE,
                restore_best_weights=True, verbose=0
            )
            timer_cb = LiveEpochTimer(config_tag, fold_def["label"])

            log(f"  --- {fold_def['label']} | seed={seed} | train={len(y_train)} "
                f"val={len(y_val)} test={len(y_test)} epochs | test subjects={test_subjects} ---")

            hist = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[es, timer_cb],
                class_weight=class_weight_dict,
                shuffle=True,
                verbose=0,
            )

            y_pred_prob = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_prob_1 = y_pred_prob[:, 1]
            metrics = compute_metrics(y_test, y_pred, y_prob_1)
            fold_duration_min = (time.time() - fold_t0) / 60

            history_dict = {k: [float(x) for x in v] for k, v in hist.history.items()}
            best_epoch = best_epoch_from_history(history_dict)
            epochs_run = len(history_dict.get("loss", []))
            best_val_loss = float(np.min(history_dict.get("val_loss", [np.nan])))
            best_val_acc = float(np.max(history_dict.get("val_accuracy", [np.nan])))
            final_train_acc = float(history_dict.get("accuracy", [np.nan])[-1])
            final_val_acc = float(history_dict.get("val_accuracy", [np.nan])[-1])

            summary = {
                "timestamp": now_iso(),
                "run_version": RUN_VERSION,
                "model": MODEL_NAME,
                "protocol": protocol,
                "config_tag": config_tag,
                **config_params,
                "fold": fold_number,
                "fold_label": fold_def["label"],
                "seed": seed,
                "train_subjects": ";".join(subject_code(s) for s in train_subjects),
                "validation_subjects": ";".join(subject_code(s) for s in val_subjects),
                "test_subjects": ";".join(subject_code(s) for s in test_subjects),
                "n_train_epochs": int(len(y_train)),
                "n_validation_epochs": int(len(y_val)),
                "n_test_epochs": int(len(y_test)),
                "epochs_run": int(epochs_run),
                "best_epoch_by_val_loss": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_acc,
                "final_train_accuracy": final_train_acc,
                "final_val_accuracy": final_val_acc,
                "train_minus_val_accuracy_final": float(final_train_acc - final_val_acc),
                "duration_minutes": float(fold_duration_min),
                **metrics,
            }

            # 1) Save the data required to reconstruct every metric FIRST.
            atomic_write_npz(
                folder / "predictions.npz",
                y_true=np.asarray(y_test, dtype=np.int32),
                y_pred=np.asarray(y_pred, dtype=np.int32),
                y_prob=np.asarray(y_prob_1, dtype=np.float32),
                subject_ids=np.asarray(test_subject_ids, dtype=np.int32),
            )
            atomic_write_json(folder / "fold_summary.json", summary)
            write_history_csv(history_dict, folder / "history.csv")

            # 2) Save one-row subject summaries for this fold.
            fold_subject_rows = aggregate_subject_predictions(y_test, y_pred, y_prob_1, test_subject_ids)
            for r in fold_subject_rows:
                r.update({
                    "model": MODEL_NAME, "protocol": protocol, "config_tag": config_tag,
                    "fold": fold_number, "seed": seed, **config_params,
                })
            if fold_subject_rows:
                safe_write_rows_csv(
                    folder / "subject_predictions.csv",
                    list(fold_subject_rows[0].keys()),
                    fold_subject_rows,
                )

            # 3) Save model weights after EarlyStopping restored the best weights.
            if SAVE_MODEL_WEIGHTS:
                model.save_weights(folder / "model_best.weights.h5")

            # 4) Save all per-fold diagnostic plots.
            cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
            save_confusion_plot(cm, folder / "confusion_matrix.png", f"{config_tag}\n{fold_def['label']}")
            save_training_plots(history_dict, folder, f"{config_tag} | {fold_def['label']}")
            save_probability_plot(y_test, y_prob_1, folder / "probability_distribution.png", f"{config_tag} | {fold_def['label']}")
            save_roc_plot(y_test, y_prob_1, folder / "roc_curve.png", f"{config_tag}\n{fold_def['label']}")

            # 5) DONE marker is deliberately LAST. Only now is this fold resumable/skippable.
            done_payload = {
                "timestamp": now_iso(),
                "run_version": RUN_VERSION,
                "model": MODEL_NAME,
                "protocol": protocol,
                "config_tag": config_tag,
                "fold": fold_number,
                "fold_label": fold_def["label"],
                "seed": seed,
                **config_params,
                "status": "COMPLETE",
            }
            atomic_write_json(folder / "DONE.json", done_payload)

            # Secondary convenient master CSV. A lock here cannot destroy the checkpoint.
            safe_append_csv(
                ALL_FOLD_RESULTS_CSV,
                list(summary.keys()), summary,
                unique_keys=["run_version", "model", "config_tag", "fold"],
            )

            fold_results.append({
                "summary": summary,
                "y_true": np.asarray(y_test, dtype=np.int32),
                "y_pred": np.asarray(y_pred, dtype=np.int32),
                "y_prob": np.asarray(y_prob_1, dtype=np.float32),
                "subject_ids": np.asarray(test_subject_ids, dtype=np.int32),
            })
            log(f"  --- {fold_def['label']} DONE in {fold_duration_min:.1f} min | "
                f"acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f} auc={metrics['auc']:.4f} ---")

        except BaseException as exc:
            err_text = (
                f"Timestamp: {now_iso()}\nConfig: {config_tag}\nFold: {fold_number}\n"
                f"Seed: {seed}\nException: {repr(exc)}\n\n{traceback.format_exc()}"
            )
            (folder / "ERROR_OR_INTERRUPTION.txt").write_text(err_text, encoding="utf-8")
            log(f"STOPPED inside {config_tag}, fold {fold_number}. This fold has NOT been marked DONE and will rerun on restart.")
            raise

    # Reconstruct this configuration ONLY from completed/saved fold predictions.
    if len(fold_results) != len(folds):
        raise RuntimeError(f"Config {config_tag} has {len(fold_results)}/{len(folds)} folds available.")

    all_y_true = np.concatenate([r["y_true"] for r in fold_results])
    all_y_pred = np.concatenate([r["y_pred"] for r in fold_results])
    all_y_prob = np.concatenate([r["y_prob"] for r in fold_results])
    all_subject_ids = np.concatenate([r["subject_ids"] for r in fold_results])

    epoch_metrics = compute_metrics(all_y_true, all_y_pred, all_y_prob)
    subject_rows = aggregate_subject_predictions(all_y_true, all_y_pred, all_y_prob, all_subject_ids)
    subj_metrics = subject_level_metrics(subject_rows)

    fold_summaries = [r["summary"] for r in fold_results]
    fold_metric_names = ["accuracy", "precision", "sensitivity", "specificity", "f1", "kappa", "auc"]
    fold_stats = {}
    for m in fold_metric_names:
        s = nan_summary([r.get(m, np.nan) for r in fold_summaries])
        for stat_name, stat_value in s.items():
            fold_stats[f"fold_{m}_{stat_name}"] = stat_value

    config_summary = {
        "timestamp": now_iso(),
        "run_version": RUN_VERSION,
        "model": MODEL_NAME,
        "protocol": protocol,
        "config_tag": config_tag,
        **config_params,
        "n_folds": len(folds),
        "n_test_epochs_pooled": int(len(all_y_true)),
        "n_unique_test_subjects": int(len(np.unique(all_subject_ids))),
        "total_fold_training_minutes": float(sum(float(r["duration_minutes"]) for r in fold_summaries)),
        "session_wall_minutes_for_this_config": float((time.time() - session_start) / 60),
        **{f"epoch_{k}": v for k, v in epoch_metrics.items()},
        **{f"subject_{k}": v for k, v in subj_metrics.items()},
        **fold_stats,
    }

    atomic_write_json(config_dir / "config_summary.json", config_summary)
    safe_write_rows_csv(config_dir / "config_summary.csv", list(config_summary.keys()), [config_summary])

    for r in subject_rows:
        r.update({
            "run_version": RUN_VERSION, "model": MODEL_NAME,
            "protocol": protocol, "config_tag": config_tag, **config_params,
        })
    safe_write_rows_csv(config_dir / "subject_predictions.csv", list(subject_rows[0].keys()), subject_rows)
    safe_write_rows_csv(config_dir / "fold_summary.csv", list(fold_summaries[0].keys()), fold_summaries)

    # Config-level plots, reconstructed from saved out-of-fold predictions.
    cm_epoch = np.array([[epoch_metrics["tn"], epoch_metrics["fp"]], [epoch_metrics["fn"], epoch_metrics["tp"]]])
    save_confusion_plot(cm_epoch, config_dir / "pooled_epoch_confusion_matrix.png", f"{config_tag}\nPooled epoch confusion matrix")
    save_roc_plot(all_y_true, all_y_prob, config_dir / "pooled_epoch_roc_curve.png", f"{config_tag}\nPooled epoch ROC")

    subj_y_true = np.array([r["true_label"] for r in subject_rows], dtype=int)
    subj_y_pred = np.array([r["subject_predicted_label"] for r in subject_rows], dtype=int)
    subj_y_prob = np.array([r["mean_probability_schizophrenia"] for r in subject_rows], dtype=float)
    cm_subj = confusion_matrix(subj_y_true, subj_y_pred, labels=[0, 1])
    save_confusion_plot(cm_subj, config_dir / "subject_level_confusion_matrix.png", f"{config_tag}\nSubject-level confusion matrix")
    save_roc_plot(subj_y_true, subj_y_prob, config_dir / "subject_level_roc_curve.png", f"{config_tag}\nSubject-level ROC")

    save_fold_bar_plot(fold_summaries, config_dir / "fold_accuracy.png", "accuracy", f"{config_tag} - fold accuracy")
    save_subject_bar_plot(subject_rows, config_dir / "subject_epoch_accuracy.png", f"{config_tag} - held-out subject difficulty")

    log(f"CONFIG COMPLETE: {config_tag}")
    log(f"  Epoch-level: acc={epoch_metrics['accuracy']:.4f}, kappa={epoch_metrics['kappa']:.4f}, auc={epoch_metrics['auc']:.4f}")
    log(f"  Subject-level: acc={subj_metrics['accuracy']:.4f}, kappa={subj_metrics['kappa']:.4f}, auc={subj_metrics['auc']:.4f}")
    return config_summary

def save_eegnet_metric_plot(protocol, summaries, metric_key, fname, title):
    rows = sorted([s for s in summaries if s["protocol"] == protocol], key=lambda s: LR_GRID.index(float(s["lr"])))
    if not rows:
        return
    xs = [str(s["lr"]) for s in rows]
    ys = [finite_or_nan(s.get(metric_key)) for s in rows]
    fig = plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Learning rate")
    plt.ylabel(metric_key.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(GLOBAL_PLOTS / fname, dpi=180)
    plt.close(fig)


def save_global_investigation_plots(all_summaries):
    for protocol in PROTOCOLS_TO_RUN:
        prot = [s for s in all_summaries if s["protocol"] == protocol]
        if not prot:
            continue
        for metric_key, short in [
            ("epoch_accuracy", "epoch_accuracy"),
            ("epoch_kappa", "epoch_kappa"),
            ("epoch_auc", "epoch_auc"),
            ("subject_accuracy", "subject_accuracy"),
            ("subject_kappa", "subject_kappa"),
            ("subject_auc", "subject_auc"),
        ]:
            save_eegnet_metric_plot(
                protocol, prot, metric_key,
                f"EEGNet_{protocol}_{short}_vs_LR.png",
                f"EEGNet {protocol}: {short.replace('_', ' ')} vs learning rate",
            )
        tags = [s["config_tag"] for s in prot]
        save_subject_difficulty_heatmap(protocol, tags, "EEGNet")

def main():
    write_run_status("RUNNING")
    save_readme()
    write_manifest({
        "learning_rate_grid": LR_GRID,
        "protocols": PROTOCOLS_TO_RUN,
        "planned_fold_trainings": int(len(LR_GRID) * sum(len(STRATK_FOLDS if p == "stratkfold" else LOSO_FOLDS) for p in PROTOCOLS_TO_RUN)),
        "input_representation": "raw normalized channel-time epochs; no STFT",
        "ablation_note": "Learning-rate sensitivity only. No best setting is selected from test performance inside this script.",
    })

    log("\n" + "#" * 78)
    log("EEGNet CONTROLLED ABLATION")
    log(f"Base seed={BASE_SEED}; same fold seed schedule across every LR config and 3D-CNN.")
    log(DETERMINISM_STATUS)
    log("#" * 78)

    X_eeg = X_1[..., np.newaxis]
    input_shape = (19, 250, 1)
    all_summaries = []
    for protocol in PROTOCOLS_TO_RUN:
        folds = STRATK_FOLDS if protocol == "stratkfold" else LOSO_FOLDS
        for lr in LR_GRID:
            tag = f"EEGNet_{protocol}_lr{lr}"
            config_params = {"lr": float(lr)}
            summary = run_one_config(protocol, tag, folds, config_params, X_eeg, input_shape)
            all_summaries.append(summary)
            rebuild_global_tables()

    save_global_investigation_plots(all_summaries)
    rebuild_global_tables()
    write_run_status("COMPLETE", completed_configurations=len(all_summaries))
    log("\nEEGNet CONTROLLED ABLATION COMPLETE.")
    log(f"Output folder: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_run_status("STOPPED_OR_FAILED", exception=repr(exc), traceback=traceback.format_exc())
        raise
