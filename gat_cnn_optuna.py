#!/usr/bin/env python3
"""
Author: Milan Andras Fodor (@milaniusz)
Project: https://github.com/MILANIUSZ/cVEP-GAT-CNN
Used in: Graph-Attentive CNN for cVEP-BCI with Insights into Electrode Significance (IWANN 2025)
Optuna hyperparameter tuning for the cVEP GAT‑CNN model.

This script will:
  1. Load and preprocess each subject’s CSV data.
  2. Define an Optuna objective that trains & evaluates per‑subject.
  3. Run a study (with optional storage) and report best parameters.

Usage:
    python gat_cnn_optuna.py \
      --base-dir /path/to/data \
      --results-dir ./results_optuna \
      --study-name cvep_gatcnn_opt \
      --n-trials 100 \
      --n-jobs 1
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import optuna
from typing import Optional
from scipy.signal import butter, decimate, iirnotch, lfilter
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (accuracy_score, log_loss,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.class_weight import compute_class_weight
from spektral.layers import GATConv, GlobalAvgPool, GlobalMaxPool
from tensorflow.keras import backend as K, layers, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# ─── Fixed defaults & “best” hyperparameters ───────────────────────────────────

# Channel setup (keep as in paper)
CHANNEL_NAMES = [
    "Gtec_EEG_P7","Gtec_EEG_P3","Gtec_EEG_Pz","Gtec_EEG_P4","Gtec_EEG_P8",
    "Gtec_EEG_PO7","Gtec_EEG_PO3","Gtec_EEG_POz","Gtec_EEG_PO4","Gtec_EEG_PO8",
    "Gtec_EEG_O1","Gtec_EEG_Oz","Gtec_EEG_O2","Gtec_EEG_O9","Gtec_EEG_Iz","Gtec_EEG_O10"
]
NUM_CHANNELS = len(CHANNEL_NAMES)
NUM_CLASSES  = 2

# 2‑D electrode coords (for spatial graph)
CHANNEL_POSITIONS = {
    "Gtec_EEG_P7":  (-0.611, 0.620), "Gtec_EEG_P3": (-0.447, 0.665),
    "Gtec_EEG_Pz":  ( 0.003, 0.684), "Gtec_EEG_P4": ( 0.470, 0.663),
    "Gtec_EEG_P8":  ( 0.616, 0.616), "Gtec_EEG_PO7":(-0.463, 0.823),
    "Gtec_EEG_PO3": (-0.308, 0.851), "Gtec_EEG_POz":( 0.002, 0.862),
    "Gtec_EEG_PO4": ( 0.310, 0.851), "Gtec_EEG_PO8":( 0.470, 0.823),
    "Gtec_EEG_O1":  (-0.248, 0.948), "Gtec_EEG_Oz": ( 0.001, 0.969),
    "Gtec_EEG_O2":  ( 0.252, 0.946), "Gtec_EEG_O9": (-0.251, 0.966),
    "Gtec_EEG_Iz":  ( 0.000, -1.000),"Gtec_EEG_O10":( 0.483, 0.864)
}
CHANNEL_COORDS = np.array([CHANNEL_POSITIONS[ch] for ch in CHANNEL_NAMES])

# “Original best” (from paper) — these stay fixed, except where Optuna overrides
ORIGINAL_BEST_HP = {
    "decim": 1, "window_samps": 17, "stride_samps": 1, "label_shift_ms": 0,
    "segment_initial_discard_s": 0.3, "use_notch": True, "notch_q": 25,
    "bp_low": 1.3114326655036879, "bp_high": 52.89500856512153,
    "bp_order": 4, "norm": "none", "act": "relu", "use_residual": False,
    "cnn_blocks": 3, "kernel_time": 9, "num_filters": 16,
    "use_attn": True, "attn_heads": 4, "attn_dim": 8,
    "gat_layers": 1, "gat_heads": 3, "gat_ch": 16,
    "pool": "avg", "graph_type": "spatial_knn",
    "knn_k": 5, "radius_epsilon": 0.5, "knn_symmetric": True,
    "opt": "Adam", "lr": 0.0011030607799882711,
    "dropout": 0.0692760250402743,
    "l2": 6.11883728942024e-07, "batch": 64, "mom": 0.969414440481507
}
# fixed subset that we never tune
FIXED_HP = {k: ORIGINAL_BEST_HP[k] for k in [
    "decim","window_samps","stride_samps","label_shift_ms",
    "segment_initial_discard_s","use_notch","notch_q",
    "bp_low","bp_high","bp_order","norm","act",
    "use_residual","cnn_blocks","num_filters","use_attn",
    "attn_heads","attn_dim","gat_layers","gat_heads",
    "gat_ch","pool","opt","mom"
]}

# will be populated once, then re‑used in each trial
ALL_SUBJECT_DATA = []


# ─── Data loading & preprocessing ──────────────────────────────────────────────

def apply_causal_bandpass(x, low, high, order, fs):
    """Causal Butterworth bandpass."""
    nyq = 0.5 * fs
    low_n, high_n = low / nyq, high / nyq
    if not (0 < low_n < high_n < 1):
        return x.astype(np.float32)
    b, a = butter(order, [low_n, high_n], btype="band")
    return lfilter(b.astype(np.float32), a.astype(np.float32), x, axis=0)


def apply_causal_notch(x, freq, q, fs):
    """Causal IIR notch at `freq` Hz."""
    if not (fs > 2 * freq > 0 and q > 0):
        return x.astype(np.float32)
    b, a = iirnotch(freq, q, fs)
    return lfilter(b.astype(np.float32), a.astype(np.float32), x, axis=0)


def make_windows_from_segment(raw, lbl, hp, fs):
    """
    Window a single segment into overlapping windows,
    with decimation, notch, bandpass, and label shift.
    """
    # 1) decimate if asked
    if hp["decim"] > 1 and raw.shape[0] >= hp["decim"] * 5:
        raw = decimate(raw, q=hp["decim"], axis=0, ftype="iir", zero_phase=False)
        lbl = lbl[::hp["decim"]]
        fs /= hp["decim"]
    # 2) notch & bandpass
    if hp["use_notch"] and fs > 100:
        raw = apply_causal_notch(raw, 50.0, hp["notch_q"], fs)
    raw = apply_causal_bandpass(raw, hp["bp_low"], hp["bp_high"], hp["bp_order"], fs)
    # 3) drop initial
    d0 = int(hp["segment_initial_discard_s"] * fs)
    raw, lbl = raw[d0:], lbl[d0:]
    # 4) slide windows
    W, S = hp["window_samps"], hp["stride_samps"]
    shift = int(hp["label_shift_ms"] / 1000.0 * fs)
    Xs, ys = [], []
    for start in range(0, raw.shape[0] - W + 1, S):
        Xs.append(raw[start : start + W])
        idx = min(max(0, start - shift), len(lbl) - 1)
        ys.append(lbl[idx])
    if not Xs:
        return (np.zeros((0, W, NUM_CHANNELS), np.float32),
                np.zeros((0,), int))
    return np.stack(Xs, 0), np.array(ys, int)


def build_adj(hp, coords):
    """
    Build static adjacency: spatial_knn | spatial_radius | fully_connected.
    """
    N = coords.shape[0]
    adj = np.zeros((N, N), np.float32)
    typ = hp.get("graph_type", "fully_connected")

    if typ == "spatial_knn":
        sparse = kneighbors_graph(coords,
                                  n_neighbors=hp["knn_k"],
                                  mode="connectivity",
                                  include_self=False)
        adj = sparse.toarray()
        if hp.get("knn_symmetric", True):
            adj = np.maximum(adj, adj.T)

    elif typ == "spatial_radius":
        dmat = squareform(pdist(coords, metric="euclidean"))
        adj = (dmat < hp["radius_epsilon"]).astype(np.float32)

    else:
        adj = np.ones((N, N), np.float32)

    np.fill_diagonal(adj, 1.0)
    return adj

"""
    Data Load every Subject*/cVEP4/*.csv ....highly spcific to current structure/recording
    """
def load_and_preprocess_all_subject_data(base_dir: Path,
                                         test_size: float,
                                         fs: float,
                                         fixed_hp: dict):
    global ALL_SUBJECT_DATA
    if ALL_SUBJECT_DATA:
        return

    csv_paths = sorted(base_dir.glob("Subject*/cVEP4/*.csv"))
    logging.info("Found %d files under %s", len(csv_paths), base_dir)

    for path in csv_paths:
        df = pd.read_csv(path, skiprows=1, encoding="latin-1")

        # 1) check for missing EEG channels
        missing_ch = set(CHANNEL_NAMES) - set(df.columns)
        if missing_ch:
            logging.warning("Skipping %s: missing channel columns %s",
                            path.name, sorted(missing_ch))
            continue

        # 2) check for CVEP columns
        if not {"CVEP_isHighlighted", "CVEP_flickering_state"}.issubset(df.columns):
            logging.warning("Skipping %s: missing CVEP columns", path.name)
            continue

        # 3) build segments list
        df["h"]   = df["CVEP_isHighlighted"].astype(float)
        df["chg"] = df["h"].diff().fillna(0).ne(0)
        df["blk"] = df["chg"].cumsum().where(df["h"] == 1.0)
        segments = [g for _, g in df[df["h"] == 1.0].groupby("blk")]

        if len(segments) < 2:
            logging.warning("Skipping %s: only %d segments", path.name, len(segments))
            continue

        # 4) now split into train/val
        tr_segs, va_segs = train_test_split(
            segments,
            test_size=test_size,
            random_state=args.seed,
            shuffle=True
        )

        def make_xy_list(segs):
            Xs, ys = [], []
            for seg in segs:
                raw = seg[CHANNEL_NAMES].values.astype(np.float32)
                lbl = seg["CVEP_flickering_state"].values.astype(int)
                Xw, yw = make_windows_from_segment(raw, lbl, fixed_hp, fs)
                if Xw.size:
                    Xs.append(Xw); ys.append(yw)
            if not Xs:
                return None, None
            return np.concatenate(Xs, 0), np.concatenate(ys, 0)

        X_tr, y_tr = make_xy_list(tr_segs)
        X_va, y_va = make_xy_list(va_segs)
        if X_tr is None or X_va is None:
            logging.warning("Skipping %s: no windows generated", path.name)
            continue

        ALL_SUBJECT_DATA.append({
            "id": path.stem,
            "X_tr": X_tr, "y_tr": y_tr,
            "X_va": X_va, "y_va": y_va
        })

    logging.info("Preprocessed %d subjects", len(ALL_SUBJECT_DATA))


# ─── Model + Optuna objective ───────────────────────────────────────────────────

def build_model(hp):
    """
    Build the GAT‑CNN per the paper, with hyperparameters in `hp`.
    """
    N = NUM_CHANNELS
    act_fn = layers.ReLU() if hp["act"] == "relu" else layers.LeakyReLU(0.1)

    # Inputs
    X_in = layers.Input((hp["window_samps"], N), name="X", dtype=tf.float32)
    A_in = layers.Input((N, N), name="A", dtype=tf.float32)

    # reshape for Conv2D
    x = layers.Reshape((hp["window_samps"], N, 1))(X_in)
    x = layers.Permute((2, 1, 3))(x)

    # CNN blocks
    for _ in range(hp["cnn_blocks"]):
        x = layers.Conv2D(
            hp["num_filters"],
            (1, hp["kernel_time"]),
            padding="same",
            kernel_regularizer=regularizers.l2(hp["l2"])
        )(x)
        x = layers.BatchNormalization()(x)
        x = act_fn(x)
        x = layers.MaxPool2D((1, 2), padding="same")(x)
        x = layers.Dropout(hp["dropout"])(x)

    # flatten time+filters
    s = K.int_shape(x)
    x = layers.Reshape((s[1], s[2] * s[3]))(x)

    # self‑attention
    if hp["use_attn"]:
        mha = layers.MultiHeadAttention(
            num_heads=hp["attn_heads"],
            key_dim=hp["attn_dim"],
            dropout=hp["dropout"],
            kernel_regularizer=regularizers.l2(hp["l2"])
        )
        att_out = mha(x, x)
        x = layers.Add()([x, att_out])
        x = layers.LayerNormalization()(x)

    # GAT layers
    for i in range(hp["gat_layers"]):
        x = GATConv(
            channels=hp["gat_ch"],
            attn_heads=hp["gat_heads"],
            concat_heads=True,
            dropout_rate=hp["dropout"],
            activation=hp["act"],
            kernel_regularizer=regularizers.l2(hp["l2"]),
            attn_kernel_regularizer=regularizers.l2(hp["l2"])
        )([x, A_in])

    # pooling & head
    if hp["pool"] == "avg":
        x = GlobalAvgPool()(x)
    elif hp["pool"] == "max":
        x = GlobalMaxPool()(x)
    else:
        x = layers.Concatenate()([
            GlobalAvgPool()(x), GlobalMaxPool()(x)
        ])
    x = layers.Dropout(hp["dropout"])(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="out")(x)

    model = tf.keras.Model([X_in, A_in], out)

    # optimizer
    opt_kwargs = {"learning_rate": hp["lr"], "beta_1": hp["mom"]}
    if hp["opt"] == "AdamW":
        optimizer = tfa.optimizers.AdamW(weight_decay=hp.get("wd", 1e-5), **opt_kwargs)
    else:
        optimizer = optimizers.Adam(**opt_kwargs)

    model.compile(optimizer, "categorical_crossentropy", ["accuracy"])
    return model


def objective(trial: optuna.Trial):
    """
    Optuna objective: sample hyperparameters, train & evaluate on each subject,
    and return the average validation loss across subjects.
    """
    logging.info(f"▶️ Starting trial #{trial.number}")

    # 1) Copy fixed HP and sample the ones to tune
    hp = FIXED_HP.copy()
    hp["kernel_time"] = trial.suggest_int("kernel_time", 5, 13, step=2)
    hp["graph_type"]  = trial.suggest_categorical("graph_type", ["spatial_knn", "spatial_radius"])
    if hp["graph_type"] == "spatial_knn":
        hp["knn_k"]         = trial.suggest_int("knn_k", 3, 8)
        hp["knn_symmetric"] = trial.suggest_categorical("knn_symmetric", [True, False])
    else:
        hp["radius_epsilon"] = trial.suggest_float("radius_epsilon", 0.2, 1.0)
    hp["dropout"]    = trial.suggest_float("dropout", 0.05, 0.45)
    hp["l2"]         = trial.suggest_float("l2", 1e-8, 1e-4, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    hp["lr"]         = trial.suggest_float("lr", 5e-5, 5e-3, log=True)

    # 2) Build static adjacency based on this trial’s graph_type
    A_static = build_adj(hp, CHANNEL_COORDS)

    losses = []
    accs   = []

    # 3) Loop over preprocessed subjects
    for subj in ALL_SUBJECT_DATA:
        K.clear_session()
        X_tr, y_tr = subj["X_tr"], subj["y_tr"]
        X_va, y_va = subj["X_va"], subj["y_va"]

        A_tr = np.tile(A_static, (len(X_tr), 1, 1))
        A_va = np.tile(A_static, (len(X_va), 1, 1))

        y_tr_c = to_categorical(y_tr, NUM_CLASSES)
        y_va_c = to_categorical(y_va, NUM_CLASSES)

        # class weights if needed
        classes, counts = np.unique(y_tr, return_counts=True)
        cw = None
        if len(classes) == NUM_CLASSES and counts.min() > 0:
            w = compute_class_weight("balanced", classes=classes, y=y_tr)
            cw = dict(zip(classes, w))

        # build & train
        model = build_model(hp)
        es = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
        model.fit(
            [X_tr, A_tr], y_tr_c,
            validation_data=([X_va, A_va], y_va_c),
            epochs=args.epochs_optuna,
            batch_size=hp["batch_size"],
            callbacks=[es],
            class_weight=cw,
            verbose=0
        )

        # evaluate
        loss, acc = model.evaluate([X_va, A_va], y_va_c,
                                   batch_size=hp["batch_size"],
                                   verbose=0)
        losses.append(loss)
        accs.append(acc)

    # 4) Aggregate & report
    avg_loss = float(np.mean(losses))
    logging.info(f"✔️ Trial #{trial.number} finished, avg_val_loss={avg_loss:.4f}")

    trial.report(avg_loss, step=0)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return avg_loss


def print_best_callback(study, trial):
    best = study.best_trial
    logging.info(
        "🏆 Best so far → trial #%d: val_loss=%.4f  params=%s",
        best.number, best.value, best.params
    )



# ─── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Optuna tuning for cVEP GAT-CNN"
    )
    p.add_argument(
        "--base-dir", required=True,
        help="Path to root data directory (where Subject*/cVEP4/*.csv live)"
    )
    p.add_argument(
        "--results-dir", default="./results",
        help="Directory to save study results"
    )
    p.add_argument(
        "--study-name", default="cvep_gatcnn_opt",
        help="Optuna study name (used as SQLite DB name if --storage omitted)"
    )
    p.add_argument(
        "--storage", default=None,
        help="Optuna storage URL (e.g. sqlite:///my_study.db). If omitted, uses sqlite:///<study-name>.db"
    )
    p.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of each subject’s segments to hold out for validation"
    )
    p.add_argument(
        "--fs", type=float, default=256.0,
        help="Sampling frequency (Hz)"
    )
    p.add_argument(
        "--epochs-optuna", type=int, default=50,
        help="Maximum epochs per trial"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    p.add_argument(
        "--n-trials", type=int, default=100,
        help="Total number of Optuna trials"
    )
    p.add_argument(
        "--n-jobs", type=int, default=1,
        help="Number of parallel jobs for Optuna"
    )
    args = p.parse_args()

    # now pass args into your main flow...
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    base_dir = Path(args.base_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) Preprocess all subject data once
    load_and_preprocess_all_subject_data(
        base_dir, args.test_size, args.fs, ORIGINAL_BEST_HP
    )
    if not ALL_SUBJECT_DATA:
        raise RuntimeError("No data preprocessed; check --base-dir")

    # 2) Create or load study
    storage = args.storage or f"sqlite:///{args.study_name}.db"
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner()
    )

    # 3) Enqueue original best as initial trial if empty
    if not study.trials:
        init = {
            "kernel_time": ORIGINAL_BEST_HP["kernel_time"],
            "graph_type":  ORIGINAL_BEST_HP["graph_type"],
            "dropout":     ORIGINAL_BEST_HP["dropout"],
            "l2":          ORIGINAL_BEST_HP["l2"],
            "batch_size":  ORIGINAL_BEST_HP["batch"],
            "lr":          ORIGINAL_BEST_HP["lr"]
        }
        if ORIGINAL_BEST_HP["graph_type"] == "spatial_knn":
            init["knn_k"] = ORIGINAL_BEST_HP["knn_k"]
        else:
            init["radius_epsilon"] = ORIGINAL_BEST_HP["radius_epsilon"]
        study.enqueue_trial(init)
        logging.info("Enqueued initial (paper) hyperparameters")

    # 4) Optimize
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        callbacks=[print_best_callback]
    )

    # 5) Summary
    best = study.best_trial
    logging.info("=== Study completed: best #%d loss=%.4f params=%s",
                 best.number, best.value, best.params)

    # 6) Save dataframe of all trials
    df = study.trials_dataframe()
    df.to_csv(results_dir / "all_trials.csv", index=False)
    logging.info("Saved all trials to %s", results_dir / "all_trials.csv")
