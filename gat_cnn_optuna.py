#!/usr/bin/env python3
"""
gat_cnn_optuna.py  
===========
Author: Milan Andras Fodor (@milaniusz)
Project: Graph-Attentive CNN for cVEP-BCI (IWANN 2025)
Date: 2025-03-05

Full Optuna-based hyperparameter tuning pipeline for a GAT-CNN
model used in cVEP-based Brain-Computer Interfaces (BCIs).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import PercentilePruner
from optuna.exceptions import TrialPruned

from scipy.signal import butter, lfilter, iirnotch, decimate
from sklearn.metrics import euclidean_distances
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, regularizers, optimizers, backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from spektral.layers import GATConv, GlobalAvgPool, GlobalMaxPool

# --- Constants (defaults; overridable via CLI) ---
TEST_SIZE = 0.2
FS = 256.0            # original sampling rate
EPOCHS = 100
VERBOSE = 0

CHANNEL_NAMES = [
    "Gtec_EEG_P7","Gtec_EEG_P3","Gtec_EEG_Pz","Gtec_EEG_P4","Gtec_EEG_P8",
    "Gtec_EEG_PO7","Gtec_EEG_PO3","Gtec_EEG_POz","Gtec_EEG_PO4","Gtec_EEG_PO8",
    "Gtec_EEG_O1","Gtec_EEG_Oz","Gtec_EEG_O2","Gtec_EEG_O9","Gtec_EEG_Iz","Gtec_EEG_O10"
]
NUM_CHANNELS = len(CHANNEL_NAMES)
NUM_CLASSES = 2

MSEQ_BIT_DURATION_FRAMES = 4
DISPLAY_FPS = 60.0
MSEQ_BIT_DURATION_S = MSEQ_BIT_DURATION_FRAMES / DISPLAY_FPS

HIGH_COL = "CVEP_isHighlighted"
FLICK_COL = "CVEP_flickering_state"


def apply_causal_bandpass(data, low, high, order, fs):
    nyq = 0.5 * fs
    low_n, high_n = low / nyq, high / nyq
    if not (0 < low_n < high_n < 1):
        return data.astype(np.float32)
    b, a = butter(order, [low_n, high_n], btype="band")
    return lfilter(b.astype(np.float32), a.astype(np.float32), data, axis=0)


def apply_causal_notch(data, freq, q, fs):
    if not (fs > 2 * freq > 0 and q > 0):
        return data.astype(np.float32)
    b, a = iirnotch(freq, q, fs=fs)
    return lfilter(b.astype(np.float32), a.astype(np.float32), data, axis=0)


def create_windows_and_labels(df, hp, base_fs=FS):
    """
    Convert raw dataframe into sliding windows + labels according to hyperparams hp.
    Returns X_windows (n_windows, window_samps, channels), y_labels (n_windows,), effective_fs.
    """
    all_X, all_y = [], []
    current_fs = base_fs / hp["decim"]

    if HIGH_COL not in df.columns:
        return np.empty((0, hp["window_samps"], NUM_CHANNELS)), np.empty((0,)), current_fs

    df = df.copy()
    df["is_highlighted_numeric"] = df[HIGH_COL].astype(float)
    df["block_change"] = df["is_highlighted_numeric"].diff().fillna(0).ne(0)
    df["block_id_raw"] = df["block_change"].cumsum()
    df["block_id"] = df["block_id_raw"].where(df["is_highlighted_numeric"] == 1.0)

    grouped = df[df["is_highlighted_numeric"] == 1.0].groupby("block_id")
    if grouped.ngroups == 0:
        return np.empty((0, hp["window_samps"], NUM_CHANNELS)), np.empty((0,)), current_fs

    for _, seg in grouped:
        eeg = seg[CHANNEL_NAMES].values.astype(np.float32)
        labels = seg[FLICK_COL].values.astype(int)
        fs_seg = base_fs

        # decimation
        if hp["decim"] > 1 and eeg.shape[0] > hp["decim"] * 20:
            try:
                eeg = decimate(eeg, hp["decim"], axis=0, ftype="iir", zero_phase=False)
                labels = labels[::hp["decim"]][: eeg.shape[0]]
                fs_seg /= hp["decim"]
            except ValueError:
                pass

        # notch
        if hp["use_notch"] and fs_seg > 100.0:
            eeg = apply_causal_notch(eeg, 50.0, hp["notch_q"], fs_seg)

        # bandpass
        bp_h = min(hp["bp_high"], fs_seg / 2 - 0.1)
        if hp["bp_low"] < bp_h:
            eeg = apply_causal_bandpass(eeg, hp["bp_low"], bp_h, hp["bp_order"], fs_seg)

        # discard initial
        discard = int(hp["segment_initial_discard_s"] * fs_seg)
        if eeg.shape[0] > discard + hp["window_samps"]:
            eeg = eeg[discard:]
            labels = labels[discard:]
        else:
            continue

        # sliding windows
        shift = int(hp["label_shift_ms"] / 1000 * fs_seg)
        for start in range(0, eeg.shape[0] - hp["window_samps"] + 1, hp["stride_samps"]):
            window = eeg[start : start + hp["window_samps"], :]
            if hp["norm"] == "window_z":
                mu = window.mean(0, keepdims=True)
                sd = window.std(0, keepdims=True) + 1e-7
                window = (window - mu) / sd
            label_idx = start - shift
            if 0 <= label_idx < len(labels):
                all_X.append(window)
                all_y.append(labels[label_idx])

    if not all_X:
        return np.empty((0, hp["window_samps"], NUM_CHANNELS)), np.empty((0,)), current_fs

    return np.stack(all_X), np.array(all_y, dtype=int), current_fs


def build_adjacency_matrix(hp):
    """
    Build adjacency (NUM_CHANNELS x NUM_CHANNELS) per graph_type/radius in hp.
    """
    adj = np.ones((NUM_CHANNELS, NUM_CHANNELS), dtype=np.float32)
    if hp["graph_type"] == "spatial" and hp.get("channel_positions"):
        coords = np.array([hp["channel_positions"][ch] for ch in CHANNEL_NAMES])
        D = euclidean_distances(coords)
        mask = (D <= hp["graph_radius"]).astype(np.float32)
        np.fill_diagonal(mask, 1.0)
        adj = np.maximum(mask, mask.T)
    return adj


def build_model(hp, input_feat_dim):
    act = layers.ReLU() if hp["act"] == "relu" else layers.LeakyReLU(0.1)
    X_in = layers.Input((hp["window_samps"], input_feat_dim), dtype=tf.float32, name="X")
    A_in = layers.Input((NUM_CHANNELS, NUM_CHANNELS), dtype=tf.float32, name="A")
    x = X_in

    # reshape for CNN
    mult = input_feat_dim // NUM_CHANNELS
    x = layers.Reshape((hp["window_samps"], NUM_CHANNELS, mult))(x)
    x = layers.Permute((2, 1, 3))(x)

    for i in range(hp["cnn_blocks"]):
        shortcut = x
        x = layers.Conv2D(hp["num_filters"], (1, hp["kernel_time"]),
                          padding="same", kernel_regularizer=regularizers.l2(hp["l2"]))(x)
        x = layers.BatchNormalization()(x)
        x = act(x)
        if hp["use_residual"]:
            if shortcut.shape[-1] != x.shape[-1]:
                shortcut = layers.Conv2D(x.shape[-1], 1, padding="same",
                                         kernel_regularizer=regularizers.l2(hp["l2"]))(shortcut)
            x = layers.Add()([x, shortcut])
        x = layers.MaxPooling2D((1, 2), padding="same")(x)
        x = layers.Dropout(hp["dropout"])(x)

    shp = K.int_shape(x)
    x = layers.Reshape((shp[1], shp[2] * shp[3]))(x)

    if hp["use_attn"]:
        att = layers.MultiHeadAttention(
            num_heads=hp["attn_heads"],
            key_dim=hp["attn_dim"],
            kernel_regularizer=regularizers.l2(hp["l2"])
        )(x, x)
        x = layers.LayerNormalization()(layers.Add()([x, att]))

    for _ in range(hp["gat_layers"]):
        x = act(GATConv(
            channels=hp["gat_ch"],
            attn_heads=hp["gat_heads"],
            concat_heads=True,
            dropout_rate=hp["dropout"],
            kernel_regularizer=regularizers.l2(hp["l2"])
        )([x, A_in]))

    if hp["pool"] == "avg":
        x = GlobalAvgPool()(x)
    elif hp["pool"] == "max":
        x = GlobalMaxPool()(x)
    else:
        x = layers.Concatenate()([GlobalAvgPool()(x), GlobalMaxPool()(x)])

    x = layers.Dropout(hp["dropout"])(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    opt_kw = {"learning_rate": hp["lr"], "beta_1": hp.get("mom", 0.9)}
    optimizer = (tfa.optimizers.AdamW(weight_decay=hp.get("wd", 0.0), **opt_kw)
                 if hp["opt"] == "AdamW"
                 else optimizers.Adam(**opt_kw))

    model = tf.keras.Model([X_in, A_in], out)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def objective(trial, files, logger):
    # reproducibility
    tf.random.set_seed(trial.number)
    np.random.seed(trial.number)

    hp = {}
    # preprocessing
    hp["decim"] = trial.suggest_categorical("decim", [1, 2])
    eff_fs = FS / hp["decim"]
    bit_samps = int(MSEQ_BIT_DURATION_S * eff_fs)
    min_samps = max(int(0.8 * bit_samps), 5, int(eff_fs * 0.1))
    max_samps = max(min_samps + 1, int(2.5 * bit_samps))
    hp["window_samps"] = trial.suggest_int("window_samps", min_samps, max_samps)
    hp["stride_samps"] = trial.suggest_int(
        "stride_samps", 1,
        min(20, max(1, min(bit_samps // 2, hp["window_samps"] // 2)))
    )
    hp["label_shift_ms"] = trial.suggest_int("label_shift_ms", 0, 200, step=20)
    hp["segment_initial_discard_s"] = trial.suggest_float("segment_initial_discard_s", 0.0, 0.3, step=0.05)
    hp["use_notch"] = trial.suggest_categorical("use_notch", [True, False])
    hp["notch_q"] = trial.suggest_int("notch_q", 20, 40) if hp["use_notch"] else 30
    hp["bp_low"] = trial.suggest_float("bp_low", 0.1, 5.0)
    hp["bp_high"] = trial.suggest_float("bp_high", 30.0, 61.0)
    hp["bp_order"] = trial.suggest_int("bp_order", 2, 5)
    hp["norm"] = trial.suggest_categorical("norm", ["window_z", "none"])

    # model arch
    hp["act"] = trial.suggest_categorical("act", ["relu", "leaky_relu"])
    hp["use_residual"] = trial.suggest_categorical("use_residual", [True, False])
    hp["cnn_blocks"] = trial.suggest_int("cnn_blocks", 1, 3)
    hp["kernel_time"] = trial.suggest_int("kernel_time", 3, 9, step=2)
    hp["num_filters"] = trial.suggest_categorical("num_filters", [8, 16, 24, 32, 48])
    hp["use_attn"] = trial.suggest_categorical("use_attn", [False, True])
    hp["attn_heads"] = trial.suggest_int("attn_heads", 1, 4) if hp["use_attn"] else 1
    hp["attn_dim"] = trial.suggest_categorical("attn_dim", [8, 16]) if hp["use_attn"] else 8
    hp["gat_layers"] = trial.suggest_int("gat_layers", 1, 3)
    hp["gat_heads"] = trial.suggest_int("gat_heads", 1, 4)
    hp["gat_ch"] = trial.suggest_categorical("gat_ch", [8, 16, 24, 32])
    hp["pool"] = trial.suggest_categorical("pool", ["avg", "max", "concat_avg_max"])
    hp["graph_type"] = trial.suggest_categorical("graph_type", ["fully_connected", "spatial"])
    hp["graph_radius"] = trial.suggest_float("graph_radius", 0.1, 1.0) if hp["graph_type"] == "spatial" else 0.5

    # training
    hp["opt"] = trial.suggest_categorical("opt", ["Adam", "AdamW"])
    hp["lr"] = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    hp["dropout"] = trial.suggest_float("dropout", 0.05, 0.5)
    hp["l2"] = trial.suggest_float("l2", 1e-7, 1e-4, log=True)
    hp["batch"] = trial.suggest_categorical("batch", [32, 64, 128])
    hp["wd"] = trial.suggest_float("wd", 1e-6, 1e-3, log=True) if hp["opt"] == "AdamW" else 0.0
    hp["mom"] = trial.suggest_float("mom", 0.85, 0.99)

    # Optionally include channel positions for spatial graph:
    hp["channel_positions"] = {}  # fill if you have real (x,y) dict

    losses, accs = [], []

    for f in files:
        try:
            df = pd.read_csv(f, skiprows=1, encoding="latin-1")
        except Exception as e:
            logger.warning(f"Failed to read {f.name}: {e}")
            continue

        X, y, _ = create_windows_and_labels(df, hp)
        if X.shape[0] == 0:
            continue

        A = np.tile(build_adjacency_matrix(hp), (X.shape[0], 1, 1))
        split = int((1 - TEST_SIZE) * X.shape[0])
        X_tr, X_va = X[:split], X[split:]
        A_tr, A_va = A[:split], A[split:]
        y_tr = to_categorical(y[:split], NUM_CLASSES)
        y_va = to_categorical(y[split:], NUM_CLASSES)

        cw = None
        cls = np.unique(y[:split])
        if len(cls) == NUM_CLASSES:
            weights = compute_class_weight("balanced", cls, y[:split])
            cw = dict(zip(cls, weights))

        tf.keras.backend.clear_session()
        model = build_model(hp, X.shape[2])

        pruning_cb = optuna.integration.TFKerasPruningCallback(trial, "val_loss")
        es = EarlyStopping("val_loss", patience=15, restore_best_weights=True)

        hist = model.fit(
            [X_tr.astype(np.float32), A_tr],
            y_tr.astype(np.float32),
            validation_data=([X_va, A_va], y_va.astype(np.float32)),
            epochs=EPOCHS,
            batch_size=hp["batch"],
            callbacks=[pruning_cb, es],
            class_weight=cw,
            verbose=VERBOSE,
        )

        best_i = np.argmin(hist.history["val_loss"])
        losses.append(hist.history["val_loss"][best_i])
        accs.append(hist.history["val_accuracy"][best_i])

    if not losses:
        return float("inf")

    mean_loss = float(np.mean(losses))
    mean_acc = float(np.mean(accs))
    trial.set_user_attr("mean_val_accuracy", mean_acc)
    return mean_loss


def main():
    p = argparse.ArgumentParser(
        prog="cvep_optuna.py",
        description="Optuna hyperparameter search for GAT-CNN cVEP BCI"
    )
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Root folder with Subject*/cVEP4/*.csv")
    p.add_argument("--output-dir", type=Path, default=Path("."),
                   help="Where to write study DB and logs")
    p.add_argument("--num-files", type=int, default=6,
                   help="Number of CSV files per trial (0=all)")
    p.add_argument("--study-name", type=str,
                   default="cvep_cnn_v3_highfreq",
                   help="Optuna study name")
    p.add_argument("--trials", type=int, default=1000,
                   help="Total number of Optuna trials")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_file = args.output_dir / f"{args.study_name}.db"

    log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_dir / "run.log")
        ]
    )
    logger = logging.getLogger("cvep_optuna")

    all_csv = sorted(args.data_dir.glob("Subject*/cVEP4/*.csv"))
    files = all_csv if args.num_files == 0 else all_csv[: args.num_files]
    if not files:
        logger.error("No CSV files found. Exiting.")
        sys.exit(1)
    logger.info(f"{len(files)} CSV files will be used in each trial.")

    sampler = TPESampler(seed=42, n_startup_trials=15, multivariate=True)
    pruner = PercentilePruner(percentile=33.0, n_startup_trials=10, n_warmup_steps=30)
    storage = f"sqlite:///{db_file}"
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    logger.info(f"Starting study '{args.study_name}' ({len(study.trials)} existing trials)")
    study.optimize(
        lambda t: objective(t, files, logger),
        n_trials=args.trials,
    )

    best = study.best_trial
    logger.info(
        f"Best trial #{best.number}: loss={best.value:.4f}, "
        f"acc={best.user_attrs.get('mean_val_accuracy', 'N/A'):.4f}"
    )
    logger.info(f"Parameters: {json.dumps(best.params, indent=2)}")


if __name__ == "__main__":
    main()
