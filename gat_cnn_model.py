#!/usr/bin/env python3
"""
Author: Milan Andras Fodor (@milaniusz)
Project: https://github.com/MILANIUSZ/cVEP-GAT-CNN
Used in: Graph-Attentive CNN for cVEP-BCI with Insights into Electrode Significance (IWANN 2025)

End‑to‑end training & evaluation script for cVEP data. (setup for single subject training)

Usage:
    python gat_cnn_model.py \
        --base-dir "/path/to/data" \
        --held-out "Subject x","Subject y","Subject z \
        --test-size 0.2 \
        --fs 256 \
        --epochs 100 \
        --seed 42
"""

import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from typing import Optional
from scipy.signal import butter, decimate, iirnotch, lfilter
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (accuracy_score, log_loss,
                             precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight
from spektral.layers import GATConv, GlobalAvgPool, GlobalMaxPool
from tensorflow.keras import backend as K, layers, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# ─── Default channel info & HPs (do not modify unless retraining) ────────────

CHANNEL_NAMES = [
    "Gtec_EEG_P7", "Gtec_EEG_P3", "Gtec_EEG_Pz", "Gtec_EEG_P4", "Gtec_EEG_P8",
    "Gtec_EEG_PO7", "Gtec_EEG_PO3", "Gtec_EEG_POz", "Gtec_EEG_PO4", "Gtec_EEG_PO8",
    "Gtec_EEG_O1", "Gtec_EEG_Oz", "Gtec_EEG_O2", "Gtec_EEG_O9", "Gtec_EEG_Iz", "Gtec_EEG_O10"
]
NUM_CHANNELS = len(CHANNEL_NAMES)
NUM_CLASSES = 2

# 2‑D electrode positions (for graph construction)
CHANNEL_POSITIONS = {
    "Gtec_EEG_P7":  (-0.6713, -0.5254),
    "Gtec_EEG_P3":  (-0.3870, -0.5156),
    "Gtec_EEG_Pz":  ( 0.0000, -0.5277),
    "Gtec_EEG_P4":  ( 0.3870, -0.5156),
    "Gtec_EEG_P8":  ( 0.6713, -0.5254),
    "Gtec_EEG_PO7": (-0.4878, -0.7090),
    "Gtec_EEG_PO3": (-0.3051, -0.6920),
    "Gtec_EEG_POz": ( 0.0000, -0.7236),
    "Gtec_EEG_PO4": ( 0.3051, -0.6920),
    "Gtec_EEG_PO8": ( 0.4878, -0.7090),
    "Gtec_EEG_O1":  (-0.2564, -0.8269),
    "Gtec_EEG_Oz":  ( 0.0000, -0.8675),
    "Gtec_EEG_O2":  ( 0.2564, -0.8269),
    "Gtec_EEG_O9":  (-0.2836, -0.9106),
    "Gtec_EEG_Iz":  ( 0.0000, -0.9555),
    "Gtec_EEG_O10": ( 0.2836, -0.9106),
}
CHANNEL_COORDS = np.array([CHANNEL_POSITIONS[ch] for ch in CHANNEL_NAMES])

#best hp from optuan manual paste or import json)
BEST_HP = {
    "decim": 1,
    "window_samps": 17,
    "stride_samps": 1,
    "label_shift_ms": 0,
    "segment_initial_discard_s": 0.3,
    "use_notch": True,
    "notch_q": 25,
    "bp_low": 1.3114326655036879,
    "bp_high": 52.89500856512153,
    "bp_order": 4,
    "act": "relu",
    "use_residual": False,
    "cnn_blocks": 3,
    "kernel_time": 9,
    "num_filters": 16,
    "use_attn": True,
    "attn_heads": 4,
    "attn_dim": 8,
    "gat_layers": 1,
    "gat_heads": 3,
    "gat_ch": 16,
    "pool": "avg",
    "graph_type": "spatial_radius",
    "knn_k": 6,
    "radius_epsilon": 0.333,
    "knn_symmetric": True,
    "opt": "Adam",
    "lr": 0.0011030607799882711,
    "dropout": 0.0692760250402743,
    "l2": 6.11883728942024e-07,
    "batch": 128,
    "mom": 0.969414440481507,
    "patience": 30
}


# ─── Signal preprocessing ────────────────────────────────────────────────────

def apply_causal_bandpass(x: np.ndarray, low: float, high: float,
                          order: int, fs: float) -> np.ndarray:
    """Apply a causal Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low_n, high_n = low / nyq, high / nyq
    if not (0 < low_n < high_n < 1):
        return x.astype(np.float32)
    b, a = butter(order, [low_n, high_n], btype="band")
    return lfilter(b.astype(np.float32), a.astype(np.float32), x, axis=0)


def apply_causal_notch(x: np.ndarray, freq: float, q: float,
                       fs: float) -> np.ndarray:
    """Apply a causal IIR notch filter."""
    if not (fs > 2 * freq > 0 and q > 0):
        return x.astype(np.float32)
    b, a = iirnotch(freq, q, fs)
    return lfilter(b.astype(np.float32), a.astype(np.float32), x, axis=0)


def make_windows_from_segment(
    raw: np.ndarray,
    lbl: np.ndarray,
    hp: dict,
    fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cut one continuous segment into overlapping windows, with optional
    decimation, notch, bandpass, and label-shift.
    """
    # decimate (if requested)
    if hp["decim"] > 1 and raw.shape[0] >= hp["decim"] * 5:
        raw = decimate(raw, q=hp["decim"], axis=0, ftype="iir", zero_phase=False)
        lbl = lbl[::hp["decim"]]
        fs /= hp["decim"]

    # optional 50 Hz notch on high‑fs data
    if hp["use_notch"] and fs > 100:
        raw = apply_causal_notch(raw, 50.0, hp["notch_q"], fs)

    # bandpass
    raw = apply_causal_bandpass(
        raw, hp["bp_low"], hp["bp_high"], hp["bp_order"], fs
    )

    # drop initial samples
    d0 = int(hp["segment_initial_discard_s"] * fs)
    raw, lbl = raw[d0:], lbl[d0:]

    W, S = hp["window_samps"], hp["stride_samps"]
    shift_samples = int(hp["label_shift_ms"] / 1_000 * fs)

    Xs, ys = [], []
    for start in range(0, raw.shape[0] - W + 1, S):
        Xs.append(raw[start : start + W])
        # ensure label index is in-bounds
        lbl_idx = min(max(0, start - shift_samples), len(lbl) - 1)
        ys.append(lbl[lbl_idx])

    if not Xs:
        empty_shape = (0, hp["window_samps"], NUM_CHANNELS)
        return np.zeros(empty_shape, np.float32), np.zeros((0,), int)

    return np.stack(Xs, axis=0), np.array(ys, dtype=int)


# ─── Graph adjacency ─────────────────────────────────────────────────────────

def build_adj(hp: dict, coords: np.ndarray) -> np.ndarray:
    """
    Build a static adjacency matrix based on the chosen graph_type:
    - spatial_knn
    - spatial_radius
    - fully_connected
    """
    N = coords.shape[0]
    adj = np.zeros((N, N), dtype=np.float32)

    if hp["graph_type"] == "spatial_knn":
        from sklearn.neighbors import kneighbors_graph

        sparse = kneighbors_graph(
            coords,
            n_neighbors=hp["knn_k"],
            mode="connectivity",
            include_self=False,
        )
        adj = sparse.toarray()
        if hp.get("knn_symmetric", True):
            adj = np.maximum(adj, adj.T)

    elif hp["graph_type"] == "spatial_radius":
        dist = squareform(pdist(coords, metric="euclidean"))
        adj = (dist < hp["radius_epsilon"]).astype(np.float32)

    elif hp["graph_type"] == "fully_connected":
        adj = np.ones((N, N), dtype=np.float32)

    else:
        logging.warning(
            "Unknown graph_type '%s'; defaulting to fully_connected",
            hp["graph_type"],
        )
        adj = np.ones((N, N), dtype=np.float32)

    # add self‑loops
    np.fill_diagonal(adj, 1.0)
    return adj


# ─── Model building ──────────────────────────────────────────────────────────

def build_model(
    hp: dict,
    return_gat_attn: bool = False,
    return_mha_attn: bool = False
) -> tf.keras.Model:
    """
    Construct the spatio‑temporal GAT+CNN model.
    """
    N = NUM_CHANNELS
    act_fn = layers.ReLU() if hp["act"] == "relu" else layers.LeakyReLU(0.1)

    # Inputs
    X_in = layers.Input((hp["window_samps"], N), name="X", dtype=tf.float32)
    A_in = layers.Input((N, N), name="A", dtype=tf.float32)

    # reshape for 2D conv: (batch, channels, time, 1)
    x = layers.Reshape((hp["window_samps"], N, 1))(X_in)
    x = layers.Permute((2, 1, 3))(x)

    # temporal CNN blocks
    for _ in range(hp["cnn_blocks"]):
        shortcut = x
        x = layers.Conv2D(
            hp["num_filters"],
            kernel_size=(1, hp["kernel_time"]),
            padding="same",
            kernel_regularizer=regularizers.l2(hp["l2"]),
        )(x)
        x = layers.BatchNormalization()(x)
        x = act_fn(x)
        if hp["use_residual"] and shortcut.shape == x.shape:
            x = layers.Add()([x, shortcut])
        x = layers.MaxPool2D((1, 2), padding="same")(x)
        x = layers.Dropout(hp["dropout"])(x)

    # flatten time+filters
    s = K.int_shape(x)
    x = layers.Reshape((s[1], s[2] * s[3]))(x)

    # optional self‑attention
    if hp["use_attn"]:
        mha = layers.MultiHeadAttention(
            num_heads=hp["attn_heads"],
            key_dim=hp["attn_dim"],
            dropout=hp["dropout"],
            kernel_regularizer=regularizers.l2(hp["l2"]),
            name="mha_self",
        )
        if return_mha_attn:
            att_out, att_scores = mha(x, x, return_attention_scores=True)
        else:
            att_out = mha(x, x)
            att_scores = None
        x = layers.Add()([x, att_out])
        x = layers.LayerNormalization()(x)

    # GAT layers
    gat_coefs = []
    for i in range(hp["gat_layers"]):
        gat = GATConv(
            channels=hp["gat_ch"],
            attn_heads=hp["gat_heads"],
            concat_heads=True,
            dropout_rate=hp["dropout"],
            activation=hp["act"],
            kernel_regularizer=regularizers.l2(hp["l2"]),
            attn_kernel_regularizer=regularizers.l2(hp["l2"]),
            return_attn_coef=return_gat_attn,
            name=f"gat_{i}",
        )
        if return_gat_attn:
            x, coef = gat([x, A_in])
            gat_coefs.append(coef)
        else:
            x = gat([x, A_in])

    # global pooling & output
    if hp["pool"] == "avg":
        x = GlobalAvgPool()(x)
    elif hp["pool"] == "max":
        x = GlobalMaxPool()(x)
    else:
        x = layers.Concatenate()(
            [GlobalAvgPool()(x), GlobalMaxPool()(x)]
        )
    x = layers.Dropout(hp["dropout"])(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="out")(x)

    outputs = [out]
    if return_mha_attn and att_scores is not None:
        outputs.append(layers.Lambda(lambda t: t, name="mha_attn")(att_scores))
    for idx, coef in enumerate(gat_coefs):
        outputs.append(layers.Lambda(lambda t: t, name=f"gat_attn_{idx}")(coef))

    model = tf.keras.Model(inputs=[X_in, A_in], outputs=outputs)

    # optimizer
    opt_kwargs = {"learning_rate": hp["lr"], "beta_1": hp["mom"]}
    if hp["opt"] == "AdamW":
        optimizer = tfa.optimizers.AdamW(weight_decay=hp.get("wd", 1e-5), **opt_kwargs)
    else:
        optimizer = optimizers.Adam(**opt_kwargs)

    # compile
    loss = "categorical_crossentropy"
    model.compile(optimizer, loss, metrics=["accuracy"])
    return model


# ─── Permutation Feature Importance ─────────────────────────────────────────

def compute_pfi(
    model: tf.keras.Model,
    X_val: np.ndarray,
    A_val: np.ndarray,
    y_val: np.ndarray,
    base_acc: float,
    batch_size: int,
    seed: int
) -> np.ndarray:
    """
    Compute permutation feature importance (drop in accuracy per channel).
    """
    rng = np.random.RandomState(seed)
    pfi_vals = np.zeros(NUM_CHANNELS, dtype=float)

    for ch in range(NUM_CHANNELS):
        Xp = X_val.copy()
        perm = rng.permutation(len(Xp))
        Xp[:, :, ch] = X_val[perm, :, ch]

        probs = model.predict([Xp, A_val], batch_size=batch_size, verbose=0)
        if isinstance(probs, list):
            probs = probs[0]
        preds = np.argmax(probs, axis=1)
        pfi_vals[ch] = base_acc - accuracy_score(y_val, preds)

    return pfi_vals


# ─── File processing ─────────────────────────────────────────────────────────

def process_file(
    path: Path,
    label: str,
    hp: dict,
    fs: float,
    test_size: float,
    seed: int,
    adj_matrix: np.ndarray
) -> Optional[dict]:
    """
    Load one CSV, segment into windows, train/evaluate on that file,
    and return metrics dict or None if skipped.
    """
    fn = path.stem
    logging.info("Processing [%s] %s", label, fn)

    # --- load & validate ---
    df = pd.read_csv(path, skiprows=1, encoding="latin-1")
    missing = set(CHANNEL_NAMES) - set(df.columns)
    if missing:
        logging.warning("  ↳ skipping %s: missing channels %s", fn, sorted(missing))
        return None
    if not {"CVEP_isHighlighted", "CVEP_flickering_state"}.issubset(df.columns):
        logging.warning("  ↳ skipping %s: missing CVEP columns", fn)
        return None

    # --- identify highlighted segments ---
    df["h"] = df["CVEP_isHighlighted"].astype(float)
    df["chg"] = df["h"].diff().fillna(0).ne(0)
    df["blk"] = df["chg"].cumsum().where(df["h"] == 1.0)
    segments = [g for _, g in df[df["h"] == 1.0].groupby("blk")]
    if not segments:
        logging.warning("  ↳ no highlighted segments in %s", fn)
        return None

    # --- train/val split of segments ---
    rng = np.random.RandomState(seed)
    rng.shuffle(segments)
    split = int(len(segments) * (1 - test_size))
    train_segs, val_segs = segments[:split], segments[split:]
    if not train_segs or not val_segs:
        logging.warning("  ↳ not enough segments in %s", fn)
        return None

    def make_xy(segs):
        Xs, ys = [], []
        for seg in segs:
            raw = seg[CHANNEL_NAMES].values.astype(np.float32)
            lbl = seg["CVEP_flickering_state"].values.astype(int)
            Xw, yw = make_windows_from_segment(raw, lbl, hp, fs)
            if Xw.size:
                Xs.append(Xw)
                ys.append(yw)
        if not Xs:
            return None, None
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

    X_tr, y_tr = make_xy(train_segs)
    X_va, y_va = make_xy(val_segs)
    if X_tr is None or X_va is None:
        logging.warning("  ↳ no windows for %s", fn)
        return None

    # --- tile adjacency & one-hot labels & class weights ---
    A_tr = np.tile(adj_matrix, (len(X_tr), 1, 1))
    A_va = np.tile(adj_matrix, (len(X_va), 1, 1))
    y_tr_cat = to_categorical(y_tr, NUM_CLASSES)
    y_va_cat = to_categorical(y_va, NUM_CLASSES)

    cw = None
    classes, counts = np.unique(y_tr, return_counts=True)
    if len(classes) == NUM_CLASSES:
        weights = compute_class_weight("balanced", classes=classes, y=y_tr)
        cw = dict(zip(classes, weights))

    # --- model build & training ---
    K.clear_session()
    tf.random.set_seed(seed)
    model = build_model(hp)
    es = EarlyStopping(
        monitor="val_loss",
        patience=hp["patience"],
        restore_best_weights=True,
        verbose=1
    )
    model.fit(
        [X_tr, A_tr],
        y_tr_cat,
        validation_data=([X_va, A_va], y_va_cat),
        epochs=args.epochs,
        batch_size=hp["batch"],
        class_weight=cw,
        callbacks=[es],
        verbose=1
    )

    # --- evaluation ---
    def eval_split(X, A, y):
        probs = model.predict([X, A], batch_size=hp["batch"], verbose=0)
        if isinstance(probs, list):
            probs = probs[0]
        preds = np.argmax(probs, axis=1)
        return {
            "loss": log_loss(y, probs, labels=list(range(NUM_CLASSES))),
            "acc": accuracy_score(y, preds),
            **dict(zip(
                ["precision", "recall", "f1", "_"],
                precision_recall_fscore_support(
                    y, preds, average="binary", zero_division=0
                )
            ))
        }

    tr_metrics = eval_split(X_tr, A_tr, y_tr)
    va_metrics = eval_split(X_va, A_va, y_va)

    return {
        "file": fn,
        "path": str(path),
        "set": label,
        "train_loss": float(tr_metrics["loss"]),
        "train_acc": float(tr_metrics["acc"]),
        "val_loss": float(va_metrics["loss"]),
        "val_acc": float(va_metrics["acc"]),
        "val_precision": float(va_metrics["precision"]),
        "val_recall": float(va_metrics["recall"]),
        "val_f1": float(va_metrics["f1"]),
    }


# ─── Main execution ───────────────────────────────────────────────────────────

def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    base = Path(args.base_dir)
    all_files = sorted(base.glob("Subject*/cVEP4/*.csv"))

    held_out = set(args.held_out.split(","))
    trainval = [
        f for f in all_files
        if f.parent.parent.name not in held_out
    ]
    holdout = [
        f for f in all_files
        if f.parent.parent.name in held_out
    ]

    logging.info("Found %d train+val files, %d held-out files",
                 len(trainval), len(holdout))

    # build adjacency matrix
    A_static = build_adj(BEST_HP, CHANNEL_COORDS)

    # process all
    results = []
    for fpath in trainval:
        res = process_file(
            fpath, "trainval", BEST_HP,
            args.fs, args.test_size, args.seed, A_static
        )
        if res:
            results.append(res)

    for fpath in holdout:
        res = process_file(
            fpath, "holdout", BEST_HP,
            args.fs, args.test_size, args.seed, A_static
        )
        if res:
            results.append(res)

    # save
    out_dir = Path(args.save_dir or f"./results_{BEST_HP['graph_type']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / (args.output_csv or f"results_{BEST_HP['graph_type']}.csv")

    df = pd.DataFrame(results)
    cols = [
        "file", "path", "set",
        "train_loss", "train_acc",
        "val_loss", "val_acc",
        "val_precision", "val_recall", "val_f1"
    ]
    df = df[cols]
    df.to_csv(out_csv, index=False, float_format="%.6f")
    logging.info("✅ Results written to %s", out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & evaluate cVEP GAT‑CNN")
    parser.add_argument(
        "--base-dir", required=True,
        help="Path to your root data directory (e.g. where each subject's EEG CSVs live)"
    )
    parser.add_argument(
        "--held-out", default="",
        help=(
            "Comma‑separated list of subject IDs to hold out (e.g. \"38,41,42\"); "
            "leave empty to use all subjects"
        )
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of segments to reserve for validation"
    )
    parser.add_argument(
        "--fs", type=float, default=256.0,
        help="Sampling frequency (Hz)"
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="Directory to save results (overrides default)"
    )
    parser.add_argument(
        "--output-csv", default=None,
        help="Filename for results CSV (overrides default)"
    )

    args = parser.parse_args()
    main(args)

