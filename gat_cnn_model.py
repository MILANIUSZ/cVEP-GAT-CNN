#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gat_cnn_model.py
===========
Author: Milan Andras Fodor (@milaniusz)
Project: https://github.com/MILANIUSZ/cVEP-GAT-CNN
Used in: Graph-Attentive CNN for cVEP-BCI with Insights into Electrode Significance (IWANN 2025)
Date: 2025-03-05
"""

import argparse
import logging
import sys
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from scipy.signal import butter, lfilter, iirnotch, decimate
from tensorflow.keras import layers, regularizers, optimizers, backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from spektral.layers import GATConv, GlobalAvgPool, GlobalMaxPool
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss

# ─── CORE LOGIC (unchanged) ───────────────────────────────────────────────────

CHANNEL_NAMES = [
    "Gtec_EEG_P7","Gtec_EEG_P3","Gtec_EEG_Pz","Gtec_EEG_P4","Gtec_EEG_P8",
    "Gtec_EEG_PO7","Gtec_EEG_PO3","Gtec_EEG_POz","Gtec_EEG_PO4","Gtec_EEG_PO8",
    "Gtec_EEG_O1","Gtec_EEG_Oz","Gtec_EEG_O2","Gtec_EEG_O9","Gtec_EEG_Iz","Gtec_EEG_O10"
]
NUM_CHANNELS = len(CHANNEL_NAMES)
NUM_CLASSES  = 2

DEFAULT_BEST_HP = {
    "decim": 1,
    "window_samps": 30,
    "stride_samps": 2,
    "label_shift_ms": 0,
    "segment_initial_discard_s": 0.3,
    "use_notch": True,
    "notch_q": 25,
    "bp_low": 1.3114326655036879,
    "bp_high": 52.89500856512153,
    "bp_order": 4,
    "norm": "none",
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
    "graph_type": "spatial",
    "opt": "Adam",
    "lr": 0.0011030607799882711,
    "dropout": 0.0692760250402743,
    "l2": 6.11883728942024e-07,
    "batch": 64,
    "mom": 0.969414440481507
}

def apply_causal_bandpass(x, low, high, order, fs):
    nyq = 0.5 * fs
    low_n, high_n = low/nyq, high/nyq
    if not (0 < low_n < high_n < 1):
        return x.astype(np.float32)
    b, a = butter(order, [low_n, high_n], btype='band')
    return lfilter(b.astype(np.float32), a.astype(np.float32), x, axis=0)

def apply_causal_notch(x, freq, q, fs):
    if not (fs > 2*freq > 0 and q > 0):
        return x.astype(np.float32)
    b, a = iirnotch(freq, q, fs)
    return lfilter(b.astype(np.float32), a.astype(np.float32), x, axis=0)

def make_windows_from_segment(raw, lbl, hp, fs):
    if hp['decim'] > 1 and raw.shape[0] >= hp['decim']*5:
        raw = decimate(raw, q=hp['decim'], axis=0, ftype='iir', zero_phase=False)
        lbl = lbl[::hp['decim']]
        fs /= hp['decim']
    if hp['use_notch'] and fs > 100:
        raw = apply_causal_notch(raw, 50.0, hp['notch_q'], fs)
    raw = apply_causal_bandpass(raw, hp['bp_low'], hp['bp_high'], hp['bp_order'], fs)
    d0 = int(hp['segment_initial_discard_s'] * fs)
    raw, lbl = raw[d0:], lbl[d0:]
    W, S = hp['window_samps'], hp['stride_samps']
    shift = int(hp['label_shift_ms']/1000.0 * fs)
    Xs, ys = [], []
    for i in range(0, raw.shape[0]-W+1, S):
        Xs.append(raw[i:i+W])
        j = min(max(0, i-shift), len(lbl)-1)
        ys.append(lbl[j])
    if Xs:
        return np.stack(Xs, 0), np.array(ys, int)
    else:
        return np.zeros((0, W, NUM_CHANNELS), np.float32), np.zeros((0,), int)

def build_adj(_hp):
    return np.ones((NUM_CHANNELS, NUM_CHANNELS), np.float32)

def build_model(hp, return_attn=False):
    act_fn = layers.ReLU() if hp['act']=='relu' else layers.LeakyReLU(0.1)
    X_in = layers.Input((hp['window_samps'], NUM_CHANNELS), name='X', dtype=tf.float32)
    A_in = layers.Input((NUM_CHANNELS, NUM_CHANNELS), name='A', dtype=tf.float32)

    # CNN trunk
    mult = NUM_CHANNELS
    x = layers.Reshape((hp['window_samps'], NUM_CHANNELS, mult))(X_in)
    x = layers.Permute((2,1,3))(x)
    for i in range(hp['cnn_blocks']):
        x0 = x
        x  = layers.Conv2D(hp['num_filters'], (1, hp['kernel_time']),
                           padding='same', kernel_regularizer=regularizers.l2(hp['l2']))(x)
        x  = layers.BatchNormalization()(x)
        x  = act_fn(x)
        if hp['use_residual']:
            x = layers.Add()([x, x0])
        x  = layers.MaxPool2D((1,2), padding='same')(x)
        x  = layers.Dropout(hp['dropout'])(x)
    b1,b2,b3,b4 = K.int_shape(x)
    x = layers.Reshape((b2, b3*b4))(x)

    # optional self-attn
    if hp['use_attn']:
        att = layers.MultiHeadAttention(
            num_heads=hp['attn_heads'],
            key_dim=hp['attn_dim'],
            dropout=hp['dropout'],
            kernel_regularizer=regularizers.l2(hp['l2'])
        )(x, x)
        x   = layers.Add()([x, att])
        x   = layers.LayerNormalization()(x)

    # GAT layers
    attn_coefs = []
    for i in range(hp['gat_layers']):
        gat = GATConv(
            channels=hp['gat_ch'],
            attn_heads=hp['gat_heads'],
            concat_heads=True,
            dropout_rate=hp['dropout'],
            activation=hp['act'],
            kernel_regularizer=regularizers.l2(hp['l2']),
            attn_kernel_regularizer=regularizers.l2(hp['l2']),
            return_attn_coef=return_attn,
            name=f'gat_{i}'
        )
        out = gat([x, A_in])
        if return_attn:
            x, coef = out
            attn_coefs.append(coef)
        else:
            x = out

    if   hp['pool']=='avg': x = GlobalAvgPool()(x)
    elif hp['pool']=='max': x = GlobalMaxPool()(x)
    else:                   x = layers.Concatenate()([GlobalAvgPool()(x), GlobalMaxPool()(x)])
    x      = layers.Dropout(hp['dropout'])(x)
    logits = layers.Dense(NUM_CLASSES, activation='softmax', name='out')(x)

    outputs = [logits] + attn_coefs if return_attn else logits
    model   = tf.keras.Model([X_in, A_in], outputs)

    opt_kws = {'learning_rate':hp['lr'], 'beta_1':hp['mom']}
    if hp['opt']=='AdamW':
        opt = tfa.optimizers.AdamW(weight_decay=hp['wd'], **opt_kws)
    else:
        opt = optimizers.Adam(**opt_kws)

    if return_attn:
        model.compile(opt, {'out':'categorical_crossentropy'}, metrics=['accuracy'])
    else:
        model.compile(opt, 'categorical_crossentropy', ['accuracy'])
    return model

def compute_pfi(model, Xv, Av, yv, base_acc, seed, batch):
    rng = np.random.RandomState(seed)
    pfi = np.zeros(NUM_CHANNELS)
    for c in range(NUM_CHANNELS):
        Xp = Xv.copy()
        perm = rng.permutation(len(Xp))
        Xp[:,:,c] = Xv[perm,:,c]
        probs = model.predict([Xp, Av], batch_size=batch)
        preds = np.argmax(probs, axis=1)
        pfi[c] = base_acc - accuracy_score(yv, preds)
    return pfi

# ─── WRAPPER & MAIN ──────────────────────────────────────────────────────────

def process_file(path, hp, args, logger):
    fn = path.stem
    logger.info(f"[{fn}]")

    df0 = pd.read_csv(path, skiprows=1, encoding="latin-1")
    required = {'CVEP_isHighlighted','CVEP_flickering_state'}
    if required - set(df0.columns):
        logger.warning(" missing required columns; skipping")
        return None

    df0['h']   = df0['CVEP_isHighlighted'].astype(float)
    df0['chg'] = df0['h'].diff().fillna(0).ne(0)
    df0['blk'] = df0['chg'].cumsum().where(df0['h']==1.0)
    segs = [seg for _,seg in df0[df0['h']==1.0].groupby('blk')]
    if not segs:
        logger.warning(" no highlighted segments; skipping")
        return None

    rng = np.random.RandomState(args.seed)
    rng.shuffle(segs)
    ntr = int(len(segs)*(1-args.test_size))
    train_segs, val_segs = segs[:ntr], segs[ntr:]
    if not train_segs or not val_segs:
        logger.warning(" insufficient segments; skipping")
        return None

    # build windows
    X_tr_list, y_tr_list = [], []
    for seg in train_segs:
        raw = seg[CHANNEL_NAMES].values.astype(np.float32)
        lbl = seg['CVEP_flickering_state'].values.astype(int)
        Xs, ys = make_windows_from_segment(raw, lbl, hp, args.fs)
        if Xs.size:
            X_tr_list.append(Xs); y_tr_list.append(ys)
    if not X_tr_list:
        logger.warning(" no train windows; skipping"); return None
    X_tr = np.concatenate(X_tr_list, 0); y_tr = np.concatenate(y_tr_list, 0)

    X_va_list, y_va_list = [], []
    for seg in val_segs:
        raw = seg[CHANNEL_NAMES].values.astype(np.float32)
        lbl = seg['CVEP_flickering_state'].values.astype(int)
        Xs, ys = make_windows_from_segment(raw, lbl, hp, args.fs)
        if Xs.size:
            X_va_list.append(Xs); y_va_list.append(ys)
    if not X_va_list:
        logger.warning(" no val windows; skipping"); return None
    X_va = np.concatenate(X_va_list,0); y_va = np.concatenate(y_va_list,0)

    A = build_adj(hp)
    A_tr = np.tile(A, (len(X_tr),1,1))
    A_va = np.tile(A, (len(X_va),1,1))

    u, counts = np.unique(y_tr, return_counts=True)
    cw = None
    if len(u)==NUM_CLASSES:
        w = compute_class_weight('balanced', classes=u, y=y_tr)
        cw = dict(zip(u, w))

    y_tr_cat = to_categorical(y_tr, NUM_CLASSES)
    y_va_cat = to_categorical(y_va, NUM_CLASSES)

    K.clear_session()
    model = build_model(hp, return_attn=False)
    es    = EarlyStopping('val_loss', patience=10, restore_best_weights=True, verbose=0)
    model.fit(
        [X_tr, A_tr], y_tr_cat,
        validation_data=([X_va, A_va], y_va_cat),
        epochs=args.epochs,
        batch_size=hp['batch'],
        callbacks=[es],
        verbose=0,
        class_weight=cw
    )

    probs_va = model.predict([X_va, A_va], batch_size=hp['batch'])
    val_loss = log_loss(y_va, probs_va, labels=list(range(NUM_CLASSES)))
    val_pred = np.argmax(probs_va, axis=1)
    val_acc  = accuracy_score(y_va, val_pred)
    p,r,f1,_ = precision_recall_fscore_support(y_va, val_pred, average='binary', zero_division=0)

    row = {
        'file': fn, 'path': str(path),
        'train_loss': float(model.evaluate([X_tr, A_tr], y_tr_cat, verbose=0)[0]),
        'train_acc' : float(model.evaluate([X_tr, A_tr], y_tr_cat, verbose=0)[1]),
        'val_loss'  : float(val_loss),
        'val_acc'   : float(val_acc),
        'val_precision': float(p),
        'val_recall'   : float(r),
        'val_f1'       : float(f1)
    }

    logger.info(" • PFI")
    pfi = compute_pfi(model, X_va, A_va, y_va, val_acc, args.seed, DEFAULT_BEST_HP['batch'])
    for j, ch in enumerate(CHANNEL_NAMES):
        row[f'PFI_{ch}'] = float(pfi[j])

    logger.info(" • GAT attention")
    attn_model = build_model(hp, return_attn=True)
    attn_model.set_weights(model.get_weights())
    preds = attn_model.predict(
        [X_va[:1000], A_va[:1000]],
        batch_size=hp['batch']
    )
    coefs   = preds[1:]
    agg_imp = np.zeros(NUM_CHANNELS, np.float32)
    for c in coefs:
        hl = c.mean(axis=(0,1))
        agg_imp += hl.sum(axis=0)
    agg_imp /= (agg_imp.sum() + 1e-9)
    for j, ch in enumerate(CHANNEL_NAMES):
        row[f'GAT_{ch}'] = float(agg_imp[j])

    return row

def main():
    p = argparse.ArgumentParser(description="Evaluate per-file group with PFI & GAT")
    p.add_argument("--data-dir",    type=Path,   default=Path("."),  help="Root with Subject*/cVEP4/*.csv")
    p.add_argument("--save-dir",    type=Path,   default=Path("./results"), help="Directory to save outputs")
    p.add_argument("--output-csv",  type=str,    default="results.csv", help="Name of the output CSV")
    p.add_argument("--test-size",   type=float,  default=0.2,          help="Validation split fraction")
    p.add_argument("--fs",          type=float,  default=256.0,        help="Sampling frequency")
    p.add_argument("--epochs",      type=int,    default=100,          help="Epochs per training")
    p.add_argument("--seed",        type=int,    default=42,           help="Random seed")
    args = p.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV = args.save_dir / args.output_csv

    log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt, handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger("evaluate_by_file_group")

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    pattern = str(args.data_dir / "Subject*" / "cVEP4" / "*.csv")
    paths = sorted(Path(p) for p in glob.glob(pattern))
    logger.info(f"Found {len(paths)} files")

    results = []
    for path in paths:
        row = process_file(path, DEFAULT_BEST_HP, args, logger)
        if row is not None:
            results.append(row)

    if not results:
        logger.error("No valid results to save; exiting.")
        sys.exit(1)

    df = pd.DataFrame(results)
    cols = [
      'file','path','train_loss','train_acc',
      'val_loss','val_acc','val_precision','val_recall','val_f1'
    ]
    for ch in CHANNEL_NAMES:
        cols += [f'PFI_{ch}', f'GAT_{ch}']
    df[cols].to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Wrote results to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
