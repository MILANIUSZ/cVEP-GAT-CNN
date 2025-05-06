#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Author: Milan Andras Fodor
Github:@milaniusz
Used in: Graph-Attentive CNN for cVEP-BCI with Insights into Electrode Significance (IWANN 2025)
Date: 2025-03-05

"""

import os, glob, json
import numpy as np, pandas as pd, tensorflow as tf
from datetime import datetime
from scipy.signal import butter, sosfilt, iirnotch, lfilter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from spektral.layers import GATConv, GlobalAvgPool

# ───────────────────── Settings ─────────────────────
BASE_DIR       = r"......"
OUTPUT_SUMMARY = "results.csv"
SAVE_DIR       = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

FS             = 256
WINDOW_SIZE    = 80
STRIDE         = 1
TEST_SIZE      = 0.2
RND_SEED       = 42

# If your experiment used phases and boxes:
VALID_PHASE    = 3
VALID_BOXES    = [1,2,3]

HP = {
    'band_low': 0.1, 'band_high': 30.0, 'filter_order': 5,
    'notch_freq': 50.0, 'notch_q': 30,
    'normalize': 'session',    # 'session' or 'window'
    'batch_size': 256, 'epochs': 120,
    'learning_rate': 2.718e-4, 'decay_rate': 0.91,
    'dropout': 0.1, 'l2_reg': 5.37e-5,
    'num_filters': 256, 'cnn_blocks': 5, 'convs_per_block': 2,
    'kernel_first': 5, 'kernel_later': 7,
    'gat_heads': 2, 'gat_channels': 32, 'gat_layers': 3
}

CHANNELS = [
    "Gtec_EEG_P7","Gtec_EEG_P3","Gtec_EEG_Pz","Gtec_EEG_P4","Gtec_EEG_P8",
    "Gtec_EEG_PO7","Gtec_EEG_PO3","Gtec_EEG_POz","Gtec_EEG_PO4","Gtec_EEG_PO8",
    "Gtec_EEG_O1","Gtec_EEG_Oz","Gtec_EEG_O2","Gtec_EEG_O9","Gtec_EEG_Iz","Gtec_EEG_O10"
]

np.random.seed(RND_SEED)
tf.random.set_seed(RND_SEED)

# ───────────────────── DSP Helpers ─────────────────────
def design_sos(low, high, fs, order):
    nyq = 0.5 * fs
    return butter(order, [low/nyq, high/nyq], 'band', output='sos')

def preprocess_window(win, mean, std):
    sos = design_sos(HP['band_low'], HP['band_high'], FS, HP['filter_order'])
    win = sosfilt(sos, win, axis=0)
    b, a = iirnotch(HP['notch_freq']/(0.5*FS), HP['notch_q'])
    win = lfilter(b, a, win, axis=0)
    if HP['normalize']=='session':
        return (win - mean) / (std + 1e-6)
    else:
        return (win - win.mean(0)) / (win.std(0) + 1e-6)

# ───────────────────── Window Generation ─────────────────────
def create_windows(df):
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Filter by phase & box if available
    if 'hhcvep_phase_label' in df.columns:
        df = df[df['hhcvep_phase_label'] == VALID_PHASE]
    if 'hhcvep_box_label' in df.columns:
        df = df[df['hhcvep_box_label'].isin(VALID_BOXES)]
    
    # Detect flicker-state column
    flick_cols = [c for c in df.columns if 'flicker' in c]
    if not flick_cols:
        print(f"ERROR: no flicker column found in {df.columns.tolist()}")
        return None, None
    flick_key = flick_cols[0]
    
    # Optionally filter highlighted trials
    highlight_cols = [c for c in df.columns if 'highlight' in c]
    if highlight_cols:
        df = df[df[highlight_cols[0]] == 1.0]
    
    # Extract data & labels
    missing = set(ch.lower() for ch in CHANNELS) - set(df.columns)
    if missing:
        print(f"ERROR: missing EEG channels {missing}")
        return None, None
    
    Xall = df[[c.lower() for c in CHANNELS]].values
    Yall = df[flick_key].astype(int).values
    
    # Session stats
    mean, std = Xall.mean(0), Xall.std(0)
    std[std == 0] = 1.0

    X, Y = [], []
    for i in range(0, len(Xall)-WINDOW_SIZE+1, STRIDE):
        win = Xall[i:i+WINDOW_SIZE]
        X.append(preprocess_window(win, mean, std))
        Y.append(Yall[i])
    return np.array(X), np.array(Y)

# ───────────────────── Model Builder ─────────────────────
def build_model():
    X_in = layers.Input((WINDOW_SIZE, len(CHANNELS)))
    A_in = layers.Input((len(CHANNELS), len(CHANNELS)))

    # CNN feature extractor
    x = layers.Permute((2,1))(X_in)
    x = layers.Reshape((len(CHANNELS), WINDOW_SIZE,1))(x)
    for blk in range(HP['cnn_blocks']):
        ks = HP['kernel_first'] if blk==0 else HP['kernel_later']
        for _ in range(HP['convs_per_block']):
            x = layers.Conv2D(
                HP['num_filters'], (1,ks), padding='same',
                kernel_regularizer=regularizers.l2(HP['l2_reg'])
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('leaky_relu')(x)
        x = layers.MaxPooling2D((1,2))(x)
        x = layers.Dropout(HP['dropout'])(x)

    # Flatten temporal dims
    shp = tf.keras.backend.int_shape(x)
    x = layers.Reshape((shp[1], shp[2]*shp[3]))(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(HP['l2_reg']))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('leaky_relu')(x)
    x = layers.Dropout(HP['dropout'])(x)

    # Stacked GATConv layers
    for _ in range(HP['gat_layers']):
        x = GATConv(
            HP['gat_channels'], HP['gat_heads'],
            activation='leaky_relu',
            kernel_regularizer=regularizers.l2(HP['l2_reg'])
        )([x, A_in])
        x = layers.Dropout(HP['dropout'])(x)

    # Classifier head
    x = GlobalAvgPool()(x)
    x = layers.Dropout(HP['dropout'])(x)
    x = layers.Dense(
        128, activation='leaky_relu',
        kernel_regularizer=regularizers.l2(HP['l2_reg'])
    )(x)
    x = layers.Dropout(HP['dropout'])(x)
    out = layers.Dense(2, activation='softmax')(x)

    # Compile
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        HP['learning_rate'], decay_steps=1000,
        decay_rate=HP['decay_rate'], staircase=True
    )
    model = tf.keras.Model([X_in, A_in], out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ───────────────────── Train per CSV ─────────────────────
def train_file(path):
    df = pd.read_csv(path, skiprows=1)
    X, y = create_windows(df)
    if X is None or len(X) < 10:
        return None

    idx = np.arange(len(X))
    tr, val = train_test_split(idx, test_size=TEST_SIZE,
                               random_state=RND_SEED, stratify=y)
    X_tr, X_val = X[tr], X[val]
    y_tr, y_val = y[tr], y[val]
    y_tr_oh, y_val_oh = to_categorical(y_tr), to_categorical(y_val)

    A = np.ones((len(CHANNELS), len(CHANNELS)), np.float32)
    A_tr = np.tile(A, (len(X_tr),1,1))
    A_val= np.tile(A, (len(X_val),1,1))

    model = build_model()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt = os.path.join(SAVE_DIR, f"{os.path.basename(path)}_{ts}.h5")
    mc   = ModelCheckpoint(ckpt, monitor='val_accuracy',
                           save_best_only=True, verbose=1)
    es   = EarlyStopping(monitor='val_accuracy', patience=100,
                         restore_best_weights=True, verbose=1)

    model.fit(
        [X_tr, A_tr], y_tr_oh,
        validation_data=([X_val, A_val], y_val_oh),
        batch_size=HP['batch_size'],
        epochs=HP['epochs'],
        callbacks=[mc, es],
        verbose=1
    )

    loss, acc = model.evaluate([X_val, A_val], y_val_oh, verbose=0)
    preds     = model.predict([X_val, A_val], verbose=0).argmax(axis=1)
    cm        = confusion_matrix(y_val, preds).tolist()
    cr        = classification_report(y_val, preds).replace("\n","\\n")

    return {
        'file': os.path.basename(path),
        'val_loss': float(loss),
        'val_acc': float(acc),
        'confusion_matrix': json.dumps(cm),
        'classification_report': cr
    }

# ───────────────────── Main Runner ─────────────────────
def main():
    results = []
    for root, _, files in os.walk(BASE_DIR):
        for f in files:
            if not f.lower().endswith('.csv'): continue
            res = train_file(os.path.join(root, f))
            if res: results.append(res)

    pd.DataFrame(results).to_csv(OUTPUT_SUMMARY, index=False)
    print("Saved summary to", OUTPUT_SUMMARY)

if __name__ == '__main__':
    main()
