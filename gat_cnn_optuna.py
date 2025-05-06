import os
import glob
import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from scipy.signal import butter, lfilter, iirnotch
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from spektral.layers import GATConv, GlobalAvgPool

# ─────────── Custom pruning callback ───────────
class OptunaPruningCallback(Callback):
    def __init__(self, trial, monitor: str):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        self.trial.report(current, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Epoch {epoch}: `{self.monitor}` == {current:.5f}")

# ─────────── User settings ───────────
BASE_DIR        = r"C:\Dev\newstudy_data\VEPdata"
WINDOW_SIZE     = 80
STRIDE          = 1
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
FS              = 256
HIGH_COL        = "CVEP_isHighlighted"
FLICK_COL       = "CVEP_flickering_state"
CHANNEL_NAMES   = [
    "Gtec_EEG_P7","Gtec_EEG_P3","Gtec_EEG_Pz","Gtec_EEG_P4","Gtec_EEG_P8",
    "Gtec_EEG_PO7","Gtec_EEG_PO3","Gtec_EEG_POz","Gtec_EEG_PO4","Gtec_EEG_PO8",
    "Gtec_EEG_O1","Gtec_EEG_Oz","Gtec_EEG_O2","Gtec_EEG_O9","Gtec_EEG_Iz","Gtec_EEG_O10"
]
NUM_CHANNELS    = len(CHANNEL_NAMES)
NUM_CLASSES     = 2

# Pick exactly two files to cycle through
all_csv = sorted(glob.glob(os.path.join(BASE_DIR, "Subject*", "cVEP1", "*.csv")))
if len(all_csv) < 2:
    raise RuntimeError("Need at least two CSVs under Subject*/cVEP1/")
CSV_SELECTION = all_csv[:2]

# ─────────── Data prep ───────────
def init_normalization(df):
    act = df[df[HIGH_COL] == 1.0]
    X = act[CHANNEL_NAMES].values
    m, s = X.mean(0), X.std(0)
    s[s == 0] = 1.0
    return m, s

def create_windows(df, hp):
    m, s = init_normalization(df)
    df = df[df[HIGH_COL] == 1.0]
    data = df[CHANNEL_NAMES].values.astype(np.float32)
    labels = df[FLICK_COL].values.astype(int)
    Xs, ys = [], []
    for st in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
        w = data[st:st+WINDOW_SIZE]
        if hp["bandpass"]:
            nyq = 0.5 * FS
            b, a = butter(hp["bp_order"], [hp["bp_low"]/nyq, hp["bp_high"]/nyq], btype='band')
            w = lfilter(b, a, w, axis=0)
        if hp["notch"]:
            b, a = iirnotch(50.0/(0.5*FS), hp["notch_q"])
            w = lfilter(b, a, w, axis=0)
        if hp["normalize"] == "session":
            w = (w - m) / s
        elif hp["normalize"] == "window_z":
            w = (w - w.mean(0)) / (w.std(0) + 1e-6)
        else:
            mn, mx = w.min(0), w.max(0)
            w = (w - mn) / (mx - mn + 1e-6)
        Xs.append(w)
        ys.append(labels[st])
    return np.array(Xs), np.array(ys, dtype=int)

# ─────────── Model builder ───────────
def build_model(hp):
    if hp["activation_name"] == "relu":
        activation = layers.ReLU()
    else:
        activation = layers.LeakyReLU()

    X_in = layers.Input((WINDOW_SIZE, NUM_CHANNELS), name="X_in")
    A_in = layers.Input((NUM_CHANNELS, NUM_CHANNELS), name="A_in")

    x = layers.Permute((2,1))(X_in)
    x = layers.Reshape((NUM_CHANNELS, WINDOW_SIZE, 1))(x)
    for b in range(hp["cnn_blocks"]):
        k = hp["kernel_first"] if b == 0 else hp["kernel_later"]
        for _ in range(hp["convs_per_block"]):
            x = layers.Conv2D(
                hp["num_filters"], (1,k), padding='same',
                kernel_regularizer=regularizers.l2(hp["l2_reg"])
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
        x = layers.MaxPooling2D((1,2))(x)
        x = layers.Dropout(hp["dropout"])(x)

    shp = tf.keras.backend.int_shape(x)
    x = layers.Reshape((shp[1], shp[2]*shp[3]))(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(hp["l2_reg"]))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(hp["dropout"])(x)

    for _ in range(hp["gat_layers"]):
        x = GATConv(
            channels=hp["gat_channels"],
            attn_heads=hp["gat_heads"],
            concat_heads=True,
            activation=activation,
            kernel_regularizer=regularizers.l2(hp["l2_reg"])
        )([x, A_in])
        x = layers.Dropout(hp["dropout"])(x)

    x = GlobalAvgPool()(x)
    x = layers.Dropout(hp["dropout"])(x)
    x = layers.Dense(128, activation=activation)(x)
    x = layers.Dropout(hp["dropout"])(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        hp["lr"], decay_steps=1000, decay_rate=hp["decay"], staircase=True
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_sched)
    model = tf.keras.Model([X_in, A_in], out)
    model.compile(opt, 'categorical_crossentropy', ['accuracy'])
    return model

# ─────────── Objective ───────────
def objective(trial):
    hp = {
        "bandpass":   trial.suggest_categorical("bandpass",   [True, False]),
        "notch":      trial.suggest_categorical("notch",      [True, False]),
        "notch_q":    trial.suggest_int("notch_q",    10, 50),
        "bp_low":     trial.suggest_categorical("bp_low",     [0.1, 1.0]),
        "bp_high":    trial.suggest_categorical("bp_high",    [31.0, 60.0]),
        "bp_order":   trial.suggest_int("bp_order",    2, 8),
        "normalize":  trial.suggest_categorical("normalize",  ["session","window_z","minmax"]),
        "batch_size": trial.suggest_categorical("batch_size",[16,64,128,256,512]),
        "lr":         trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "decay":      trial.suggest_float("decay", 0.8, 0.99),
        "dropout":    trial.suggest_float("dropout",0.0,0.5),
        "l2_reg":     trial.suggest_float("l2_reg",1e-6,1e-3, log=True),
        "num_filters":trial.suggest_categorical("num_filters",[64,128,256,512]),
        "cnn_blocks": trial.suggest_int("cnn_blocks", 2, 6),
        "convs_per_block": trial.suggest_int("convs_per_block",1,3),
        "kernel_first":   trial.suggest_categorical("kernel_first",[3,5,7]),
        "kernel_later":   trial.suggest_categorical("kernel_later",[3,5,7]),
        "gat_heads":      trial.suggest_int("gat_heads", 1, 8),
        "gat_channels":   trial.suggest_categorical("gat_channels",[8,16,32,64]),
        "gat_layers":     trial.suggest_int("gat_layers",1,4),
        "activation_name":trial.suggest_categorical("activation",["relu","leaky_relu"]),
    }

    print(f"\n=== Trial {trial.number} hyperparams ===")
    print(json.dumps(hp, indent=2))

    val_accs = []
    for path in CSV_SELECTION:
        df = pd.read_csv(path, skiprows=1, encoding='latin-1')
        X, y = create_windows(df, hp)
        if len(X) < 10:
            return 0.0

        idx = np.arange(len(X))
        tr, val = train_test_split(idx, test_size=TEST_SIZE,
                                   random_state=RANDOM_STATE, stratify=y)
        X_tr, X_val = X[tr], X[val]
        y_tr = to_categorical(y[tr], NUM_CLASSES)
        y_val = to_categorical(y[val], NUM_CLASSES)

        A = np.ones((NUM_CHANNELS, NUM_CHANNELS), np.float32)
        A_tr = np.tile(A, (len(X_tr),1,1))
        A_val = np.tile(A, (len(X_val),1,1))

        model = build_model(hp)
        pruning_cb = OptunaPruningCallback(trial, "val_accuracy")
        history = model.fit(
            [X_tr, A_tr], y_tr,
            validation_data=([X_val, A_val], y_val),
            batch_size=hp["batch_size"],
            epochs=100,
            callbacks=[pruning_cb],
            verbose=1
        )
        val_accs.append(history.history["val_accuracy"][-1])

    return float(np.mean(val_accs))

if __name__ == "__main__":
    storage = "sqlite:///optuna_cvep_study.db"
    study = optuna.create_study(
        study_name="cvep_opt",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=200)
    print("Best trial:", study.best_trial.params)
