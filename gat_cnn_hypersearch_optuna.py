import os
import glob
import json
import time
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical

# Spektral
import spektral
from spektral.layers import GATConv, GlobalAvgPool

# EEG filtering
from scipy.signal import butter, filtfilt, iirnotch

###############################################################################
# 1) USER SETTINGS
###############################################################################
BASE_DIR = r"C:\Dev\newstudy_data\VEPdata"
SUBJECT_LIST = ["Subject 4", "Subject 5", "Subject 10"]  # The 3 subjects
SUBFOLDERS   = ["cVEP1", "cVEP4"]

WINDOW_SIZE   = 80
STRIDE        = 1
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
BANDPASS      = True
NOTCH         = True
NORMALIZE     = True
FS            = 256
NUM_CLASSES   = 2

CHANNEL_NAMES = [
    "Gtec_EEG_P7", "Gtec_EEG_P3", "Gtec_EEG_Pz", "Gtec_EEG_P4", "Gtec_EEG_P8",
    "Gtec_EEG_PO7", "Gtec_EEG_PO3", "Gtec_EEG_POz", "Gtec_EEG_PO4", "Gtec_EEG_PO8",
    "Gtec_EEG_O1", "Gtec_EEG_Oz", "Gtec_EEG_O2", "Gtec_EEG_O9", "Gtec_EEG_Iz", "Gtec_EEG_O10"
]
FLICK_COL  = "CVEP_flickering_state"
HIGH_COL   = "CVEP_isHighlighted"
NUM_CHANNELS = len(CHANNEL_NAMES)

###############################################################################
# 2) HELPER FUNCTIONS
###############################################################################
def build_fully_connected_adjacency(n=16):
    A = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(A, 1.0)
    return A

def bandpass(data, low=0.1, high=30.0, fs=256, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def notch(data, freq=50.0, fs=256, q=30):
    b, a = iirnotch(freq / (0.5 * fs), q)
    return filtfilt(b, a, data, axis=0)

def norm_data(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def create_sliding_windows(df, window_size=80, stride=1):
    active = df[df[HIGH_COL] == 1.0].copy()
    Xvals = active[CHANNEL_NAMES].values
    Yvals = active[FLICK_COL].values.astype(int)
    X_list, y_list = [], []
    for start in range(0, len(Xvals) - window_size + 1, stride):
        window = Xvals[start:start+window_size]
        if BANDPASS:
            window = bandpass(window, fs=FS)
        if NOTCH:
            window = notch(window, fs=FS)
        if NORMALIZE:
            window = norm_data(window)
        X_list.append(window)
        y_list.append(Yvals[start])
    return np.array(X_list), np.array(y_list, dtype=int)

def load_data_for_subjects(subj_list):
    """
    Load and concatenate all data from the given subjects' cVEP1/cVEP4 subfolders.
    Returns X_all, y_all.
    """
    X_all = []
    y_all = []
    for subj in subj_list:
        for subf in SUBFOLDERS:
            cvep_path = os.path.join(BASE_DIR, subj, subf)
            if os.path.isdir(cvep_path):
                for csv_file in glob.glob(os.path.join(cvep_path, "*.csv")):
                    try:
                        df = pd.read_csv(csv_file, skiprows=1, encoding='latin-1')
                        X_win, y_win = create_sliding_windows(df, WINDOW_SIZE, STRIDE)
                        X_all.append(X_win)
                        y_all.append(y_win)
                    except Exception as e:
                        print(f"Error loading file {csv_file}: {e}")
    if len(X_all) == 0:
        return None, None
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return X_all, y_all

###############################################################################
# 3) CUSTOM GAT LAYER
###############################################################################
class MyGATConv(tf.keras.layers.Layer):
    def __init__(self, channels, attn_heads, concat_heads=True, activation=None, **kwargs):
        super(MyGATConv, self).__init__(**kwargs)
        self.gat = GATConv(
            channels=channels,
            attn_heads=attn_heads,
            concat_heads=concat_heads,
            activation=activation,
            return_attn_coef=True
        )
        self.attn_coef = None

    def call(self, inputs):
        output, attn_coef = self.gat(inputs)
        self.attn_coef = attn_coef
        return output

###############################################################################
# 4) MODEL BUILDER WITH MANY HYPERPARAMS
###############################################################################
def build_cnn_gnn_model(
    dropout_rate,
    learning_rate,
    l2_reg,
    num_filters,
    gat_heads,
    gat_channels,
    num_cnn_blocks,
    convs_per_block,
    kernel_size_first,
    kernel_size_later,
    activation_choice,
    decay_rate,
    optimizer_choice
):
    from tensorflow.keras import layers, regularizers

    # Map string -> actual activation
    def activation_layer(x):
        if activation_choice == 'relu':
            return tf.nn.relu(x)
        elif activation_choice == 'elu':
            return tf.nn.elu(x)
        elif activation_choice == 'leaky_relu':
            return tf.nn.leaky_relu(x, alpha=0.2)
        else:
            return tf.nn.relu(x)  # fallback

    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    X_in = layers.Input(shape=(WINDOW_SIZE, NUM_CHANNELS), name="X_in")
    A_in = layers.Input(shape=(NUM_CHANNELS, NUM_CHANNELS), name="A_in")

    # Permute + reshape
    x = layers.Permute((2, 1))(X_in)
    x = layers.Reshape((NUM_CHANNELS, WINDOW_SIZE, 1))(x)

    # CNN blocks
    for block_idx in range(num_cnn_blocks):
        # pick kernel size
        if block_idx < 2:
            ksize = kernel_size_first
        else:
            ksize = kernel_size_later

        for c_idx in range(convs_per_block):
            x = layers.Conv2D(
                num_filters, (1, ksize),
                padding='same',
                kernel_regularizer=reg
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Lambda(activation_layer)(x)

        x = layers.MaxPooling2D(pool_size=(1,2))(x)
        x = layers.Dropout(dropout_rate)(x)

    # Flatten
    x = layers.Reshape((NUM_CHANNELS, -1))(x)
    x = layers.Dense(256, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Lambda(activation_layer)(x)
    x = layers.Dropout(dropout_rate)(x)

    # GAT portion
    gat_layer = MyGATConv(channels=gat_channels, attn_heads=gat_heads,
                          concat_heads=True, activation=None)
    x_out = gat_layer([x, A_in])
    x_out = layers.Dropout(dropout_rate * 0.5)(x_out)

    x_out = GlobalAvgPool()(x_out)
    x_out = layers.Dropout(dropout_rate)(x_out)

    x_out = layers.Dense(128, kernel_regularizer=reg)(x_out)
    x_out = layers.Lambda(activation_layer)(x_out)
    x_out = layers.Dropout(dropout_rate)(x_out)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=reg)(x_out)

    # learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=decay_rate,
        staircase=True
    )

    # pick optimizer
    if optimizer_choice == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        # RMSProp as an alternative
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

    model = tf.keras.Model([X_in, A_in], outputs)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=False
    )
    model.my_gat_layer = gat_layer
    return model

###############################################################################
# 5) LOAD + PREPARE DATA
###############################################################################
X_all, y_all = load_data_for_subjects(SUBJECT_LIST)
if X_all is None or len(X_all) < 10:
    print("No data or too few windows loaded. Exiting.")
    exit()

print("Combined shape from these 3 subjects:", X_all.shape, y_all.shape)

A_fixed = build_fully_connected_adjacency(NUM_CHANNELS)
A_all   = np.tile(A_fixed, (len(X_all), 1, 1))

###############################################################################
# 6) DEFINE TRAIN/VAL SPLIT + OBJECTIVE
###############################################################################
def train_val_split(X, y, A):
    idx_train, idx_val = train_test_split(
        np.arange(len(X)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    X_train, y_train = X[idx_train], y[idx_train]
    X_val,   y_val   = X[idx_val], y[idx_val]
    A_train  = A[idx_train]
    A_val    = A[idx_val]
    return X_train, y_train, X_val, y_val, A_train, A_val

def objective(trial):
    # 1) Hyperparams
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    l2_reg = trial.suggest_float("l2_reg", 0.0, 1e-3, log=False)
    num_filters = trial.suggest_categorical("num_filters", [64, 128, 192, 256])
    gat_heads = trial.suggest_categorical("gat_heads", [2, 4, 8])
    gat_channels = trial.suggest_categorical("gat_channels", [32, 64, 96])
    num_cnn_blocks = trial.suggest_int("num_cnn_blocks", 2, 5)
    convs_per_block = trial.suggest_int("convs_per_block", 1, 2)
    kernel_size_first = trial.suggest_categorical("kernel_size_first", [3, 5, 7])
    kernel_size_later = trial.suggest_categorical("kernel_size_later", [3, 5, 7])
    activation_choice = trial.suggest_categorical("activation", ["relu", "elu", "leaky_relu"])
    decay_rate = trial.suggest_float("decay_rate", 0.90, 0.99, step=0.01)
    optimizer_choice = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

    # 2) Split data
    X_train, y_train, X_val, y_val, A_train, A_val = train_val_split(X_all, y_all, A_all)

    # 3) Build model
    model = build_cnn_gnn_model(
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        l2_reg=l2_reg,
        num_filters=num_filters,
        gat_heads=gat_heads,
        gat_channels=gat_channels,
        num_cnn_blocks=num_cnn_blocks,
        convs_per_block=convs_per_block,
        kernel_size_first=kernel_size_first,
        kernel_size_later=kernel_size_later,
        activation_choice=activation_choice,
        decay_rate=decay_rate,
        optimizer_choice=optimizer_choice
    )

    # 4) Prepare data for training
    y_train_oh = to_categorical(y_train, NUM_CLASSES)
    y_val_oh   = to_categorical(y_val,   NUM_CLASSES)

    # 5) Train
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        x=[X_train, A_train],
        y=y_train_oh,
        validation_data=([X_val, A_val], y_val_oh),
        epochs=40,          # fewer epochs for big search
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop]
    )

    # 6) Evaluate
    val_loss, val_acc = model.evaluate([X_val, A_val], y_val_oh, verbose=0)
    return val_acc

###############################################################################
# 7) RUN OPTUNA
###############################################################################
def run_optuna_search(n_trials=100):
    study = optuna.create_study(
    study_name="my_cvep_study",
    storage="sqlite:///my_cvep_study.db",  # Creates/uses "my_cvep_study.db" in current dir
    load_if_exists=True,                   # Resume if file already exists
    direction="maximize"
)

    # Enqueue your known best param set from prior experience
    # (example best param values)
    best_known_params = {
        'dropout_rate': 0.1,
        'learning_rate': 0.0002718156376992547,
        'l2_reg': 5.371848345403141e-05,
        'num_filters': 256,
        'gat_heads': 2,
        'gat_channels': 32,
        'num_cnn_blocks': 5,
        'convs_per_block': 2,
        'kernel_size_first': 5,
        'kernel_size_later': 7,
        'activation': 'leaky_relu',
        'decay_rate': 0.91,
        'optimizer': 'adam',
        'batch_size': 512
    }
    study.enqueue_trial(best_known_params)

    # Now run the optimization
    study.optimize(objective, n_trials=n_trials)

    print("\nBest params:", study.best_params)
    print("Best value (Val Accuracy):", study.best_value)
    return study

###############################################################################
# 8) MAIN
###############################################################################
if __name__ == "__main__":
    study = run_optuna_search(n_trials=200)  # or more, if you have time!

    # Print final best hyperparams
    print("Best hyperparams found by Optuna:", study.best_params)
    print("Best val_acc found:", study.best_value)

    # Optional final model training on entire dataset with best params
    # ...
