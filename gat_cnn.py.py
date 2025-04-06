import os
import glob
import json
import time
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, regularizers

# Spektral for GNN
from spektral.layers import GATConv, GlobalAvgPool
from scipy.signal import butter, filtfilt, iirnotch

# ------------------------ User Settings ------------------------
BASE_DIR = r"C:\Dev\newstudy_data\VEPdata"
MODEL_SAVE_DIR = "saved_models_24_03_2025"  # Directory to save models
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

WINDOW_SIZE = 80
STRIDE = 1
TEST_SIZE = 0.2
RANDOM_STATE = 42
FS = 256

BANDPASS = True
NOTCH = True
NORMALIZE = True

# Best hyperparameters from Optuna
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 0.0002718
DROPOUT = 0.1
ACTIVATION = tf.keras.layers.LeakyReLU()
L2_REG = 5.371848345403141e-05
DECAY_RATE = 0.91

NUM_FILTERS = 256
GAT_HEADS = 2
GAT_CHANNELS = 32
CNN_BLOCKS = 5
CONVS_PER_BLOCK = 2
KERNEL_FIRST = 5
KERNEL_LATER = 7

CHANNEL_NAMES = [
    "Gtec_EEG_P7", "Gtec_EEG_P3", "Gtec_EEG_Pz", "Gtec_EEG_P4", "Gtec_EEG_P8",
    "Gtec_EEG_PO7", "Gtec_EEG_PO3", "Gtec_EEG_POz", "Gtec_EEG_PO4", "Gtec_EEG_PO8",
    "Gtec_EEG_O1", "Gtec_EEG_Oz", "Gtec_EEG_O2", "Gtec_EEG_O9", "Gtec_EEG_Iz", "Gtec_EEG_O10"
]
FLICK_COL = "CVEP_flickering_state"
HIGH_COL = "CVEP_isHighlighted"
NUM_CHANNELS = len(CHANNEL_NAMES)
NUM_CLASSES = 2

# ------------------------ Preprocessing ------------------------
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
    std = np.std(data, axis=0)
    std[std == 0] = 1
    return (data - np.mean(data, axis=0)) / std

def create_sliding_windows(df, window_size=80, stride=1):
    active = df[df[HIGH_COL] == 1.0].copy()
    Xvals = active[CHANNEL_NAMES].values
    Yvals = active[FLICK_COL].values.astype(int)

    X_list, y_list = [], []
    for start in range(0, len(Xvals) - window_size + 1, stride):
        window = Xvals[start:start + window_size]
        if BANDPASS:
            window = bandpass(window, fs=FS)
        if NOTCH:
            window = notch(window, fs=FS)
        if NORMALIZE:
            window = norm_data(window)
        X_list.append(window)
        y_list.append(Yvals[start])

    return np.array(X_list), np.array(y_list, dtype=int)

# ------------------------ GAT Layer ------------------------
class MyGATConv(tf.keras.layers.Layer):
    def __init__(self, channels, attn_heads, concat_heads=True, activation=None, **kwargs):
        super(MyGATConv, self).__init__(**kwargs)
        self.channels = channels
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.activation = activation
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

    def get_config(self):
        config = super(MyGATConv, self).get_config()
        config.update({
            "channels": self.channels,
            "attn_heads": self.attn_heads,
            "concat_heads": self.concat_heads,
            "activation": tf.keras.activations.serialize(self.activation),
        })
        return config

# ------------------------ Model Builder ------------------------
def build_cnn_gnn_model():
    X_in = layers.Input(shape=(WINDOW_SIZE, NUM_CHANNELS), name="X_in")
    A_in = layers.Input(shape=(NUM_CHANNELS, NUM_CHANNELS), name="A_in")

    x = layers.Permute((2, 1))(X_in)
    x = layers.Reshape((NUM_CHANNELS, WINDOW_SIZE, 1))(x)

    for block in range(CNN_BLOCKS):
        for _ in range(CONVS_PER_BLOCK):
            kernel_size = (1, KERNEL_FIRST) if block == 0 else (1, KERNEL_LATER)
            x = layers.Conv2D(NUM_FILTERS, kernel_size, padding='same',
                              kernel_regularizer=regularizers.l2(L2_REG))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(ACTIVATION)(x)
        x = layers.MaxPooling2D(pool_size=(1, 2))(x)
        x = layers.Dropout(DROPOUT)(x)

    x = layers.Reshape((NUM_CHANNELS, -1))(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION)(x)
    x = layers.Dropout(DROPOUT)(x)

    gat1 = MyGATConv(channels=GAT_CHANNELS, attn_heads=GAT_HEADS, activation=ACTIVATION)
    x = gat1([x, A_in])
    x = layers.Dropout(DROPOUT)(x)
    x = GlobalAvgPool()(x)
    x = layers.Dense(128, activation=ACTIVATION)(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=1000,
        decay_rate=DECAY_RATE,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model = tf.keras.Model(inputs=[X_in, A_in], outputs=outputs)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
    model.my_gat_layers = [gat1]
    return model

# ------------------------ Training Loop ------------------------
def train_on_csv(csv_path, subject_name, subfolder):
    start_time = time.time()
    print(f"\n--- Training on file: {csv_path} ---")

    df = pd.read_csv(csv_path, skiprows=1, encoding='latin-1')
    X_all, y_all = create_sliding_windows(df)

    if len(X_all) < 10:
        raise ValueError("Not enough data")

    idx_train, idx_val = train_test_split(np.arange(len(X_all)), test_size=TEST_SIZE, 
                                            random_state=RANDOM_STATE, stratify=y_all)
    X_train, y_train = X_all[idx_train], y_all[idx_train]
    X_val, y_val = X_all[idx_val], y_all[idx_val]
    y_train_oh = to_categorical(y_train, NUM_CLASSES)
    y_val_oh = to_categorical(y_val, NUM_CLASSES)

    A_fixed = build_fully_connected_adjacency(NUM_CHANNELS)
    A_train = np.tile(A_fixed, (len(X_train), 1, 1))
    A_val = np.tile(A_fixed, (len(X_val), 1, 1))

    model = build_cnn_gnn_model()

    # Prepare callbacks: early stopping and model checkpoint (to save best model)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_filepath = os.path.join(MODEL_SAVE_DIR, 
        f"{subject_name}_{subfolder}_{timestamp}_checkpoint.h5")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max',
        save_format="h5"  # Explicitly set the save format to HDF5
    )
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                  patience=15, restore_best_weights=True)

    model.fit([X_train, A_train], y_train_oh, 
              validation_data=([X_val, A_val], y_val_oh),
              epochs=EPOCHS, batch_size=BATCH_SIZE, 
              callbacks=[early_stop, checkpoint],
              verbose=1)

    val_loss, val_acc = model.evaluate([X_val, A_val], y_val_oh, verbose=0)
    y_pred = np.argmax(model.predict([X_val, A_val], verbose=0), axis=1)

    cm = confusion_matrix(y_val, y_pred)
    cr = classification_report(y_val, y_pred).replace('\n', '\\n')

    attn = model.my_gat_layers[0].attn_coef
    importance = np.sum(np.mean(attn, axis=(0, 1)), axis=0) if attn is not None else [None] * NUM_CHANNELS

    # Save final model with best weights restored
    model_save_path = os.path.join(MODEL_SAVE_DIR, f"{subject_name}_{subfolder}_{timestamp}.h5")
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

    results = {
        "subject": subject_name,
        "subfolder": subfolder,
        "csv_file": csv_path,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "training_time_sec": time.time() - start_time,
        "timestamp": timestamp,
        "confusion_matrix": json.dumps(cm.tolist()),
        "classification_report": cr
    }
    for ch_name, imp_val in zip(CHANNEL_NAMES, importance):
        results[f"importance_{ch_name}"] = float(imp_val) if imp_val is not None else None

    return results

# ------------------------ Main ------------------------
def run_all_trainings(base_dir):
    results_list = []
    for subject_dir in glob.glob(os.path.join(base_dir, "Subject*")):
        subject_name = os.path.basename(subject_dir)
        for subfolder in ["cVEP1", "cVEP4"]:
            folder_path = os.path.join(subject_dir, subfolder)
            if os.path.isdir(folder_path):
                for csv_file in glob.glob(os.path.join(folder_path, "*.csv")):
                    try:
                        res = train_on_csv(csv_file, subject_name, subfolder)
                        results_list.append(res)
                    except Exception as e:
                        print(f"Error: {csv_file} -> {e}")

    df = pd.DataFrame(results_list)
    df.to_csv("all_results.csv", index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")
    print("\nAll results saved to: all_results.csv")

if __name__ == "__main__":
    run_all_trainings(BASE_DIR)
