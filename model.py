"""
Owl TinyML pipeline - CNN training + Post-Training Quantization (adapted from Mathieu Bornet's cnn_int_ptq.py)

Loads the spectrograms (.npy) listed in manifest.csv of a data_vN folder made by the
data pipeline, trains the CNN (depthwise-separable blocks + Squeeze-and-Excitation),
converts it to an INT8 TFLite model (PTQ) and writes C arrays (.cc/.h) for the ESP32.
Train / validation / test come from split.csv (grouped by source recording).
All shared parameters come from config.py.
"""

import argparse
import csv
import datetime
import json
import time
from pathlib import Path


import numpy as np
import librosa 
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score, recall_score

import keras
import tensorflow as tf
from keras import layers, models, callbacks, regularizers
from ai_edge_litert.interpreter import Interpreter

import config

parser = argparse.ArgumentParser(description="Train the owl CNN on a data_vN folder and export INT8 TFLite.")
parser.add_argument("--data_dir", type=Path, default=config.PROCESSED / "data_v1",
                    help="data_vN folder containing manifest.csv and the .npy files")
parser.add_argument("--no_c_files", action="store_true", help="Skip generating .cc and .h files")
parser.add_argument("--metrics_out", type=str, default=None, help="Path to save JSON metrics")
parser.add_argument("--save_errors", action="store_true",
                    help="Save the misclassified test windows as .wav (needs the source recordings)")
args = parser.parse_args()

print(f"TensorFlow  : {tf.__version__}")
print(f"Keras       : {keras.__version__}")
print(f"numpy       : {np.__version__}")
print(f"GPU devices : {tf.config.list_physical_devices('GPU')}")

# Same seed = same weights initialisation and same shuffling every run
keras.utils.set_random_seed(config.SEED)

# =============================================================================
#                          PARAMETER ZONE
# =============================================================================

DATA_DIR = args.data_dir
MANIFEST = DATA_DIR / "manifest.csv"

# Create a unique output directory with timestamp
timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = config.ROOT / "outputs" / f"model_{timestamp}"

if not args.metrics_out:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved in: {OUTPUT_DIR}\n")

MODEL_FLOAT32_KERAS  = OUTPUT_DIR / "cnn_float32.keras"
MODEL_TFLITE_FLOAT32 = OUTPUT_DIR / "cnn_float32.tflite"
MODEL_TFLITE_INT8    = OUTPUT_DIR / "cnn_int8_ptq.tflite"
PLOT_PATH            = OUTPUT_DIR / "training_curves.png"
REPORT_PATH          = OUTPUT_DIR / "report.txt"
CSV_LOG_PATH         = OUTPUT_DIR / "test_predictions_log.csv"

# --- CLASSES (order = order of the model outputs, must match the firmware) ---
CLASSES        = config.CLASSES                      # ["pygmy", "boreal", "negative"]
N_CLASSES      = len(CLASSES)
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}

LEARNING_RATE = 0.001


# =============================================================================

class Timer:
    def __init__(self):
        self.records = {}
        self._starts = {}

    def start(self, name):
        self._starts[name] = time.time()
        print(f"\n⏱  [{name}] started...")

    def stop(self, name):
        elapsed = time.time() - self._starts[name]
        self.records[name] = elapsed
        m, s = divmod(int(elapsed), 60)
        print(f"✅  [{name}] done in {m}m {s:02d}s")
        return elapsed

    def report(self):
        lines = []
        lines.append("=" * 45)
        lines.append("         TIMING REPORT (Local Machine)")
        lines.append("=" * 45)
        total = 0
        for name, elapsed in self.records.items():
            m, s = divmod(int(elapsed), 60)
            lines.append(f"  {name:<30s} : {m}m {s:02d}s")
            total += elapsed
        lines.append("-" * 45)
        m, s = divmod(int(total), 60)
        lines.append(f"  {'TOTAL':<30s} : {m}m {s:02d}s")
        lines.append("=" * 45)
        return "\n".join(lines)

timer = Timer()

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def read_manifest(manifest_path):
    """Read manifest.csv: one row per .npy (file, split, class, subclass, recording, ...)."""
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_split(rows, split):
    """Load all spectrograms of one split.

    Returns X (n, n_mels, n_frames) float32, y (n,) class indices, and the
    manifest rows of these spectrograms (same order, for later analysis).
    """
    split_rows = [r for r in rows if r["split"] == split]
    X = np.stack([np.load(DATA_DIR / r["file"]) for r in split_rows]).astype(np.float32)
    y = np.array([CLASS_TO_INDEX[r["class"]] for r in split_rows], dtype=np.int32)
    return X, y, split_rows


def standardize(X, mu, std):
    """Standardise with the TRAIN mean and std (the same values go to the firmware)."""
    if config.NORM_MODE == "instance":
        m = X.mean(axis=(1, 2), keepdims=True)
        s = X.std(axis=(1, 2), keepdims=True) + 1e-8
        return (X - m) / s
    return (X - mu) / std


timer.start("Data loading & Norm")

rows = read_manifest(MANIFEST)
X_train, y_train, train_rows = load_split(rows, "train")
X_val,   y_val,   val_rows   = load_split(rows, "validation")
X_test,  y_test,  test_rows  = load_split(rows, "test")

n_mels, T = X_train.shape[1], X_train.shape[2]
for name, X in (("validation", X_val), ("test", X_test)):
    assert X.shape[1:] == (n_mels, T), f"{name} spectrograms are {X.shape[1:]}, train are {(n_mels, T)}"
print(f"Spectrogram shape: {n_mels} x {T}")
for name, y in (("train", y_train), ("validation", y_val), ("test", y_test)):
    print(f"  {name:10s}: {len(y):6d} | " + ", ".join(f"{c}={np.sum(y == i)}" for i, c in enumerate(CLASSES)))

mu  = float(X_train.mean())
std = float(X_train.std()) + 1e-8
X_train = standardize(X_train, mu, std)[..., np.newaxis]
X_val   = standardize(X_val,   mu, std)[..., np.newaxis]
X_test  = standardize(X_test,  mu, std)[..., np.newaxis]

y_train_oh = keras.utils.to_categorical(y_train, N_CLASSES)
y_val_oh   = keras.utils.to_categorical(y_val,   N_CLASSES)
y_test_oh  = keras.utils.to_categorical(y_test,  N_CLASSES)


timer.stop("Data loading & Norm")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

def se_block(inputs, ratio=8):
    """Squeeze-and-Excitation: learn one weight per channel and rescale the channels."""
    channels = inputs.shape[-1]
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Reshape((1, 1, channels))(se)
    se = layers.Dense(channels // ratio, activation="relu", use_bias=False)(se)
    se = layers.Dense(channels, activation="sigmoid", use_bias=False)(se)
    return layers.Multiply()([inputs, se])


def ds_block(x, filters, pool=True):
    """Depthwise-separable block: 3x3 depthwise conv + 1x1 pointwise conv (+ optional 2x2 max pooling)."""
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same",
                               depthwise_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, kernel_size=(1, 1), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if pool:
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x


def build_model(input_shape, n_classes):
    inp = layers.Input(shape=input_shape, name="mel_input")
    x = layers.BatchNormalization()(inp)

    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = ds_block(x, 64)
    x = ds_block(x, 128)
    x = ds_block(x, 256, pool=False)

    x = se_block(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    return models.Model(inp, out, name="Owl_CNN")


model = build_model((n_mels, T, 1), N_CLASSES)
model.summary()

opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

# ─────────────────────────────────────────────────────────────────────────────
# FLOAT32 TRAINING
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== Standard Training (Float32) ===")
timer.start("Training")

train_samples = len(X_train)

if train_samples < 400: BATCH_SIZE = 16
elif train_samples < 800: BATCH_SIZE = 32
else: BATCH_SIZE = 64

if train_samples < 500:
    MAX_EPOCHS = 150
    PATIENCE = 20
else:
    MAX_EPOCHS = 80
    PATIENCE = 10

# Class weights: each class counts the same in the loss, even if the train set is unbalanced
# (e.g. owls have augmented copies and negatives do not). Equal to 1.0 when classes are balanced.
counts = np.bincount(y_train, minlength=N_CLASSES)
CLASS_WEIGHT = {i: train_samples / (N_CLASSES * counts[i]) for i in range(N_CLASSES)}
print("Class weights:", {CLASSES[i]: round(w, 3) for i, w in CLASS_WEIGHT.items()})

cb_early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)

cb_reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,      # Wait 5 epochs without improvement before reducing LR
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train, y_train_oh,
    batch_size=BATCH_SIZE,
    epochs=MAX_EPOCHS,
    validation_data=(X_val, y_val_oh),
    class_weight=CLASS_WEIGHT,
    shuffle=True,
    verbose=1,
    callbacks=[cb_early_stop, cb_reduce_lr]
)
timer.stop("Training")

f32_loss, f32_acc = model.evaluate(X_test, y_test_oh, verbose=0)
y_pred_f32 = np.argmax(model.predict(X_test, verbose=0), axis=1)
print(f"Float32 test accuracy: {f32_acc * 100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE, TFLITE EXPORT & POST-TRAINING QUANTIZATION (PTQ)
# ─────────────────────────────────────────────────────────────────────────────

if not args.metrics_out:
    model.save(MODEL_FLOAT32_KERAS)

timer.start("TFLite Post-Training Quantization (PTQ)")

@tf.function(input_signature=[tf.TensorSpec([1, n_mels, T, 1], tf.float32, name="mel_input")])
def serving_fn(x):
    return model(x, training=False)

concrete_fn = serving_fn.get_concrete_function()


converter_f32 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], model)
tflite_f32 = converter_f32.convert()

if not args.metrics_out:
    with open(MODEL_TFLITE_FLOAT32, "wb") as f:
        f.write(tflite_f32)


converter_int8 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]

# Calibration set: 300 train spectrograms picked at random (all classes), not the first 300
rep_idx = np.random.default_rng(config.SEED).choice(len(X_train), size=min(300, len(X_train)), replace=False)

def representative_dataset():
    for i in rep_idx:
        yield [X_train[i:i+1].astype(np.float32)]

converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type  = tf.int8
converter_int8.inference_output_type = tf.int8

tflite_int8 = converter_int8.convert()

if not args.metrics_out:
    with open(MODEL_TFLITE_INT8, "wb") as f:
        f.write(tflite_int8)

print(f"TFLite sizes: float32 {len(tflite_f32) / 1024:.1f} KB | int8 {len(tflite_int8) / 1024:.1f} KB")
timer.stop("TFLite Post-Training Quantization (PTQ)")


def make_interpreter(tflite_model_bytes):
    interp = Interpreter(model_content=tflite_model_bytes)
    interp.allocate_tensors()
    return interp


def evaluate_tflite(tflite_model_bytes, X, y):
    """Run a .tflite model (float32 or int8) on X one spectrogram at a time.

    Returns (accuracy, list of predicted class indices).
    """
    if len(X) == 0:
        return 0.0, []
    interp = make_interpreter(tflite_model_bytes)
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale, in_zero = inp_d["quantization"]
    out_scale, out_zero = out_d["quantization"]

    y_preds = []
    for i in range(len(X)):
        img = X[i:i+1].astype(np.float32)
        if inp_d["dtype"] == np.int8:                      # quantize the input like the ESP32 will
            img = np.clip(np.round(img / in_scale + in_zero), -128, 127).astype(np.int8)
        interp.set_tensor(inp_d["index"], img)
        interp.invoke()
        out = interp.get_tensor(out_d["index"])
        if out_d["dtype"] == np.int8:                      # back to probabilities
            out = (out.astype(np.float32) - out_zero) * out_scale
        y_preds.append(int(np.argmax(out)))

    accuracy = float(np.mean(np.array(y_preds) == y))
    return accuracy, y_preds


timer.start("Evaluating TFLite models")
tflite_f32_acc, _ = evaluate_tflite(tflite_f32, X_test, y_test)
ptq_acc, y_pred_ptq = evaluate_tflite(tflite_int8, X_test, y_test)
print(f"Test accuracy: Keras float32 {f32_acc * 100:.2f}% | TFLite float32 {tflite_f32_acc * 100:.2f}% | TFLite int8 {ptq_acc * 100:.2f}%")

timer.stop("Evaluating TFLite models")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS EVALUATION AND TEXT FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def accuracy_by_subclass(rows, y_true, y_pred):
    """Accuracy for each (class, subclass) of the test set, worst subclass first inside each class."""
    groups = {}
    for r, t, p in zip(rows, y_true, y_pred):
        key = (r["class"], r["subclass"] or "-")
        groups.setdefault(key, []).append(int(t == p))
    lines = [f"  {'class':<10s} {'subclass':<35s} {'n':>6s} {'errors':>7s} {'accuracy':>9s}"]
    for (cls, sub), ok in sorted(groups.items(), key=lambda kv: (CLASSES.index(kv[0][0]), np.mean(kv[1]))):
        n, n_ok = len(ok), sum(ok)
        lines.append(f"  {cls:<10s} {sub[:35]:<35s} {n:6d} {n - n_ok:7d} {100 * n_ok / n:8.2f}%")
    return "\n".join(lines)


def confusion_text(y_true, y_pred):
    """Confusion matrix (counts) as text, for the report."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    header = " " * 12 + "".join(f"{c:>10s}" for c in CLASSES)
    lines = [f"  {CLASSES[i]:<10s}" + "".join(f"{v:10d}" for v in cm[i]) for i in range(N_CLASSES)]
    return "  rows = true class, columns = predicted class\n" + header + "\n" + "\n".join(lines)


def report_block(title, y_pred, acc):
    """Full text block for one model: per-class report, balanced accuracy, confusion matrix, per subclass."""
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    text  = f"--- {title} ---\n"
    text += classification_report(y_test, y_pred, labels=list(range(N_CLASSES)),
                                   target_names=CLASSES, digits=4, zero_division=0) + "\n"
    text += f"Accuracy          : {acc * 100:.2f}%\n"
    text += f"Balanced accuracy : {bal_acc * 100:.2f}%\n\n"
    text += "Confusion matrix:\n" + confusion_text(y_test, y_pred) + "\n\n"
    text += "Accuracy per subclass:\n" + accuracy_by_subclass(test_rows, y_test, y_pred) + "\n\n"
    return text, bal_acc


results_text  = "\n" + "=" * 70 + "\n"
results_text += "=== TEST RESULTS ===\n"
results_text += "=" * 70 + "\n"
results_text += f"Data          : {DATA_DIR}\n"
results_text += f"Spectrogram   : {n_mels} x {T}\n"
results_text += f"Test set      : {len(y_test)} spectrograms | " + ", ".join(f"{c}={np.sum(y_test == i)}" for i, c in enumerate(CLASSES)) + "\n"
results_text += f"TFLite float32: {tflite_f32_acc * 100:.2f}% (must be ~ equal to Keras float32)\n\n"

text_f32, bal_acc_f32 = report_block("Keras float32", y_pred_f32, f32_acc)
text_ptq, bal_acc_ptq = report_block("TFLite INT8 (PTQ) - model for the ESP32", y_pred_ptq, ptq_acc)
results_text += text_f32 + text_ptq

print(results_text)

# ─────────────────────────────────────────────────────────────────────────────
# CSV LOGGING & MISCLASSIFIED AUDIO CLIPS
# ─────────────────────────────────────────────────────────────────────────────

timer.start("Error Logging & Audio Clips")

MISCLASSIFIED_DIR = OUTPUT_DIR / "misclassified"

if not args.metrics_out:
    # --- One row per test spectrogram: where it comes from + what both models predicted ---
    with open(CSV_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "class", "subclass", "recording", "start_s",
                         "pred_float32", "pred_int8", "correct_int8"])
        for r, p32, p8 in zip(test_rows, y_pred_f32, y_pred_ptq):
            writer.writerow([r["file"], r["class"], r["subclass"], r["recording"], r["start_s"],
                             CLASSES[p32], CLASSES[p8], int(p8 == CLASS_TO_INDEX[r["class"]])])
    n_errors = int(np.sum(np.array(y_pred_ptq) != y_test))
    print(f"\nPredictions log (CSV) saved at: {CSV_LOG_PATH} ({n_errors} INT8 errors)")

    # --- Optional: cut the misclassified 2 s windows from the source recordings, to listen to them ---
    if args.save_errors and n_errors > 0:
        saved, missing = 0, 0
        for r, p8 in zip(test_rows, y_pred_ptq):
            if p8 == CLASS_TO_INDEX[r["class"]]:
                continue
            src = config.ROOT / r["recording"]
            if not src.exists():
                missing += 1
                continue
            clip, _ = librosa.load(src, sr=config.SR, mono=True,
                                   offset=float(r["start_s"]), duration=config.WINDOW_SAMPLES / config.SR)
            clip = np.pad(clip, (0, max(0, config.WINDOW_SAMPLES - len(clip))))
            out_dir = MISCLASSIFIED_DIR / CLASSES[p8]                      # folder = predicted class
            out_dir.mkdir(parents=True, exist_ok=True)
            sub = "".join(c if c.isalnum() else "_" for c in (r["subclass"] or "-"))[:30]
            sf.write(out_dir / f"{r['class']}_{sub}_{Path(r['file']).stem}.wav", clip, config.SR)
            saved += 1
        print(f"Misclassified clips: {saved} saved under {MISCLASSIFIED_DIR}"
              + (f" | {missing} skipped (source recording not found)" if missing else ""))

timer.stop("Error Logging & Audio Clips")

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION & PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
timer.start("Plotting")

if not args.metrics_out:
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)
    best_epoch = int(np.argmin(h["val_loss"])) + 1          # weights restored by EarlyStopping

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Owl CNN - training diagnostics & PTQ (INT8)", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

    # --- Row 1: training curves ---
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, h["accuracy"],     color="#2196F3", lw=2, label="Train")
    ax.plot(epochs, h["val_accuracy"], color="#FF9800", lw=2, ls="--", label="Val")
    ax.axhline(f32_acc, color="#4CAF50", ls=":", lw=1.5, label=f"Test {f32_acc * 100:.1f}%")
    ax.axvline(best_epoch, color="grey", ls=":", lw=1, label=f"Best epoch {best_epoch}")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim([0, 1.05])

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, h["loss"],     color="#2196F3", lw=2, label="Train")
    ax.plot(epochs, h["val_loss"], color="#FF9800", lw=2, ls="--", label="Val")
    ax.axvline(best_epoch, color="grey", ls=":", lw=1, label=f"Best epoch {best_epoch}")
    ax.set_title("Loss"); ax.set_xlabel("Epoch")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    gap = np.array(h["accuracy"]) - np.array(h["val_accuracy"])
    ax.fill_between(epochs, gap, 0, where=(gap > 0),  alpha=0.4, color="#F44336", label="Train > Val (overfit)")
    ax.fill_between(epochs, gap, 0, where=(gap <= 0), alpha=0.4, color="#4CAF50", label="Val >= Train")
    ax.plot(epochs, gap, color="#333", lw=1.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Train - Val accuracy gap"); ax.set_xlabel("Epoch")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Row 2: learning rate, accuracy comparison, recall per class ---
    ax = fig.add_subplot(gs[1, 0])
    if "learning_rate" in h:
        ax.plot(epochs, h["learning_rate"], color="#607D8B", lw=2)
        ax.set_yscale("log")
    ax.set_title("Learning rate (ReduceLROnPlateau)"); ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    names = ["Keras\nfloat32", "TFLite\nfloat32", "TFLite\nINT8 (PTQ)"]
    accs  = [f32_acc * 100, tflite_f32_acc * 100, ptq_acc * 100]
    ax.bar(names, accs, color=["#2196F3", "#03A9F4", "#9C27B0"], width=0.5)
    ax.set_ylim([max(0, min(accs) - 5), 100])
    ax.set_ylabel("Test accuracy (%)"); ax.set_title("Accuracy comparison (y axis does not start at 0)")
    for i, v in enumerate(accs):
        ax.text(i, v + 0.3, f"{v:.2f}%", ha="center", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[1, 2])
    rec_f32 = recall_score(y_test, y_pred_f32, labels=list(range(N_CLASSES)), average=None, zero_division=0)
    rec_ptq = recall_score(y_test, y_pred_ptq, labels=list(range(N_CLASSES)), average=None, zero_division=0)
    xpos = np.arange(N_CLASSES)
    ax.bar(xpos - 0.2, rec_f32 * 100, width=0.4, color="#2196F3", label="Keras float32")
    ax.bar(xpos + 0.2, rec_ptq * 100, width=0.4, color="#9C27B0", label="TFLite INT8")
    ax.set_xticks(xpos, CLASSES)
    ax.set_ylim([max(0, min(rec_f32.min(), rec_ptq.min()) * 100 - 5), 100])
    ax.set_ylabel("Recall (%)"); ax.set_title("Recall per class (found / total)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # --- Row 3: confusion matrices + worst subclasses (INT8) ---
    for col, (y_p, title) in enumerate([(y_pred_f32, "Keras float32"), (y_pred_ptq, "TFLite INT8 (PTQ)")]):
        ax = fig.add_subplot(gs[2, col])
        cm = confusion_matrix(y_test, y_p, labels=list(range(N_CLASSES)))
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                    xticklabels=CLASSES, yticklabels=CLASSES,
                    ax=ax, linewidths=0.5, linecolor="grey")
        ax.set_title(f"Confusion matrix - {title}")
        ax.set_ylabel("True"); ax.set_xlabel("Predicted")

    ax = fig.add_subplot(gs[2, 2])
    groups = {}
    for r, t, p in zip(test_rows, y_test, y_pred_ptq):
        groups.setdefault(f"{r['class']} / {(r['subclass'] or '-')[:20]}", []).append(int(t == p))
    worst = sorted(groups.items(), key=lambda kv: np.mean(kv[1]))[:10]
    labels = [f"{k} (n={len(v)})" for k, v in worst]
    values = [np.mean(v) * 100 for k, v in worst]
    ax.barh(labels[::-1], values[::-1], color="#F44336")
    ax.set_xlim([0, 100])
    ax.set_xlabel("INT8 accuracy (%)"); ax.set_title("10 worst subclasses (INT8)")
    ax.tick_params(axis="y", labelsize=7); ax.grid(True, alpha=0.3, axis="x")

    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved -> {PLOT_PATH}")

timer.stop("Plotting")

# ─────────────────────────────────────────────────────────────────────────────
# C ARRAY GENERATION, REPORT & METRICS
# ─────────────────────────────────────────────────────────────────────────────

# Quantization parameters of the INT8 model input/output (the firmware needs them)
_q = make_interpreter(tflite_int8)
in_scale,  in_zero  = _q.get_input_details()[0]["quantization"]
out_scale, out_zero = _q.get_output_details()[0]["quantization"]

if not args.no_c_files and not args.metrics_out:
    timer.start("C Array Generation")

    # --- model_data.cc: the INT8 model as a byte array, compiled into the firmware ---
    hex_lines = []
    for i in range(0, len(tflite_int8), 12):
        hex_lines.append("  " + ", ".join(f"0x{b:02x}" for b in tflite_int8[i:i + 12]))
    with open(OUTPUT_DIR / "model_data.cc", "w") as f:
        f.write('#include "model_data.h"\n\n')
        f.write("alignas(16) const unsigned char cnn_int8_tflite[] = {\n")
        f.write(",\n".join(hex_lines))
        f.write(f"\n}};\nconst unsigned int cnn_int8_tflite_len = {len(tflite_int8)};\n")

    # --- model_data.h: declarations + every constant the firmware front-end must match ---
    class_names = ", ".join(f'"{c}"' for c in CLASSES)
    with open(OUTPUT_DIR / "model_data.h", "w") as f:
        f.write(f"// Generated by model.py on {timestamp} from {DATA_DIR.name} - do not edit by hand\n")
        f.write("#pragma once\n\n")
        f.write("extern const unsigned char cnn_int8_tflite[];\n")
        f.write("extern const unsigned int cnn_int8_tflite_len;\n\n")
        f.write("// Audio front-end (must be identical to spectrogram.py / config.py)\n")
        f.write(f"constexpr int   SAMPLE_RATE    = {config.SR};\n")
        f.write(f"constexpr int   WINDOW_SAMPLES = {config.WINDOW_SAMPLES};\n")
        f.write(f"constexpr int   N_FFT          = {config.N_FFT};\n")
        f.write(f"constexpr int   HOP_LENGTH     = {config.HOP};\n")
        f.write(f"constexpr int   N_MELS         = {n_mels};\n")
        f.write(f"constexpr int   N_FRAMES       = {T};\n")
        f.write(f"constexpr float FMIN_HZ        = {float(config.FMIN):.1f}f;\n")
        f.write(f"constexpr float FMAX_HZ        = {float(config.FMAX):.1f}f;\n")
        f.write(f"constexpr float DC_POLE        = {-config.DC_FILTER_A[1]:.6f}f;  // y[n] = x[n] - x[n-1] + DC_POLE * y[n-1]\n")
        f.write(f"constexpr float REF_FLOOR      = {config.REF_FLOOR:.6f}f;\n")
        f.write(f"constexpr float NOISE_FLOOR_DB = {config.NOISE_FLOOR_DB:.1f}f;\n\n")
        f.write("// Standardisation: x_std = (x_db - NORM_MU) / NORM_STD\n")
        f.write(f"constexpr float NORM_MU        = {mu:.6f}f;\n")
        f.write(f"constexpr float NORM_STD       = {std:.6f}f;\n\n")
        f.write("// INT8 quantization: q = round(x_std / INPUT_SCALE) + INPUT_ZERO_POINT ; p = (q - OUTPUT_ZERO_POINT) * OUTPUT_SCALE\n")
        f.write(f"constexpr float INPUT_SCALE        = {in_scale:.8f}f;\n")
        f.write(f"constexpr int   INPUT_ZERO_POINT   = {in_zero};\n")
        f.write(f"constexpr float OUTPUT_SCALE       = {out_scale:.8f}f;\n")
        f.write(f"constexpr int   OUTPUT_ZERO_POINT  = {out_zero};\n\n")
        f.write("// Model outputs, in this order\n")
        f.write(f"constexpr int   N_CLASSES = {N_CLASSES};\n")
        f.write(f"constexpr const char* CLASS_NAMES[N_CLASSES] = {{{class_names}}};\n")

    timer.stop("C Array Generation")
    print(f"C files saved -> {OUTPUT_DIR / 'model_data.cc'} and model_data.h")

if not args.metrics_out:
    # --- report.txt: timings, settings used, firmware constants, all test results ---
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(timer.report())
        f.write("\n\nVersions & device:\n")
        f.write(f"  TensorFlow {tf.__version__} | Keras {keras.__version__} | numpy {np.__version__}\n")
        f.write(f"  GPU : {tf.config.list_physical_devices('GPU')}\n")
        f.write("\nTraining:\n")
        f.write(f"  data           : {DATA_DIR}\n")
        f.write(f"  epochs run     : {len(history.history['loss'])} (best {int(np.argmin(history.history['val_loss'])) + 1})\n")
        f.write(f"  batch size     : {BATCH_SIZE} | initial LR {LEARNING_RATE} | seed {config.SEED}\n")
        f.write(f"  class weights  : {', '.join(f'{CLASSES[i]}={w:.3f}' for i, w in CLASS_WEIGHT.items())}\n")
        f.write(f"  parameters     : {model.count_params():,}\n")
        f.write(f"  TFLite sizes   : float32 {len(tflite_f32) / 1024:.1f} KB | int8 {len(tflite_int8) / 1024:.1f} KB\n")
        f.write("\nFirmware constants (also in model_data.h):\n")
        f.write(f"  NORM_MU  = {mu:.6f}   NORM_STD = {std:.6f}\n")
        f.write(f"  INPUT  scale = {in_scale:.8f}  zero_point = {in_zero}\n")
        f.write(f"  OUTPUT scale = {out_scale:.8f}  zero_point = {out_zero}\n")
        f.write(results_text)
    print(f"Report saved -> {REPORT_PATH}")
    print("\nDone! Model trained, quantized (PTQ INT8) and exported.")

if args.metrics_out:
    # --- JSON metrics for ablation studies (no files written in OUTPUT_DIR) ---
    recall_ptq = recall_score(y_test, y_pred_ptq, labels=list(range(N_CLASSES)), average=None, zero_division=0)
    metrics = {
        "accuracy_f32":          float(f32_acc),
        "accuracy_tflite_f32":   float(tflite_f32_acc),
        "accuracy_ptq":          float(ptq_acc),
        "balanced_accuracy_f32": float(bal_acc_f32),
        "balanced_accuracy_ptq": float(bal_acc_ptq),
        "f1_macro_ptq":          float(f1_score(y_test, y_pred_ptq, average="macro")),
        "recall_ptq":            {c: float(r) for c, r in zip(CLASSES, recall_ptq)},
        "loss_f32":              float(f32_loss),
        "confusion_matrix_ptq":  confusion_matrix(y_test, y_pred_ptq, labels=list(range(N_CLASSES))).tolist(),
        "classes":               CLASSES,
        "model_size_int8_kb":    len(tflite_int8) / 1024,
    }
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {args.metrics_out}")