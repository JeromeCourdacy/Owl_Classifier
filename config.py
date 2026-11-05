# config.py — réglages partagés par tout le pipeline (données, entraînement, firmware)
# Toute modification : PR approuvée par Noé ET Jérôme + incrémenter CONFIG_VERSION

from pathlib import Path

CONFIG_VERSION = 1

# --- Classes (l'ordre compte : c'est l'ordre des sorties du modèle) ---
CLASSES = ["boreal", "pygmy", "tawny", "other_owls", "noise"]
NOISE_CLASS = "noise"          # classe qui n'a droit qu'au bruit gaussien + SNR

# --- Audio ---
SR = 16000
WINDOW_SAMPLES = 32768         # 2,048 s — À DÉCIDER avec Jérôme (32000 = 2,0 s)

# --- Spectrogramme ---
N_FFT = 1024        #samples by window
HOP = 512       #step
N_MELS = 40
FMIN, FMAX = 0, 8000
NORM_MODE = "peak"
REF_FLOOR = 1e-2        #distinction between silence and not silence
NOISE_FLOOR_DB = -52.0

# --- Augmentation ---
TARGET_PER_CLASS = 400
PITCH_SEMITONES = 1.0          # ±1 demi-ton
STRETCH_RANGE = (0.9, 1.1)
SNR_RANGE_DB = (0, 15)
GAUSS_RANGE = (0.005, 0.015)
SEED = 42

# --- Chemins ---
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CLIPS_TRAIN = DATA / "data_train_2s"     
CLIPS_TEST  = DATA / "data_test_2s"
NOISE_BANK  = DATA / "noise_bank"
AUGMENTED   = DATA / "augmented_train"
PROCESSED   = DATA / "processed"