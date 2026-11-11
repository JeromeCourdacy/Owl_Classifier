# config.py — réglages partagés par tout le pipeline (données, entraînement, firmware)
# Toute modification : PR approuvée par Noé ET Jérôme + incrémenter CONFIG_VERSION

from pathlib import Path

CONFIG_VERSION = 1

# --- Classes (l'ordre compte : c'est l'ordre des sorties du modèle) ---
CLASSES = ["pygmy", "boreal", "negative"]
NOISE_CLASS = "negative"       # class that only gets noise mixing + Gaussian noise

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
DC_FILTER_B = (1.0, -1.0)      # DC-block filter, identical to the firmware
DC_FILTER_A = (1.0, -0.995)

# --- Augmentation ---
PITCH_SEMITONES = 1.0          # ±1 demi-ton
STRETCH_RANGE = (0.9, 1.1)
SNR_RANGE_DB = (0, 15)
GAUSS_SNR_RANGE_DB = (20, 40)   # Gaussian hiss level relative to the song (dB)
SEED = 42
NOISE_BANK_SUBCLASSES = ("risoux_background",)   # split.csv subclasses used as mixing noise
N_AUG_PER_WINDOW = {"pygmy": 1, "boreal": 1, "negative": 0}   # augmented copies per train window
SPECAUG_COPIES = 0   # SpecAugment copies per train spectrogram (0 = off)

# --- Chemins ---
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"  
PROCESSED   = DATA / "processed"
SPLIT_CSV   = ROOT / "split.csv"