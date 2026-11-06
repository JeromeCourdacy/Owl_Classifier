# Project context: Edge TinyML owl song detection (GenoRobotics, Jura forest)

> **How to use this file:** paste it into a new chat as context. You (the assistant) are helping a student on the team. The reader has access to Mathieu Bornet's original repo and to Noé's repo `Boreal_Owl_Classifier`. Answer in English unless asked otherwise. When unsure about a detail, ask to see the relevant file rather than guessing.

---

## 1. Goal

- Sensor **nodes fixed on trees** in a forest of the **Jura (Switzerland)**.
- Each node runs a small neural network **on the microcontroller** that detects owl songs in real time, then sends **species + precise timestamp** over **LoRa**.
- Other team members use the timestamps from several nodes to **triangulate** the bird's position (not part of the TinyML work).
- Hardware: **LilyGO T-Beam S3** (ESP32-S3 + LoRa + GPS). Microphone choice still open.
- **Target species:**
  - **Boreal / Tengmalm's Owl** (*Aegolius funereus*)
  - **Eurasian Pygmy Owl** (*Glaucidium passerinum*)
- Decision taken: **build our own model** by adapting Mathieu Bornet's pipeline (TensorFlow → TFLite Micro). We are **not** using Edge Impulse.

## 2. Starting point: Mathieu Bornet's semester project

Mathieu's repo ("Edge AI noise detection", PDF report included) detects sounds (e.g. chainsaws) on an ESP32 with a CNN. We reuse his processing chain and firmware ideas.

**Audio front-end (must be identical in Python training and in C++ on device):**
- 16 kHz mono, window **2.048 s = 32768 samples**
- DC-block / pre-emphasis: `lfilter([1, -1], [1, -0.995])`
- STFT: `n_fft=1024`, Hann, `hop=512` → **63 frames**
- **40 Mel bands** (Slaney), dB with a floored peak reference (`REF_FLOOR = 1e-2`), floor at −52 dB
- Global normalization: μ = −34.394428, σ = 12.728
- Model input: **40 × 63** spectrogram

**Model (DS-CNN):** Conv 32 → depthwise-separable blocks 64 / 128 / 256 → Squeeze-and-Excitation → Global Average Pooling → Dense 128 → Dropout 0.4 → softmax. Adam lr 1e-3, early stopping, ReduceLROnPlateau.

**Deployment:** post-training quantization to **INT8 TFLite**, exported as a C array, run with **TFLite Micro**.

**Firmware (`esp32/esp32_mainTest/src/main.cpp`):** ESP-IDF / PlatformIO, I2S PDM mic (pins 41/42 on a XIAO ESP32-S3), spectrogram computed in C++ with **ESP-DSP** FFT, tensor arena ~120 KB. Capture and inference run one after the other, so ~20 % of the time is not listened to. There is also a USB streaming test firmware and an `esp32_aug` firmware used for on-device re-recording augmentation.

**Scripts (numbered pipeline):** `0.*` data state/folders, `1.*` extractors (chainsaw, UrbanSound, background noise bank…), `2.TrainTestSeparation`, `3.TimeWindowChunker`, `4.NoiseAugmenting`, `5.MelSpectrogramPreProcess`, `7.*` ablation studies / robustness, `8.*` ESP tests, plotting, debounce study, `9.DataAugmentation` / `9.AugSlicer` (re-recording through the real mic, ~+20 % on device), `10.TestRelabeler`, `u.*` utilities (spectral analysis, comparer, ytb→wav), `esp32/cnn_int_ptq.py`, `cnn_int_qat.py`.

**Known weaknesses to fix in our version:**
- Validation split is random per window → windows of the same recording end up in train and validation (**leakage**). We will use a **split grouped by recording**.
- Class names are hard-coded in ~20 files → we plan a single **`config.py`** (classes + all shared parameters).

## 3. Our repo: `Boreal_Owl_Classifier`

GitHub: `JeromeCourdacy/Boreal_Owl_Classifier`. History was rewritten on 25.10.2026 (single commit) to remove an API key. **If you cloned before that date, delete your clone and clone again.**

```
Boreal_Owl_Classifier/
├── .env                 # XC_KEY=<xeno-canto key>  (NOT in git; each person makes their own)
├── .gitignore           # .env, .DS_Store, _backup/, __pycache__/
├── xc_download.py       # download Xeno-canto recordings
├── extract_calls.py     # cut 2 s clips around the songs
├── data/
│   ├── <species>_full/<species>_{train,test}_full/   # full recordings (wav 16 kHz mono)
│   ├── data_train_2s/<species>_train/               # 2 s clips
│   ├── data_test_2s/<species>_test/
│   └── metadata/<folder>.csv                        # recordist, country, licence, url... (cite in report)
└── _backup/             # local only, ignored
```

### `xc_download.py`
Uses the Xeno-canto API v3. Key read from `.env` (`XC_KEY=...`), never hard-coded. Converts to WAV mono 16 kHz with ffmpeg.
Defaults: type exactly `song`, quality A or B, ≤ 4 min, ≤ 3 recordings per recordist.

```bash
python xc_download.py "Aegolius funereus" -n 30 -o data/boreal_full/boreal_train_full
python xc_download.py "Aegolius funereus" -n 10 -o data/boreal_full/boreal_test_full \
       --exclude data/boreal_full/boreal_train_full      # no shared recording OR recordist
# options: -q A,B  --type song|call|any  --include-mixed  --max-per-recordist N
#          --max-duration S  --min-duration S  --allow-same-recordists
```

### `extract_calls.py`
Stdlib + ffmpeg only. Measures energy every 50 ms in the species' frequency band; a frame counts as "song" if it is > threshold dB above the background **and** most of its energy is inside the band; keeps the most energetic non-overlapping windows.

```bash
python extract_calls.py IN_FOLDER OUT_FOLDER --band 550 1100 --max 3
# options: --duration 2  --threshold 10  --min-band-ratio 0.4  --min-active 0.1  --min-gap
```

**Measured song bands (use with `--band`):**

| Species | Band (Hz) | Note |
|---|---|---|
| Boreal Owl | 550–1100 | fundamental ~650–950 |
| Tawny Owl | 550–1100 | same band as Boreal, differs by rhythm |
| Eurasian Pygmy Owl | 1150–2100 | ~1.3–2.0 kHz, scale song rises |
| Long-eared Owl, Eagle Owl | 230–650 | |
| Barn Owl | 1200–4000 | use `--min-band-ratio 0.3` |

## 4. Dataset (state on 29.10.2026)

| Class | Recordings train / test | 2 s clips train / test |
|---|---|---|
| boreal | 30 / 10 | 90 / 30 |
| pygmy | 20 / 10 | 60 / 25 |
| tawny | 30 / 10 | 88 / 29 |
| other_owls: long-eared | 12 / 4 | 32 / 9 |
| other_owls: eagle | 10 / 3 | 30 / 9 |
| other_owls: barn | 15 / 6 | 26 / 12 |
| noise (ESC-50 nature sounds) | 44 / 17 | 86 / 33 |

Verified: **no recording, no recordist and no clip shared between train and test.** Sources: Xeno-canto (owls), ESC-50 (noise). Clips are 2.0 s.

## 5. Classes and reasoning

Planned classes: **boreal, pygmy** (targets), **tawny**, **other_owls**, **noise**, and maybe **other_animals** (to decide).

- A softmax always picks one class, so we need "rejection" classes or every sound would be labelled as a target.
- **Tawny** gets its own class: it is the most common owl in the Jura and sings in the same band as Boreal (a hard negative). A separate class forces the model to learn the difference (rhythm) and makes errors visible in the confusion matrix.
- Other owls are grouped: they use different bands and we have little data for each.
- On the device only boreal/pygmy detections are sent; other classes are ignored.
- To check: ablation **6 classes vs 3 classes** (boreal / pygmy / other), comparing boreal and pygmy precision/recall on the same test set.
- On device: **confidence threshold** (e.g. p > 0.8) plus a **debounce** (several consecutive windows agree) to cut false alarms over LoRa.

## 6. Open decisions / next steps

1. Window length: our clips are **2.0 s**, Mathieu's firmware uses **2.048 s**; 3 s is also an option (Boreal song is a series of notes). Must be the same in training and firmware.
2. 40 vs 64 Mel bands and fmin/fmax (targets are between ~550 and 2100 Hz).
3. Create `config.py` and adapt Mathieu's scripts 3 → 5 → training to our classes.
4. Grouped train/validation split by recording.
5. **Baseline training** with Mathieu's settings, then iterate (augmentation, class weights, hard-negative mining).
6. Port firmware to the **T-Beam S3** (pins, mic, LoRa message: species + GPS-synced timestamp), choose the microphone.
7. On-device re-recording augmentation (Mathieu's script 9) once the mic is chosen.

## 7. Practical notes

- Environment: Python 3.11 conda env recommended (TensorFlow does not support the newest Python versions), plus `ffmpeg`.
- The spectrogram is computed **outside** the model: librosa in Python for training, ESP-DSP in C++ on device. Both must give the same numbers, or the model will fail on the device.
- Never commit `.env` or API keys.
- Code and docs are written in English.
