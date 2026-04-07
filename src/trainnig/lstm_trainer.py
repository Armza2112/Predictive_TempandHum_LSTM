from __future__ import annotations

import argparse
import logging
import os
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # ซ่อน TF info logs

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS  —  แก้ได้ตรงนี้
# ══════════════════════════════════════════════════════════════════════════════

# ── Data ─────────────────────────────────────────────────────────────────────
FREQ_MIN       = 10          # ความถี่ข้อมูล sensor (นาที)
LOOKBACK_STEPS = 144         # lookback window  144 steps = 24h
                              # เพิ่มเป็น 288 (48h) ถ้ามีข้อมูลเยอะ
N_HOURS_AHEAD  = 24          # ทำนายล่วงหน้า 24 ชั่วโมง
STEPS_PER_HOUR = 6           # 6 × 10min = 1h

TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15        # test = 1 - 0.70 - 0.15 = 0.15

# ── Features per timestep ────────────────────────────────────────────────────
# temp, hum, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos
N_FEATURES     = 8

# ── Architecture ─────────────────────────────────────────────────────────────
LSTM_UNITS     = [128, 64]   # BiLSTM layers (จะเป็น 256 และ 128 จริงเพราะ Bidirectional)
DENSE_UNITS    = [128, 64]
DROPOUT        = 0.2
RECURRENT_DROPOUT = 0.1

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE     = 64
MAX_EPOCHS     = 500
PATIENCE_STOP  = 40          # early stopping patience
PATIENCE_LR    = 15          # ReduceLROnPlateau patience
LR_INIT        = 1e-3
LR_MIN         = 1e-6
LR_FACTOR      = 0.5         # ลด LR ครึ่งหนึ่งเมื่อไม่ improve

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH     = Path("data/processed/train.csv")
VAL_PATH       = Path("data/processed/val.csv")
TEST_PATH      = Path("data/processed/test.csv")
CLEAN_PATH     = Path("data/processed/sensor_data_clean.csv")
MODEL_DIR      = Path("model")


# ══════════════════════════════════════════════════════════════════════════════
#  1. Simple StandardScaler (ไม่พึ่ง sklearn)
# ══════════════════════════════════════════════════════════════════════════════
class NumpyScaler:
    """Standardize: (x - mean) / std  สำหรับ 2D array"""

    def fit(self, X: np.ndarray) -> "NumpyScaler":
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ══════════════════════════════════════════════════════════════════════════════
#  2. Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════
def add_time_features(df: pd.DataFrame) -> np.ndarray:
    """
    สร้าง feature matrix shape (N, N_FEATURES)
    คอลัมน์: temp, hum, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos
    """
    idx = df.index
    hour_rad   = 2 * np.pi * idx.hour / 24
    dow_rad    = 2 * np.pi * idx.dayofweek / 7
    month_rad  = 2 * np.pi * (idx.month - 1) / 12

    return np.column_stack([
        df["temp"].values,
        df["hum"].values,
        np.sin(hour_rad),
        np.cos(hour_rad),
        np.sin(dow_rad),
        np.cos(dow_rad),
        np.sin(month_rad),
        np.cos(month_rad),
    ])


FEATURE_NAMES = [
    "temp", "hum",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin","month_cos",
]


# ══════════════════════════════════════════════════════════════════════════════
#  3. Sliding Window Dataset
# ══════════════════════════════════════════════════════════════════════════════
def make_sequences(
    features_scaled: np.ndarray,
    target_col_idx: int,
    lookback: int       = LOOKBACK_STEPS,
    steps_per_hour: int = STEPS_PER_HOUR,
    n_hours: int        = N_HOURS_AHEAD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    สร้าง X (N, lookback, n_features)  และ  y (N, n_hours)

    y[i, h] = ค่า target ที่ (h+1)*steps_per_hour steps หลังจาก window
    = ค่าที่ 1h, 2h, ..., n_hours ข้างหน้า
    """
    X_list, y_list = [], []
    n = len(features_scaled)
    # ต้องการ window + n_hours*steps_per_hour ข้างหน้า
    max_horizon = n_hours * steps_per_hour

    for i in range(lookback, n - max_horizon + 1):
        window = features_scaled[i - lookback : i]          # (lookback, n_feat)
        # target: ค่าที่ h*steps_per_hour ข้างหน้า (h = 1..24)
        targets = np.array([
            features_scaled[i + (h * steps_per_hour) - 1, target_col_idx]
            for h in range(1, n_hours + 1)
        ])
        X_list.append(window)
        y_list.append(targets)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  4. Model Architecture
# ══════════════════════════════════════════════════════════════════════════════
def build_model(
    lookback: int   = LOOKBACK_STEPS,
    n_features: int = N_FEATURES,
    n_output: int   = N_HOURS_AHEAD,
) -> "tf.keras.Model":
    """
    Bidirectional LSTM → Dense  Multi-output model

              Input (lookback, n_features)
                      │
           ┌──────────▼──────────────┐
           │ BiLSTM(128, seq=True)   │  → 256 units
           │ LayerNorm + Dropout     │
           └──────────┬──────────────┘
                      │
           ┌──────────▼──────────────┐
           │ BiLSTM(64,  seq=False)  │  → 128 units
           │ LayerNorm + Dropout     │
           └──────────┬──────────────┘
                      │
           ┌──────────▼──────────────┐
           │ Dense(128, GELU)        │
           │ Dense(64,  GELU)        │
           └──────────┬──────────────┘
                      │
                Dense(24)  ← ผลลัพธ์ 24 ค่า
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inp = tf.keras.Input(shape=(lookback, n_features), name="input_seq")

    # ── BiLSTM layer 1 ────────────────────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(
            LSTM_UNITS[0],
            return_sequences=True,
            recurrent_dropout=RECURRENT_DROPOUT,
        ),
        name="bilstm_1",
    )(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(DROPOUT)(x)

    # ── BiLSTM layer 2 ────────────────────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(
            LSTM_UNITS[1],
            return_sequences=False,
            recurrent_dropout=RECURRENT_DROPOUT,
        ),
        name="bilstm_2",
    )(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(DROPOUT)(x)

    # ── Dense layers ──────────────────────────────────────────────────────────
    x = layers.Dense(DENSE_UNITS[0], activation="gelu", name="dense_1")(x)
    x = layers.Dropout(DROPOUT / 2)(x)
    x = layers.Dense(DENSE_UNITS[1], activation="gelu", name="dense_2")(x)

    # ── Output: 24 hourly predictions ─────────────────────────────────────────
    out = layers.Dense(n_output, name="output")(x)

    model = Model(inputs=inp, outputs=out)
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  5. Metrics
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    if label:
        log.info("    %-30s  RMSE=%.4f  MAE=%.4f  MAPE=%.2f%%  R²=%.4f",
                 label, rmse, mae, mape, r2)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


# ══════════════════════════════════════════════════════════════════════════════
#  6. Plotting
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_history(history, target: str, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{target} — Training Loss (MSE)")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history.history["mae"],     label="Train MAE")
    axes[1].plot(history.history["val_mae"], label="Val MAE")
    axes[1].set_title(f"{target} — MAE")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    log.info("    Plot saved → %s", save_path)


def plot_predictions_per_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    save_path: Path,
) -> None:
    """วาด RMSE ต่อ horizon (h=1..24h) เพื่อดูว่า accuracy ลดลงอย่างไร"""
    rmse_per_h = [
        np.sqrt(np.mean((y_true[:, h] - y_pred[:, h]) ** 2))
        for h in range(N_HOURS_AHEAD)
    ]
    hours = list(range(1, N_HOURS_AHEAD + 1))

    plt.figure(figsize=(10, 4))
    plt.plot(hours, rmse_per_h, marker="o", linewidth=2)
    plt.fill_between(hours, rmse_per_h, alpha=0.2)
    plt.xlabel("Horizon (hours ahead)")
    plt.ylabel("RMSE")
    plt.title(f"{target} — RMSE per Horizon (Test Set)")
    plt.xticks(hours)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    log.info("    Plot saved → %s", save_path)


# ══════════════════════════════════════════════════════════════════════════════
#  7. Save / Load Bundle
# ══════════════════════════════════════════════════════════════════════════════
def save_lstm_bundle(
    keras_model,
    scaler_X: NumpyScaler,
    scaler_y: NumpyScaler,
    target: str,
    model_dir: Path,
) -> None:
    """บันทึก Keras model (.keras) + metadata (.pkl)"""
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"lstm_{target}.keras"
    meta_path  = model_dir / f"lstm_{target}_meta.pkl"

    keras_model.save(str(model_path))

    meta = {
        "scaler_X":     scaler_X,
        "scaler_y":     scaler_y,
        "target":       target,
        "lookback":     LOOKBACK_STEPS,
        "n_features":   N_FEATURES,
        "n_hours":      N_HOURS_AHEAD,
        "steps_per_hour": STEPS_PER_HOUR,
        "feature_names": FEATURE_NAMES,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    log.info("    Model   → %s", model_path)
    log.info("    Meta    → %s", meta_path)


def load_lstm_bundle(target: str, model_dir: Path = MODEL_DIR) -> dict:
    """โหลด LSTM bundle สำหรับ inference"""
    import tensorflow as tf

    model_path = model_dir / f"lstm_{target}.keras"
    meta_path  = model_dir / f"lstm_{target}_meta.pkl"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"ไม่พบ LSTM model สำหรับ target='{target}'\n"
            f"  ต้องการ: {model_path}\n"
            f"  ต้องการ: {meta_path}\n"
            f"  → รัน: python -m src.trainnig.lstm_trainer"
        )

    model = tf.keras.models.load_model(str(model_path))
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    meta["model"] = model
    return meta


# ══════════════════════════════════════════════════════════════════════════════
#  8. Inference Helper (ใช้ใน app.py)
# ══════════════════════════════════════════════════════════════════════════════
def predict_lstm(
    df_clean: pd.DataFrame,
    model_dir: Path = MODEL_DIR,
) -> pd.DataFrame:
    """
    ทำนาย 24h ข้างหน้าด้วย LSTM

    Parameters
    ----------
    df_clean  : DataFrame (Bangkok-naive index, คอลัมน์ temp, hum)
                ต้องมีอย่างน้อย LOOKBACK_STEPS แถว
    model_dir : โฟลเดอร์ที่มี lstm_temp.keras / lstm_hum.keras

    Returns
    -------
    DataFrame  index = timestamp อนาคต (Bangkok), คอลัมน์ temp_pred, hum_pred
    """
    import time as _time

    BANGKOK_OFFSET_SEC = 7 * 3600
    now_bkk = pd.Timestamp(_time.time() + BANGKOK_OFFSET_SEC, unit="s")

    # ตัด future rows จาก interpolation ออก
    df_past = df_clean[df_clean.index <= now_bkk].copy()
    if df_past.empty:
        df_past = df_clean.copy()

    last_ts = df_past.index[-1]
    base_ts = last_ts.ceil("h")

    log.info("🔮  LSTM forecasting 24h ahead")
    log.info("    last_real = %s   base_ts = %s",
             last_ts.strftime("%Y-%m-%d %H:%M"),
             base_ts.strftime("%Y-%m-%d %H:%M"))

    # ── Build input sequence ─────────────────────────────────────────────────
    if len(df_past) < LOOKBACK_STEPS:
        raise RuntimeError(
            f"ข้อมูลไม่เพียงพอ: ต้องการ {LOOKBACK_STEPS} rows "
            f"แต่มีแค่ {len(df_past)}"
        )

    df_seq = df_past.tail(LOOKBACK_STEPS).copy()
    raw_features = add_time_features(df_seq)       # (144, 8)

    records = {}
    for target in ["temp", "hum"]:
        bundle    = load_lstm_bundle(target, model_dir)
        scaler_X  = bundle["scaler_X"]
        scaler_y  = bundle["scaler_y"]
        model     = bundle["model"]

        X_scaled  = scaler_X.transform(raw_features)     # (144, 8)
        X_input   = X_scaled.reshape(1, LOOKBACK_STEPS, N_FEATURES)  # (1, 144, 8)

        y_scaled  = model.predict(X_input, verbose=0)[0]              # (24,)
        y_pred    = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()  # (24,)
        records[target] = y_pred

        log.info("    %s: %s", target,
                 "  ".join(f"+{h+1}h={v:.2f}" for h, v in enumerate(y_pred[:6])) + " ...")

    # ── Build forecast DataFrame ──────────────────────────────────────────────
    timestamps = [base_ts + pd.Timedelta(hours=h + 1) for h in range(N_HOURS_AHEAD)]
    forecast_df = pd.DataFrame({
        "temp_pred": np.round(records["temp"], 2),
        "hum_pred":  np.round(records["hum"], 2),
    }, index=pd.DatetimeIndex(timestamps))

    log.info("    ✅ LSTM forecast complete: %d hourly points", len(forecast_df))
    return forecast_df


# ══════════════════════════════════════════════════════════════════════════════
#  9. Training Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def train_lstm_target(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    target:   str,
    model_dir: Path        = MODEL_DIR,
    lookback:  int         = LOOKBACK_STEPS,
    epochs:    int         = MAX_EPOCHS,
    save_plots: bool       = True,
) -> dict:
    """เทรน LSTM สำหรับ target เดียว แล้ว save model"""
    import tensorflow as tf

    log.info("")
    log.info("━" * 62)
    log.info("  🧠  Training LSTM — target: %s", target.upper())
    log.info("━" * 62)

    # ── ตรวจ GPU ──────────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    log.info("    GPU: %s", gpus if gpus else "None (CPU training)")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # ── Feature matrix ────────────────────────────────────────────────────────
    target_col_idx = FEATURE_NAMES.index(target)   # 0=temp, 1=hum

    # concat train+val เพื่อ fit scaler บน data เยอะขึ้น
    all_df       = pd.concat([train_df, val_df, test_df])
    raw_all      = add_time_features(all_df)

    raw_train    = add_time_features(train_df)
    raw_val      = add_time_features(val_df)
    raw_test     = add_time_features(test_df)

    # ── Scaler X: fit บน train ────────────────────────────────────────────────
    scaler_X = NumpyScaler().fit(raw_train)
    X_train_s = scaler_X.transform(raw_train)
    X_val_s   = scaler_X.transform(raw_val)
    X_test_s  = scaler_X.transform(raw_test)

    # ── Scaler y: fit บน target column ของ train ──────────────────────────────
    y_col_train = raw_train[:, target_col_idx].reshape(-1, 1)
    scaler_y    = NumpyScaler().fit(y_col_train)

    # ── Sliding window sequences ──────────────────────────────────────────────
    # ต้อง scale y ด้วย (เพื่อให้ model เรียนรู้ได้ง่ายขึ้น)
    def scale_features_y(raw_feat):
        feat_s = scaler_X.transform(raw_feat)
        feat_s[:, target_col_idx] = scaler_y.transform(
            raw_feat[:, target_col_idx].reshape(-1, 1)
        ).flatten()
        return feat_s

    X_train_seq, y_train_seq = make_sequences(
        scale_features_y(raw_train), target_col_idx, lookback)
    X_val_seq,   y_val_seq   = make_sequences(
        scale_features_y(raw_val),   target_col_idx, lookback)
    X_test_seq,  y_test_seq  = make_sequences(
        scale_features_y(raw_test),  target_col_idx, lookback)

    log.info("    Sequences — train: %d  val: %d  test: %d",
             len(X_train_seq), len(X_val_seq), len(X_test_seq))

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(lookback=lookback, n_features=N_FEATURES, n_output=N_HOURS_AHEAD)
    model.summary(print_fn=lambda s: log.info("    %s", s))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_INIT),
        loss="huber",          # Huber loss: robust กับ outlier sensor
        metrics=["mae"],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_path = str(model_dir / f"lstm_{target}_best.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE_STOP,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_FACTOR,
            patience=PATIENCE_LR,
            min_lr=LR_MIN,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("    🚀 Start training  (epochs=%d  batch=%d  patience=%d)",
             epochs, BATCH_SIZE, PATIENCE_STOP)
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val   = float(np.min(history.history["val_loss"]))
    log.info("    ✅ Best epoch: %d   val_loss(Huber): %.6f", best_epoch, best_val)

    # ── Evaluate on Test ──────────────────────────────────────────────────────
    y_pred_scaled = model.predict(X_test_seq, verbose=0)   # (N, 24) — scaled

    # inverse transform y
    y_test_real = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).reshape(y_test_seq.shape)
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)

    log.info("  ── Test Metrics (all 24 horizons combined) ──")
    overall = compute_metrics(y_test_real.flatten(), y_pred_real.flatten(), f"{target} [TEST all]")
    log.info("  ── Per-horizon RMSE ──")
    for h in range(N_HOURS_AHEAD):
        rmse_h = float(np.sqrt(np.mean((y_test_real[:, h] - y_pred_real[:, h]) ** 2)))
        log.info("    +%2dh  RMSE=%.3f", h + 1, rmse_h)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if save_plots:
        plot_training_history(
            history, target,
            model_dir / f"lstm_history_{target}.png"
        )
        plot_predictions_per_horizon(
            y_test_real, y_pred_real, target,
            model_dir / f"lstm_horizon_rmse_{target}.png"
        )

    # ── Scale y ด้วย scaler_y สำหรับ inference ───────────────────────────────
    # scaler_X: scale ทุก feature
    # scaler_y_1d: scale แค่ target column 1D  (ใช้กับ predict output)
    scaler_y_1d = NumpyScaler()
    scaler_y_1d.mean_ = np.array([scaler_y.mean_[0]])
    scaler_y_1d.std_  = np.array([scaler_y.std_[0]])

    # Refit scaler_X แบบ standard (ไม่ทับ target column)
    scaler_X_infer = NumpyScaler().fit(raw_train)

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info("  💾  Saving model...")
    save_lstm_bundle(model, scaler_X_infer, scaler_y_1d, target, model_dir)

    return {
        "target":     target,
        "best_epoch": best_epoch,
        "val_loss":   best_val,
        **overall,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  10. Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════
def run_lstm_training(
    targets:    list  = None,
    model_dir:  Path  = MODEL_DIR,
    train_path: Path  = TRAIN_PATH,
    val_path:   Path  = VAL_PATH,
    test_path:  Path  = TEST_PATH,
    clean_path: Path  = CLEAN_PATH,
    lookback:   int   = LOOKBACK_STEPS,
    epochs:     int   = MAX_EPOCHS,
    use_presplit: bool = True,
    save_plots:   bool = True,
) -> None:
    if targets is None:
        targets = ["temp", "hum"]

    os.makedirs(model_dir, exist_ok=True)

    log.info("=" * 62)
    log.info("  LSTM TRAINING — Sensor EM-300")
    log.info("  Architecture : Bidirectional LSTM × 2 + Dense × 2")
    log.info("  Lookback     : %d steps = %dh",  lookback, lookback * FREQ_MIN // 60)
    log.info("  Output       : %d hourly predictions", N_HOURS_AHEAD)
    log.info("  Targets      : %s", targets)
    log.info("  Max epochs   : %d  (early stopping patience=%d)", epochs, PATIENCE_STOP)
    log.info("=" * 62)

    # โหลด data
    if use_presplit and all(p.exists() for p in [train_path, val_path, test_path]):
        def _load(p):
            df = pd.read_csv(p, index_col="_time", parse_dates=True).dropna()
            return df
        train_df, val_df, test_df = _load(train_path), _load(val_path), _load(test_path)
    else:
        log.warning("⚠️  ไม่พบ split files → split จาก clean CSV")
        df       = pd.read_csv(clean_path, index_col="_time", parse_dates=True).dropna()
        n        = len(df)
        n_train  = int(n * TRAIN_RATIO)
        n_val    = int(n * VAL_RATIO)
        train_df = df.iloc[:n_train]
        val_df   = df.iloc[n_train : n_train + n_val]
        test_df  = df.iloc[n_train + n_val :]

    log.info("    train: %d  val: %d  test: %d",
             len(train_df), len(val_df), len(test_df))

    results = []
    for target in targets:
        res = train_lstm_target(
            train_df, val_df, test_df,
            target=target,
            model_dir=model_dir,
            lookback=lookback,
            epochs=epochs,
            save_plots=save_plots,
        )
        results.append(res)

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 62)
    log.info("  ✅  LSTM TRAINING COMPLETE")
    log.info("  %-8s  %-6s  %-6s  %-6s  %-8s  %-6s",
             "Target", "Epoch", "RMSE", "MAE", "MAPE%", "R²")
    log.info("  " + "─" * 50)
    for r in results:
        log.info("  %-8s  %-6d  %-6.3f  %-6.3f  %-8.2f  %-6.4f",
                 r["target"], r["best_epoch"], r["RMSE"],
                 r["MAE"], r["MAPE"], r["R2"])
    log.info("")
    log.info("  Models saved → %s/lstm_{{target}}.keras", model_dir)
    log.info("=" * 62)

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(model_dir / "lstm_training_summary.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LSTM Training — Bidirectional LSTM for 24h forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target",     choices=["temp", "hum", "both"], default="both")
    parser.add_argument("--epochs",     type=int, default=MAX_EPOCHS,
                        help="จำนวน epoch สูงสุด (early stopping จะหยุดก่อนถ้า val ไม่ improve)")
    parser.add_argument("--lookback",   type=int, default=LOOKBACK_STEPS,
                        help="lookback window (steps, 1 step = 10 min)")
    parser.add_argument("--model-dir",  type=Path, default=MODEL_DIR)
    parser.add_argument("--no-presplit", action="store_true")
    parser.add_argument("--no-plots",   action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args    = _parse_args()
    targets = ["temp", "hum"] if args.target == "both" else [args.target]

    run_lstm_training(
        targets      = targets,
        model_dir    = args.model_dir,
        lookback     = args.lookback,
        epochs       = args.epochs,
        use_presplit = not args.no_presplit,
        save_plots   = not args.no_plots,
    )
