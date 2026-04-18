"""
tflite_inference.py  —  Lightweight LSTM inference for Raspberry Pi
=====================================================================

ใช้ tflite-runtime แทน tensorflow เต็มรูปแบบ
ทำงานร่วมกับ models ที่ export จาก run_pipeline.py --export-tflite

Usage (drop-in แทน predict_lstm ใน app.py):
    from src.trainnig.tflite_inference import predict_tflite
    forecast_df = predict_tflite(df_clean, tflite_dir=Path("models/tflite"))
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LOOKBACK_STEPS = 144       # 24h × 6 steps/h
N_FEATURES     = 8
N_HOURS_AHEAD  = 24
STEPS_PER_HOUR = 6

TFLITE_DIR_DEFAULT = Path("models/tflite")

FEATURE_NAMES = [
    "temp", "hum",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin","month_cos",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Auto-select TFLite interpreter
# ─────────────────────────────────────────────────────────────────────────────
def _get_interpreter(tflite_path: Path):
    """
    โหลด TFLite interpreter:
      1. ลองใช้ tflite_runtime ก่อน (เบา ~20MB)
      2. ถ้าไม่มี ใช้ tensorflow.lite แทน (ต้องติดตั้ง TF เต็ม)
    """
    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=str(tflite_path))
        log.debug("  Using tflite_runtime ✅")
    except ImportError:
        try:
            import tensorflow as tf
            interp = tf.lite.Interpreter(model_path=str(tflite_path))
            log.debug("  Using tensorflow.lite ✅")
        except ImportError:
            raise ImportError(
                "ต้องการ tflite-runtime หรือ tensorflow\n"
                "RPi: pip install tflite-runtime\n"
                "PC:  pip install tensorflow"
            )
    return interp


# ─────────────────────────────────────────────────────────────────────────────
#  Feature engineering  (เหมือนกับใน lstm_trainer.py)
# ─────────────────────────────────────────────────────────────────────────────
def _add_time_features(df: pd.DataFrame) -> np.ndarray:
    idx       = df.index
    hour_rad  = 2 * np.pi * idx.hour / 24
    dow_rad   = 2 * np.pi * idx.dayofweek / 7
    month_rad = 2 * np.pi * (idx.month - 1) / 12
    return np.column_stack([
        df["temp"].values,
        df["hum"].values,
        np.sin(hour_rad), np.cos(hour_rad),
        np.sin(dow_rad),  np.cos(dow_rad),
        np.sin(month_rad),np.cos(month_rad),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Single-target TFLite inference
# ─────────────────────────────────────────────────────────────────────────────
def _predict_one(
    raw_features: np.ndarray,
    target: str,
    tflite_dir: Path,
) -> np.ndarray:
    """
    ทำนาย 24h สำหรับ target เดียว ด้วย TFLite

    Returns
    -------
    np.ndarray shape (24,)  — ค่า predict รายชั่วโมง (original scale)
    """
    tflite_path = tflite_dir / f"lstm_{target}.tflite"
    meta_path   = tflite_dir / f"lstm_{target}_meta.pkl"

    if not tflite_path.exists():
        raise FileNotFoundError(
            f"ไม่พบ TFLite model: {tflite_path}\n"
            f"รัน: python run_pipeline.py --skip-fetch --skip-clean --export-tflite"
        )

    # ── Load scaler metadata ─────────────────────────────────────────────────
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    scaler_X = meta["scaler_X"]
    scaler_y = meta["scaler_y"]

    # ── Prepare input ────────────────────────────────────────────────────────
    X_scaled = scaler_X.transform(raw_features)             # (144, 8)
    X_input  = X_scaled.reshape(1, LOOKBACK_STEPS, N_FEATURES).astype(np.float32)

    # ── Run TFLite ───────────────────────────────────────────────────────────
    interp = _get_interpreter(tflite_path)
    interp.allocate_tensors()

    inp_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]

    interp.set_tensor(inp_idx, X_input)
    interp.invoke()

    y_scaled = interp.get_tensor(out_idx)[0]                # (24,)

    # ── Inverse scale ────────────────────────────────────────────────────────
    y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    return y_pred


# ─────────────────────────────────────────────────────────────────────────────
#  Public API  (drop-in แทน predict_lstm)
# ─────────────────────────────────────────────────────────────────────────────
def predict_tflite(
    df_clean: pd.DataFrame,
    tflite_dir: Path = TFLITE_DIR_DEFAULT,
) -> pd.DataFrame:
    """
    ทำนาย 24h ข้างหน้าด้วย TFLite (เหมาะสำหรับ Raspberry Pi)

    Parameters
    ----------
    df_clean   : DataFrame (Bangkok-naive index, คอลัมน์ temp, hum)
                 ต้องมีอย่างน้อย 144 แถว (24h ย้อนหลัง)
    tflite_dir : โฟลเดอร์ที่มี lstm_temp.tflite / lstm_hum.tflite

    Returns
    -------
    DataFrame  index = timestamp อนาคต (Bangkok naive),
               คอลัมน์ temp_pred, hum_pred (24 rows)
    """
    import time as _time

    if len(df_clean) < LOOKBACK_STEPS:
        raise RuntimeError(
            f"ข้อมูลไม่เพียงพอ: ต้องการ {LOOKBACK_STEPS} rows "
            f"แต่มีแค่ {len(df_clean)}"
        )

    # ตัดเอา 144 แถวสุดท้าย
    df_seq       = df_clean.tail(LOOKBACK_STEPS).copy()
    raw_features = _add_time_features(df_seq)        # (144, 8)

    last_ts  = df_seq.index[-1]
    base_ts  = last_ts.ceil("h")

    log.info("🔮  TFLite forecasting 24h ahead")
    log.info("    last_real = %s   base_ts = %s",
             last_ts.strftime("%Y-%m-%d %H:%M"),
             base_ts.strftime("%Y-%m-%d %H:%M"))

    records = {}
    for target in ["temp", "hum"]:
        y_pred = _predict_one(raw_features, target, tflite_dir)
        records[target] = y_pred
        log.info("    %s: %s", target,
                 "  ".join(f"+{h+1}h={v:.2f}" for h, v in enumerate(y_pred[:6])) + " ...")

    # ── Build forecast DataFrame ──────────────────────────────────────────────
    timestamps  = [base_ts + pd.Timedelta(hours=h + 1) for h in range(N_HOURS_AHEAD)]
    forecast_df = pd.DataFrame({
        "temp_pred": np.round(records["temp"], 2),
        "hum_pred":  np.round(records["hum"],  2),
    }, index=pd.DatetimeIndex(timestamps))

    log.info("    ✅  TFLite forecast complete: %d hourly points", len(forecast_df))
    return forecast_df
