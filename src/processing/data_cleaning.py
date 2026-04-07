from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RAW_PATH     = Path("data/raw/sensor_data_raw.csv")
DEFAULT_CLEAN_PATH   = Path("data/processed/sensor_data_clean.csv")
TIMEZONE_TARGET      = "Asia/Bangkok"
RESAMPLE_FREQ        = "10min"
INTERP_LIMIT_HOURS   = 6           # ไม่ interpolate ข้าม gap > 6 ชม.
OUTLIER_IQR_FACTOR   = 1.5         # IQR fence multiplier
OUTLIER_ZSCORE_THRESH = 3.0        # rolling z-score threshold
ROLLING_WINDOW_ROWS  = 144         # 144 * 10min = 1 วัน

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════
#  STEP 1 – โหลดข้อมูล
# ══════════════════════════════════════════════
def load_raw(path: Path) -> pd.DataFrame:
    """
    โหลด raw CSV และ parse คอลัมน์ _time เป็น datetime (tz-aware)
    คืน DataFrame ที่เรียงตาม _time
    """
    log.info("📂  Loading raw data: %s", path)
    df = pd.read_csv(path)
    df["_time"] = pd.to_datetime(df["_time"], format="mixed", utc=True)
    df = df.sort_values("_time").reset_index(drop=True)
    log.info("    Shape: %s  |  Range: %s → %s",
             df.shape,
             df["_time"].min().strftime("%Y-%m-%d %H:%M"),
             df["_time"].max().strftime("%Y-%m-%d %H:%M"))
    return df


# ══════════════════════════════════════════════
#  STEP 2 – ลบ Near-Duplicate Timestamps
# ══════════════════════════════════════════════
def remove_near_duplicates(df: pd.DataFrame, round_freq: str = "1min") -> pd.DataFrame:
    """
    Sensor บางตัวส่งค่าหลายครั้งในช่วง < 1 วินาที
    แก้ด้วยการ round timestamp ไป 1 นาที แล้ว group mean

    Parameters
    ----------
    df         : DataFrame ที่มีคอลัมน์ _time, hum, temp
    round_freq : ความถี่ที่ใช้ round (default: '1min')
    """
    before = len(df)

    # ตรวจจับ near-duplicate
    diff_sec = df["_time"].diff().dt.total_seconds()
    n_near_dupes = (diff_sec < 1).sum()
    log.info("🔍  Near-duplicate rows (Δt < 1s): %d  (%.1f%%)",
             n_near_dupes, n_near_dupes / before * 100)

    # round แล้ว aggregate
    df = df[["_time", "hum", "temp"]].copy()
    df["_time_rounded"] = df["_time"].dt.round(round_freq)

    df_dedup = (
        df.groupby("_time_rounded", as_index=False)
          .agg(hum=("hum", "mean"), temp=("temp", "mean"))
          .rename(columns={"_time_rounded": "_time"})
    )

    after = len(df_dedup)
    log.info("✅  After dedup: %d → %d rows  (removed %d)", before, after, before - after)
    return df_dedup

def convert_timezone(df: pd.DataFrame, tz: str = TIMEZONE_TARGET) -> pd.DataFrame:
    """
    แปลง _time จาก UTC → tz ที่ระบุ แล้วลบ tzinfo ออก
    เพื่อให้เป็น naive datetime สำหรับ resample

    Parameters
    ----------
    df : DataFrame ที่มีคอลัมน์ _time (tz-aware UTC)
    tz : timezone เป้าหมาย (default: 'Asia/Bangkok')
    """
    log.info("🕐  Converting timezone: UTC → %s", tz)
    df = df.copy()
    df["_time"] = (
        df["_time"]
        .dt.tz_convert(tz)
        .dt.tz_localize(None)   # ทำให้เป็น naive datetime
    )
    log.info("    New range: %s → %s",
             df["_time"].min().strftime("%Y-%m-%d %H:%M"),
             df["_time"].max().strftime("%Y-%m-%d %H:%M"))
    return df


# ══════════════════════════════════════════════
#  STEP 4 – Resample เป็น Regular Intervals
# ══════════════════════════════════════════════
def resample_regular(df: pd.DataFrame, freq: str = RESAMPLE_FREQ) -> pd.DataFrame:
    """
    สร้าง time index สม่ำเสมอทุก `freq` แล้ว resample ด้วย mean
    ช่วงที่ขาดข้อมูลจะกลายเป็น NaN รอการ interpolate

    Parameters
    ----------
    df   : DataFrame ที่มีคอลัมน์ _time (naive datetime), hum, temp
    freq : ความถี่ในการ resample (default: '10min')
    """
    log.info("📊  Resampling to %s intervals...", freq)
    df_idx = df.set_index("_time")[["hum", "temp"]]
    df_res = df_idx.resample(freq).mean()

    n_nan = df_res.isna().sum()
    log.info("    Rows after resample: %d  |  Missing → temp: %d, hum: %d",
             len(df_res), n_nan["temp"], n_nan["hum"])
    return df_res


# ══════════════════════════════════════════════
#  STEP 5 – Interpolate ช่องว่าง
# ══════════════════════════════════════════════
def interpolate_gaps(
    df: pd.DataFrame,
    limit_hours: float = INTERP_LIMIT_HOURS,
    freq: str = RESAMPLE_FREQ,
) -> pd.DataFrame:
    """
    Interpolate NaN โดยใช้ method='time' (คำนึงถึง interval จริง)
    ไม่ interpolate ข้าม gap ที่ใหญ่กว่า limit_hours

    Parameters
    ----------
    df          : DataFrame จาก resample_regular (index = datetime)
    limit_hours : ขีดจำกัดสูงสุดในการ interpolate (ชั่วโมง)
    freq        : ความถี่ของ time index (ใช้คำนวณ limit rows)
    """
    # คำนวณ limit เป็นจำนวน rows
    freq_min   = pd.tseries.frequencies.to_offset(freq).nanos // 60_000_000_000
    limit_rows = int(limit_hours * 60 / freq_min)

    log.info("🔧  Interpolating gaps  (limit = %g hr = %d rows)...", limit_hours, limit_rows)
    df_interp = df.interpolate(method="time", limit=limit_rows)

    remaining = df_interp.isna().sum()
    log.info("    Remaining NaN after interpolate → temp: %d, hum: %d",
             remaining["temp"], remaining["hum"])

    if remaining.sum() > 0:
        log.info("    ℹ️  NaN ที่เหลือคือ gap > %g ชม. (ไม่ได้ interpolate ข้ามโดยตั้งใจ)", limit_hours)

    return df_interp


# ══════════════════════════════════════════════
#  STEP 6 – ตรวจจับ & Clip Outliers
# ══════════════════════════════════════════════
def handle_outliers(
    df: pd.DataFrame,
    iqr_factor: float = OUTLIER_IQR_FACTOR,
    zscore_thresh: float = OUTLIER_ZSCORE_THRESH,
    rolling_window: int = ROLLING_WINDOW_ROWS,
    clip: bool = True,
) -> pd.DataFrame:
    """
    ตรวจจับ outlier ด้วย 2 วิธี:
      1. IQR method  : ค่านอก [Q1 - factor*IQR, Q3 + factor*IQR]
      2. Rolling z-score : |z| > threshold ใน window 1 วัน

    ถ้า clip=True จะ clip ค่าให้อยู่ใน IQR fence
    ถ้า clip=False จะ flag เท่านั้น ไม่แก้ไขค่า

    Parameters
    ----------
    df             : DataFrame ที่มี index datetime และคอลัมน์ hum, temp
    iqr_factor     : IQR multiplier (default: 1.5)
    zscore_thresh  : z-score threshold (default: 3.0)
    rolling_window : ขนาด window สำหรับ rolling z-score (rows)
    clip           : True = clip ค่า outlier ให้อยู่ใน IQR bounds
    """
    log.info("🔎  Outlier detection  (IQR ×%.1f  |  rolling z-score > %.1f)...",
             iqr_factor, zscore_thresh)
    df = df.copy()

    for col in ["temp", "hum"]:
        valid = df[col].dropna()

        # ── IQR ──
        Q1, Q3 = valid.quantile(0.25), valid.quantile(0.75)
        IQR    = Q3 - Q1
        lo     = Q1 - iqr_factor * IQR
        hi     = Q3 + iqr_factor * IQR
        n_iqr  = ((df[col] < lo) | (df[col] > hi)).sum()
        log.info("  %s | IQR fence [%.1f, %.1f] | IQR outliers: %d", col, lo, hi, n_iqr)

        # ── Rolling Z-score ──
        roll_mean = df[col].rolling(rolling_window, center=True, min_periods=1).mean()
        roll_std  = df[col].rolling(rolling_window, center=True, min_periods=1).std()
        z_score   = (df[col] - roll_mean) / roll_std.replace(0, np.nan)
        n_z       = (z_score.abs() > zscore_thresh).sum()
        log.info("  %s | rolling z-score spikes (|z|>%.1f): %d", col, zscore_thresh, n_z)

        # ── Clip (ถ้าเปิดใช้) ──
        if clip:
            before_clip = df[col].copy()
            df[col] = df[col].clip(lower=lo, upper=hi)
            n_clipped = (before_clip != df[col]).sum()
            if n_clipped:
                log.info("  %s → clipped %d values to [%.1f, %.1f]", col, n_clipped, lo, hi)

    return df


# ══════════════════════════════════════════════
#  STEP 7 – บันทึกผลลัพธ์
# ══════════════════════════════════════════════
def save_clean(df: pd.DataFrame, path: Path) -> None:
    """
    บันทึก DataFrame ที่สะอาดแล้วเป็น CSV

    Parameters
    ----------
    df   : DataFrame ที่มี index datetime
    path : path ปลายทาง
    """
    os.makedirs(path.parent, exist_ok=True)
    df.to_csv(path, index=True, index_label="_time")
    log.info("💾  Saved clean data → %s  (%d rows)", path, len(df))


# ══════════════════════════════════════════════
#  PIPELINE หลัก
# ══════════════════════════════════════════════
def run_pipeline(
    raw_path: Path  = DEFAULT_RAW_PATH,
    clean_path: Path = DEFAULT_CLEAN_PATH,
    resample_freq: str = RESAMPLE_FREQ,
    interp_limit_hours: float = INTERP_LIMIT_HOURS,
    clip_outliers: bool = True,
) -> pd.DataFrame:
    """
    รัน pipeline ทุกขั้นตอนตามลำดับและ return DataFrame สุดท้าย

    Parameters
    ----------
    raw_path           : path ไฟล์ raw CSV
    clean_path         : path สำหรับบันทึก CSV ที่สะอาดแล้ว
    resample_freq      : ความถี่ resample (default: '10min')
    interp_limit_hours : ขีดจำกัด interpolate (ชั่วโมง)
    clip_outliers      : True = clip ค่า outlier ด้วย IQR fence
    """
    log.info("=" * 55)
    log.info("  DATA CLEANING PIPELINE — Sensor EM-300")
    log.info("=" * 55)

    df = load_raw(raw_path)
    df = remove_near_duplicates(df)
    df = convert_timezone(df)
    df = resample_regular(df, freq=resample_freq)
    df = interpolate_gaps(df, limit_hours=interp_limit_hours, freq=resample_freq)
    df = handle_outliers(df, clip=clip_outliers)
    save_clean(df, clean_path)

    log.info("=" * 55)
    log.info("  ✅ DONE  →  %s", clean_path)
    log.info("  Rows (non-NaN): %d / %d", df.notna().all(axis=1).sum(), len(df))
    log.info("=" * 55)

    return df


# ══════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data Cleaning Pipeline — Sensor EM-300 Temperature & Humidity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw",      type=Path, default=DEFAULT_RAW_PATH,
                        help="Path ไปยัง raw CSV")
    parser.add_argument("--output",   type=Path, default=DEFAULT_CLEAN_PATH,
                        help="Path สำหรับบันทึก clean CSV")
    parser.add_argument("--freq",     type=str,  default=RESAMPLE_FREQ,
                        help="ความถี่ resample เช่น '10min', '5min', '1h'")
    parser.add_argument("--limit-hours", type=float, default=INTERP_LIMIT_HOURS,
                        help="ขีดจำกัด interpolate สูงสุด (ชั่วโมง)")
    parser.add_argument("--no-clip",  action="store_true",
                        help="ปิด outlier clipping (ตรวจจับอย่างเดียว)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        raw_path           = args.raw,
        clean_path         = args.output,
        resample_freq      = args.freq,
        interp_limit_hours = args.limit_hours,
        clip_outliers      = not args.no_clip,
    )
