from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# ──────────────────────────────────────────────
CLEAN_PATH  = Path("data/processed/sensor_data_clean.csv")
SPLIT_DIR   = Path("data/processed")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15   # test = 1 - train - val

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

#  Load
def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    log.info("Loading: %s", path)
    df = pd.read_csv(path, index_col="_time", parse_dates=True)
    before = len(df)
    df = df.dropna()
    log.info("    Total rows (non-NaN): %d  (dropped %d)", len(df), before - len(df))
    return df

#  Split
def split_data(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n       = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = df.iloc[:n_train]
    val   = df.iloc[n_train : n_train + n_val]
    test  = df.iloc[n_train + n_val :]

    test_ratio = 1 - train_ratio - val_ratio

    log.info("")
    log.info("Split Summary (chronological, no shuffle)")
    log.info("  %-6s │ %5d rows (%4.0f%%)  │  %s  →  %s",
             "Train", len(train), train_ratio * 100,
             train.index[0].strftime("%Y-%m-%d %H:%M"),
             train.index[-1].strftime("%Y-%m-%d %H:%M"))
    log.info("  %-6s │ %5d rows (%4.0f%%)  │  %s  →  %s",
             "Val",   len(val),   val_ratio * 100,
             val.index[0].strftime("%Y-%m-%d %H:%M"),
             val.index[-1].strftime("%Y-%m-%d %H:%M"))
    log.info("  %-6s │ %5d rows (%4.0f%%)  │  %s  →  %s",
             "Test",  len(test),  test_ratio * 100,
             test.index[0].strftime("%Y-%m-%d %H:%M"),
             test.index[-1].strftime("%Y-%m-%d %H:%M"))

    return train, val, test

#  Save
def save_splits(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    out_dir: Path = SPLIT_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, df in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=True, index_label="_time")
        log.info("Saved %-6s → %s  (%d rows)", name, path, len(df))

#  Pipeline
def run_split(
    clean_path:  Path  = CLEAN_PATH,
    out_dir:     Path  = SPLIT_DIR,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("=" * 52)
    log.info("  DATA SPLIT — Sensor EM-300")
    log.info("  Ratios : train=%.0f%%  val=%.0f%%  test=%.0f%%",
             train_ratio * 100, val_ratio * 100,
             (1 - train_ratio - val_ratio) * 100)
    log.info("=" * 52)

    df = load_clean(clean_path)

    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio ต้องน้อยกว่า 1.0")

    train, val, test = split_data(df, train_ratio, val_ratio)
    save_splits(train, val, test, out_dir)

    log.info("")
    log.info("DONE  →  %s/", out_dir)

    return train, val, test


#  CLI
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chronological train/val/test split สำหรับ sensor time-series",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",    type=Path,  default=CLEAN_PATH,
                        help="Path ไปยัง clean CSV")
    parser.add_argument("--out-dir", type=Path,  default=SPLIT_DIR,
                        help="โฟลเดอร์สำหรับ save split CSV")
    parser.add_argument("--train",   type=float, default=TRAIN_RATIO,
                        help="สัดส่วน train (0–1)")
    parser.add_argument("--val",     type=float, default=VAL_RATIO,
                        help="สัดส่วน val (0–1)  test = 1 - train - val")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_split(
        clean_path  = args.data,
        out_dir     = args.out_dir,
        train_ratio = args.train,
        val_ratio   = args.val,
    )
