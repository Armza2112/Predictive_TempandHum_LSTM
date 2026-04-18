"""
run_pipeline.py  —  Single-command training pipeline for Sensor EM-300
=======================================================================

Usage examples
--------------
# ดึงจาก InfluxDB → clean → split → train → (optional) export TFLite
python run_pipeline.py

# ใช้ CSV แทน InfluxDB (เช่น sensor_data.csv จาก Ana_data)
python run_pipeline.py --source csv --csv path/to/sensor_data.csv

# ข้าม fetch (มี data/raw/sensor_data_raw.csv อยู่แล้ว)
python run_pipeline.py --skip-fetch

# ข้าม fetch + clean (มี data/processed/sensor_data_clean.csv อยู่แล้ว)
python run_pipeline.py --skip-fetch --skip-clean

# เทรนแค่ temp และ export TFLite สำหรับ RPi
python run_pipeline.py --skip-fetch --skip-clean --target temp --export-tflite

# ตรวจสอบ library compatibility สำหรับ RPi ก่อนรัน
python run_pipeline.py --check-raspi
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent
RAW_CSV         = ROOT_DIR / "data" / "raw"    / "sensor_data_raw.csv"
CLEAN_CSV       = ROOT_DIR / "data" / "processed" / "sensor_data_clean.csv"
PROCESSED_DIR   = ROOT_DIR / "data" / "processed"
MODEL_DIR       = ROOT_DIR / "models"
TFLITE_DIR      = ROOT_DIR / "models" / "tflite"

# ─────────────────────────────────────────────────────────────────────────────
#  Logger
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        SENSOR EM-300  —  TRAINING PIPELINE               ║
║        Input  : 1 day back  (144 steps × 10 min)         ║
║        Output : 1 day ahead (24 hourly predictions)      ║
╚══════════════════════════════════════════════════════════╝"""


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 0 — Raspberry Pi compatibility check
# ══════════════════════════════════════════════════════════════════════════════
def check_raspi_compat() -> None:
    """
    ตรวจสอบว่า library ที่ติดตั้งอยู่ใช้งานได้กับ Raspberry Pi หรือไม่
    แสดงผลพร้อมคำแนะนำ
    """
    import importlib
    import platform

    arch   = platform.machine()
    is_arm = arch.startswith(("arm", "aarch"))

    print("\n" + "═" * 60)
    print("  RPi Compatibility Check")
    print(f"  Architecture : {arch}  {'← ARM ✅' if is_arm else '← x86 (running on PC)'}")
    print("═" * 60)

    libs = [
        ("tensorflow",      "2.15.1", "⚠️  Heavy (500MB+). ใช้ tflite-runtime สำหรับ inference แทน"),
        ("tflite_runtime",  None,     "✅  Lightweight inference engine สำหรับ RPi"),
        ("numpy",           None,     "✅  ARM-optimized ใช้ได้"),
        ("pandas",          None,     "✅  ใช้ได้ดีบน RPi"),
        ("influxdb_client", None,     "✅  ใช้ได้"),
        ("paho.mqtt.client",None,     "✅  เบามาก ใช้ได้"),
        ("sklearn",         None,     "✅  ใช้ได้"),
        ("scipy",           None,     "✅  ใช้ได้"),
    ]

    for lib, version_warn, note in libs:
        try:
            mod = importlib.import_module(lib.replace("-", "_"))
            ver = getattr(mod, "__version__", "?")
            status = f"✅  v{ver}"
        except ImportError:
            status = "❌  ไม่ได้ติดตั้ง"
        print(f"  {lib:<22} {status:<20} {note}")

    print()
    print("  📌  สรุปสำหรับ Raspberry Pi:")
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  Training    → ทำบน PC (TF BiLSTM ต้องการ RAM มาก) │")
    print("  │  Inference   → ใช้ tflite-runtime แทน tensorflow    │")
    print("  │  Service     → influxdb-client + paho-mqtt ✅ ใช้ได้ │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print("  วิธีติดตั้งบน RPi (inference only):")
    print("  pip install -r requirements-raspi.txt")
    print()
    print("  วิธี export TFLite หลังเทรนบน PC:")
    print("  python run_pipeline.py --skip-fetch --skip-clean --export-tflite")
    print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Fetch Data
# ══════════════════════════════════════════════════════════════════════════════
def step_fetch(source: str, csv_path: str | None = None) -> None:
    log.info("─" * 60)
    log.info("  STEP 1 / 4  —  FETCH DATA  (source=%s)", source)
    log.info("─" * 60)

    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)

    if source == "csv":
        # ─── ใช้ CSV ที่ระบุ ─────────────────────────────────────────────────
        if not csv_path:
            raise ValueError("--csv ต้องระบุ path เมื่อใช้ --source csv")
        src = Path(csv_path)
        if not src.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์: {src}")

        # ตรวจสอบ columns ที่จำเป็น
        import pandas as pd
        df_peek = pd.read_csv(src, nrows=2)
        required = {"_time", "temp", "hum"}
        if not required.issubset(df_peek.columns):
            raise ValueError(
                f"CSV ต้องมีคอลัมน์: {required}\n"
                f"พบ: {set(df_peek.columns)}"
            )

        shutil.copy2(src, RAW_CSV)
        log.info("  ✅  Copied CSV → %s", RAW_CSV)

    elif source == "influx":
        # ─── ดึงจาก InfluxDB ─────────────────────────────────────────────────
        try:
            from src.database.influx import DatabaseHandler
        except ImportError as e:
            raise ImportError(f"ไม่สามารถ import DatabaseHandler: {e}") from e

        db = DatabaseHandler()
        try:
            log.info("  📡  Fetching all data from InfluxDB (range_start=0)...")
            df = db.fetch_data(measurement="EM-300", range_start="0")
        finally:
            db.close()

        if df.empty:
            raise RuntimeError("InfluxDB ไม่มีข้อมูล")

        df.to_csv(RAW_CSV, index=False)
        log.info("  ✅  Saved %d rows → %s", len(df), RAW_CSV)

    else:
        raise ValueError(f"source ไม่รู้จัก: {source!r}  (ต้องเป็น 'csv' หรือ 'influx')")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Clean Data
# ══════════════════════════════════════════════════════════════════════════════
def step_clean() -> None:
    log.info("─" * 60)
    log.info("  STEP 2 / 4  —  CLEAN DATA")
    log.info("─" * 60)

    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"ไม่พบ raw CSV: {RAW_CSV}\n"
            "รัน step fetch ก่อน หรือใช้ --skip-fetch ถ้ามีไฟล์อยู่แล้ว"
        )

    from src.processing.data_cleaning import run_pipeline as clean_pipeline
    df_clean = clean_pipeline(
        raw_path           = RAW_CSV,
        clean_path         = CLEAN_CSV,
        resample_freq      = "10min",
        interp_limit_hours = 6,
        clip_outliers      = True,
    )
    log.info("  ✅  Clean complete  (%d rows, %d non-NaN)",
             len(df_clean), df_clean.notna().all(axis=1).sum())


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Split Data
# ══════════════════════════════════════════════════════════════════════════════
def step_split(train_ratio: float = 0.70, val_ratio: float = 0.15) -> None:
    log.info("─" * 60)
    log.info("  STEP 3 / 4  —  SPLIT  (train=%.0f%%  val=%.0f%%  test=%.0f%%)",
             train_ratio * 100, val_ratio * 100,
             (1 - train_ratio - val_ratio) * 100)
    log.info("─" * 60)

    if not CLEAN_CSV.exists():
        raise FileNotFoundError(f"ไม่พบ clean CSV: {CLEAN_CSV}")

    from src.processing.data_split import run_split
    run_split(
        clean_path  = CLEAN_CSV,
        out_dir     = PROCESSED_DIR,
        train_ratio = train_ratio,
        val_ratio   = val_ratio,
    )
    log.info("  ✅  Split saved → %s/", PROCESSED_DIR)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Train LSTM
# ══════════════════════════════════════════════════════════════════════════════
def step_train(targets: list[str], epochs: int) -> None:
    log.info("─" * 60)
    log.info("  STEP 4 / 4  —  TRAIN LSTM  (targets=%s  max_epochs=%d)",
             targets, epochs)
    log.info("─" * 60)

    from src.trainnig.lstm_trainer import run_lstm_training
    run_lstm_training(
        targets     = targets,
        model_dir   = MODEL_DIR,
        train_path  = PROCESSED_DIR / "train.csv",
        val_path    = PROCESSED_DIR / "val.csv",
        test_path   = PROCESSED_DIR / "test.csv",
        clean_path  = CLEAN_CSV,
        lookback    = 144,   # 24h × 6 steps/h
        epochs      = epochs,
        use_presplit= True,
        save_plots  = True,
    )
    log.info("  ✅  Training complete  → %s/", MODEL_DIR)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 (optional) — Export to TFLite for Raspberry Pi
# ══════════════════════════════════════════════════════════════════════════════
def step_export_tflite(targets: list[str]) -> None:
    log.info("─" * 60)
    log.info("  STEP 5 (RPi)  —  EXPORT TFLite  (targets=%s)", targets)
    log.info("─" * 60)

    try:
        import tensorflow as tf
    except ImportError:
        log.error("  ❌  tensorflow ไม่ได้ติดตั้ง — ข้าม TFLite export")
        return

    TFLITE_DIR.mkdir(parents=True, exist_ok=True)

    for target in targets:
        keras_path  = MODEL_DIR / f"lstm_{target}.keras"
        tflite_path = TFLITE_DIR / f"lstm_{target}.tflite"

        if not keras_path.exists():
            log.warning("  ⚠️  ไม่พบ %s — ข้าม", keras_path)
            continue

        log.info("  Converting lstm_%s.keras → TFLite...", target)
        model = tf.keras.models.load_model(str(keras_path))

        # ── Convert with dynamic-range quantization (ลด size ~4x, เร็วขึ้นบน RPi) ──
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # int8 dynamic range
        tflite_model = converter.convert()

        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        size_mb = tflite_path.stat().st_size / 1024 / 1024
        log.info("  ✅  Saved: %s  (%.2f MB)", tflite_path, size_mb)

    # ── บันทึก scaler metadata ด้วย (สำหรับ RPi inference) ──────────────────
    import pickle
    for target in targets:
        meta_src  = MODEL_DIR / f"lstm_{target}_meta.pkl"
        meta_dst  = TFLITE_DIR / f"lstm_{target}_meta.pkl"
        if meta_src.exists():
            shutil.copy2(meta_src, meta_dst)
            log.info("  ✅  Copied metadata: %s", meta_dst.name)

    log.info("  📌  วิธีใช้บน RPi:")
    log.info("      1. copy โฟลเดอร์ models/tflite/ ไปไว้บน RPi")
    log.info("      2. pip install tflite-runtime influxdb-client paho-mqtt pandas")
    log.info("      3. รัน: python src/service/app.py --once")
    log.info("         (app.py จะโหลด TFLite อัตโนมัติถ้าไม่มี TF)")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
def run(args: argparse.Namespace) -> None:
    log.info(BANNER)

    t0 = time.time()

    targets = ["temp", "hum"] if args.target == "both" else [args.target]

    # ── STEP 0: RPi check ────────────────────────────────────────────────────
    if args.check_raspi:
        check_raspi_compat()
        return

    # ── STEP 1: Fetch ────────────────────────────────────────────────────────
    if not args.skip_fetch:
        step_fetch(source=args.source, csv_path=args.csv)
    else:
        log.info("  [skip]  STEP 1 — Fetch  (ใช้ %s)", RAW_CSV)

    # ── STEP 2: Clean ────────────────────────────────────────────────────────
    if not args.skip_clean:
        step_clean()
    else:
        log.info("  [skip]  STEP 2 — Clean  (ใช้ %s)", CLEAN_CSV)

    # ── STEP 3: Split ────────────────────────────────────────────────────────
    if not args.skip_split:
        step_split(train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    else:
        log.info("  [skip]  STEP 3 — Split  (ใช้ %s/)", PROCESSED_DIR)

    # ── STEP 4: Train ────────────────────────────────────────────────────────
    if not args.skip_train:
        step_train(targets=targets, epochs=args.epochs)
    else:
        log.info("  [skip]  STEP 4 — Train  (ใช้ models ที่มีอยู่)")

    # ── STEP 5: TFLite export (optional) ────────────────────────────────────
    if args.export_tflite:
        step_export_tflite(targets=targets)

    # ── Done ─────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  ✅  PIPELINE COMPLETE  (%.0fs / %.1fmin)%s║",
             elapsed, elapsed / 60, " " * max(0, 18 - len(f"{elapsed:.0f}")))
    log.info("║                                                          ║")
    log.info("║  Models  → models/lstm_{{temp,hum}}.keras                 ║")
    if args.export_tflite:
        log.info("║  TFLite  → models/tflite/lstm_{{temp,hum}}.tflite        ║")
    log.info("║  Service → python src/service/app.py --once             ║")
    log.info("╚══════════════════════════════════════════════════════════╝")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-command pipeline: fetch → clean → split → train → (export TFLite)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data source ──────────────────────────────────────────────────────────
    src = p.add_argument_group("Data Source")
    src.add_argument(
        "--source", choices=["influx", "csv"], default="influx",
        help="แหล่งข้อมูล: influx = ดึงจาก InfluxDB,  csv = ใช้ไฟล์ CSV",
    )
    src.add_argument(
        "--csv", type=str, default=None, metavar="PATH",
        help="Path ของ CSV (ใช้เมื่อ --source csv)  ต้องมีคอลัมน์ _time, temp, hum",
    )

    # ── Skip flags ───────────────────────────────────────────────────────────
    skip = p.add_argument_group("Skip Steps")
    skip.add_argument("--skip-fetch", action="store_true",
                      help="ข้าม fetch — ใช้ data/raw/sensor_data_raw.csv ที่มีอยู่")
    skip.add_argument("--skip-clean", action="store_true",
                      help="ข้าม clean — ใช้ data/processed/sensor_data_clean.csv ที่มีอยู่")
    skip.add_argument("--skip-split", action="store_true",
                      help="ข้าม split — ใช้ train/val/test.csv ที่มีอยู่")
    skip.add_argument("--skip-train", action="store_true",
                      help="ข้าม training — ใช้ models ที่มีอยู่")

    # ── Training ─────────────────────────────────────────────────────────────
    train = p.add_argument_group("Training")
    train.add_argument(
        "--target", choices=["temp", "hum", "both"], default="both",
        help="ตัวแปรที่จะเทรน",
    )
    train.add_argument(
        "--epochs", type=int, default=500,
        help="จำนวน epoch สูงสุด (early stopping จะหยุดก่อนถ้า val ไม่ improve)",
    )
    train.add_argument(
        "--train-ratio", type=float, default=0.70, metavar="RATIO",
        help="สัดส่วน train set (0–1)",
    )
    train.add_argument(
        "--val-ratio", type=float, default=0.15, metavar="RATIO",
        help="สัดส่วน val set (0–1)  test = 1 - train - val",
    )

    # ── RPi / Export ─────────────────────────────────────────────────────────
    raspi = p.add_argument_group("Raspberry Pi")
    raspi.add_argument(
        "--export-tflite", action="store_true",
        help="Export .keras → .tflite หลัง training (สำหรับ inference บน RPi)",
    )
    raspi.add_argument(
        "--check-raspi", action="store_true",
        help="ตรวจสอบ library compatibility สำหรับ RPi แล้วออก",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args)
