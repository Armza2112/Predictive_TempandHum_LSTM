
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import json

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

# import pipeline modules
from src.processing.data_cleaning import (
    remove_near_duplicates,
    convert_timezone,
    resample_regular,
    interpolate_gaps,
    handle_outliers,
)
# LSTM inference helper
from src.trainnig.lstm_trainer import predict_lstm

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
TB_HOST      = os.getenv("TB_MQTT_HOST",     "thingsboard.sitearound.com")
TB_PORT      = int(os.getenv("TB_MQTT_PORT", "1889"))
TB_USERNAME  = os.getenv("TB_MQTT_USERNAME", "z0cidvhmfrv6qm43h3h4")
TB_TOPIC     = os.getenv("TB_MQTT_TOPIC",    "v1/devices/me/telemetry")

MODEL_DIR    = Path("model")
FETCH_HOURS  = 36      # ดึงย้อนหลัง 36h (lag_144 ต้องการ 24h + buffer สำหรับ rolling)
INTERVAL_MIN = 10      # รอบการส่งข้อมูล (นาที) เมื่อใช้ --loop
TIMEOUT_SEC  = 10      # MQTT connect timeout (วินาที)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  1. ดึงข้อมูลจาก InfluxDB
# ══════════════════════════════════════════════════════════════════════════════
def fetch_latest(hours: int = FETCH_HOURS):
    """
    ดึงข้อมูล sensor ล่าสุดย้อนหลัง `hours` ชั่วโมงจาก InfluxDB
    คืน DataFrame ที่มีคอลัมน์ _time, hum, temp
    """
    from src.database.influx import DatabaseHandler

    log.info("📡  Fetching latest %d hours from InfluxDB...", hours)
    db = DatabaseHandler()
    try:
        df = db.fetch_data(measurement="EM-300", range_start=f"-{hours}h")
    finally:
        db.close()

    if df.empty:
        raise RuntimeError("InfluxDB ไม่มีข้อมูลในช่วงเวลาที่ระบุ")

    log.info("    Got %d rows  (%s → %s)",
             len(df),
             df["_time"].min().strftime("%Y-%m-%d %H:%M"),
             df["_time"].max().strftime("%Y-%m-%d %H:%M"))
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  2. Clean ข้อมูล (ใช้ functions จาก data_cleaning.py)
# ══════════════════════════════════════════════════════════════════════════════
def clean_latest(df):
    """
    ทำความสะอาดข้อมูลที่ดึงมาใหม่
    ใช้ขั้นตอนเดียวกับ data_cleaning.py เพื่อให้ consistent
    """
    log.info("🧹  Cleaning data...")
    df = remove_near_duplicates(df)
    df = convert_timezone(df)
    df = resample_regular(df)
    df = interpolate_gaps(df)
    df = handle_outliers(df, clip=True)

    # drop แถว NaN ที่เหลือ (gap ใหญ่เกิน 6 ชม.)
    before = len(df)
    df = df.dropna()
    if len(df) < before:
        log.info("    Dropped %d NaN rows after cleaning", before - len(df))

    log.info("    Clean rows: %d", len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  3. Predict 24 ชั่วโมงล่วงหน้า (LSTM Multi-Output)
# ══════════════════════════════════════════════════════════════════════════════
def predict_24h(df, model_dir: Path = MODEL_DIR) -> "pd.DataFrame":
    """
    ทำนาย 24h ล่วงหน้าด้วย Bidirectional LSTM
    คืน DataFrame index=timestamp, คอลัมน์ temp_pred, hum_pred
    """
    return predict_lstm(df, model_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  4. ส่งไปยัง ThingsBoard ผ่าน MQTT
# ══════════════════════════════════════════════════════════════════════════════
def _build_history_batch(df_clean: "pd.DataFrame", hours: int = 24) -> list:
    """
    แปลงข้อมูลจริงย้อนหลัง `hours` ชั่วโมงเป็น batch entries
    Resample เป็น 1 ชั่วโมง → 24 จุด  keys: temp_actual, hum_actual
    """
    import pandas as pd

    import time as _time
    now_epoch_ms = int(_time.time() * 1000)          # UTC epoch ms (timezone-safe)
    cutoff       = now_epoch_ms - hours * 3600 * 1000

    # ตัดย้อนหลัง 24h จาก clean data แล้ว resample รายชั่วโมง
    cutoff_ts = pd.Timestamp(cutoff / 1000, unit="s")
    df_24     = df_clean[df_clean.index > cutoff_ts].copy()

    df_hourly = df_24.resample("1h").mean().dropna()

    # กรอง ts ที่เกินเวลาปัจจุบัน (epoch ms) ออก
    # ใช้ epoch ms โดยตรงเพื่อหลีกเลี่ยงปัญหา timezone ทุกรูปแบบ
    # tz_localize → timestamp() ได้ UTC epoch ที่ถูกต้องโดยไม่ขึ้นกับ system timezone
    def bkk_to_epoch_ms(t):
        return int(t.tz_localize('Asia/Bangkok').timestamp() * 1000)

    batch = [
        {
            "ts":     bkk_to_epoch_ms(ts),
            "values": {
                "temp_actual": round(float(row["temp"]), 2),
                "hum_actual":  round(float(row["hum"]),  2),
            }
        }
        for ts, row in df_hourly.iterrows()
        if bkk_to_epoch_ms(ts) <= now_epoch_ms   # ← ตัด future ออก
    ]
    return batch


def publish_telemetry(
    forecast_df: "pd.DataFrame",
    history_df:  "pd.DataFrame | None" = None,   # ข้อมูลจริงย้อนหลัง 24h
    host: str     = TB_HOST,
    port: int     = TB_PORT,
    username: str = TB_USERNAME,
    topic: str    = TB_TOPIC,
    dry_run: bool = False,
) -> bool:
    """
    Publish ไปยัง ThingsBoard ผ่าน MQTT — batch เดียวรวม:
      1. ค่าจริงย้อนหลัง 24h  → temp_actual, hum_actual  (รายชั่วโมง ~24 จุด)
      2. ค่า predict 24h       → temp_pred,   hum_pred    (รายชั่วโมง 24 จุด)

    ThingsBoard batch format:
      [{"ts": <epoch_ms>, "values": {...}}, ...]  เรียงตาม timestamp

    Parameters
    ----------
    forecast_df : DataFrame index=timestamp อนาคต, คอลัมน์ temp_pred, hum_pred
    history_df  : DataFrame clean ย้อนหลัง (10min interval) — ถ้า None ส่งแค่ forecast
    dry_run     : log แต่ไม่ส่งจริง

    Returns
    -------
    bool  True = สำเร็จ
    """
    # ── ชุดที่ 1: ค่าจริงย้อนหลัง 24h ──────────────────────────────────────
    history_batch: list = []
    if history_df is not None:
        history_batch = _build_history_batch(history_df, hours=24)
        log.info("    History : %d hourly actual points", len(history_batch))

    # ── ชุดที่ 2: ค่า predict ล่วงหน้า 24h ──────────────────────────────────
    # tz_localize → UTC epoch ที่ถูกต้องโดยไม่ขึ้นกับ system timezone ของ container
    forecast_batch = [
        {
            "ts":     int(ts.tz_localize('Asia/Bangkok').timestamp() * 1000),
            "values": {"temp_predict": row["temp_pred"], "hum_predict": row["hum_pred"]}
        }
        for ts, row in forecast_df.iterrows()
    ]


    # ── รวม + เรียงตาม timestamp ─────────────────────────────────────────────
    batch   = sorted(history_batch + forecast_batch, key=lambda x: x["ts"])
    message = json.dumps(batch)

    log.info("📤  Publishing to ThingsBoard MQTT [%s:%d] topic=%s", host, port, topic)
    log.info("    Batch: %d entries total (%d actual + %d forecast)  (%d bytes)",
             len(batch), len(history_batch), len(forecast_batch), len(message))

    if dry_run:
        log.info("    [dry-run] ไม่ส่งจริง — payload preview:")
        for entry in batch:
            import datetime
            ts_str = datetime.datetime.fromtimestamp(entry["ts"] / 1000).strftime("%Y-%m-%d %H:%M")
            vals   = "  ".join(f"{k}={v}" for k, v in entry["values"].items())
            log.info("      %s  |  %s", ts_str, vals)
        return True

    published = {"done": False, "rc": None}

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            log.info("    🔗 MQTT connected")
            result = client.publish(topic, message, qos=1)
            published["rc"] = result.rc
        else:
            log.error("    ❌ MQTT connect failed (rc=%d)", rc)

    def on_publish(client, userdata, mid):
        log.info("    ✅ ThingsBoard MQTT OK (mid=%d)", mid)
        published["done"] = True
        client.disconnect()

    try:
        client = mqtt.Client()
        client.username_pw_set(username, password="")
        client.on_connect = on_connect
        client.on_publish = on_publish

        client.connect(host, port, keepalive=60)
        client.loop_start()

        # รอ publish สำเร็จหรือ timeout
        deadline = time.time() + TIMEOUT_SEC
        while not published["done"] and time.time() < deadline:
            time.sleep(0.1)

        client.loop_stop()

        if published["done"]:
            return True
        else:
            log.error("    ❌ MQTT publish timeout (%ds)", TIMEOUT_SEC)
            return False

    except Exception as e:
        log.error("    ❌ MQTT error: %s", e)
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  Main: รัน 1 รอบ
# ══════════════════════════════════════════════════════════════════════════════
def run_once(dry_run: bool = False) -> bool:
    """
    รัน pipeline 1 รอบ: fetch → clean → predict → post

    Returns
    -------
    bool  True = สำเร็จทั้งหมด
    """
    import time as _time
    import datetime as _dt

    log.info("═" * 60)
    log.info("🚀  run_once() started")
    log.info("    system time (UTC)     : %s",
             _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("    system time (Bangkok) : %s",
             (_dt.datetime.utcnow() + _dt.timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S"))
    log.info("─" * 60)

    try:
        # ── STAGE 1: Fetch ───────────────────────────────────────────
        df_raw      = fetch_latest(hours=FETCH_HOURS)
        log.info("    [fetch] raw rows  : %d", len(df_raw))
        log.info("    [fetch] _time min : %s", df_raw["_time"].min())
        log.info("    [fetch] _time max : %s", df_raw["_time"].max())

        # ── STAGE 2: Clean ───────────────────────────────────────────
        df_clean    = clean_latest(df_raw)
        log.info("    [clean] rows      : %d", len(df_clean))
        log.info("    [clean] index min : %s", df_clean.index.min().strftime("%Y-%m-%d %H:%M"))
        log.info("    [clean] index max : %s", df_clean.index.max().strftime("%Y-%m-%d %H:%M"))

        # ── STAGE 3: Predict ─────────────────────────────────────────
        forecast_df = predict_24h(df_clean)
        log.info("    [predict] first ts: %s  temp=%.2f  hum=%.2f",
                 forecast_df.index[0].strftime("%Y-%m-%d %H:%M"),
                 forecast_df["temp_pred"].iloc[0],
                 forecast_df["hum_pred"].iloc[0])
        log.info("    [predict] last ts : %s  temp=%.2f  hum=%.2f",
                 forecast_df.index[-1].strftime("%Y-%m-%d %H:%M"),
                 forecast_df["temp_pred"].iloc[-1],
                 forecast_df["hum_pred"].iloc[-1])
        success     = publish_telemetry(forecast_df, history_df=df_clean, dry_run=dry_run)
        return success

    except FileNotFoundError as e:
        log.error("❌  %s", e)
        return False
    except RuntimeError as e:
        log.error("❌  %s", e)
        return False
    except Exception as e:
        log.exception("❌  Unexpected error: %s", e)
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  Loop mode: รันซ้ำทุก INTERVAL_MIN นาที
# ══════════════════════════════════════════════════════════════════════════════
def run_loop(interval_min: int = INTERVAL_MIN, dry_run: bool = False) -> None:
    """รัน pipeline วนซ้ำทุก interval_min นาทีจนกว่าจะ Ctrl+C"""
    log.info("🔁  Loop mode: ส่งข้อมูลทุก %d นาที  (Ctrl+C เพื่อหยุด)", interval_min)
    interval_sec = interval_min * 60

    while True:
        run_once(dry_run=dry_run)
        log.info("⏳  รอ %d นาที ก่อนรอบถัดไป...", interval_min)
        time.sleep(interval_sec)


# ══════════════════════════════════════════════════════════════════════════════
#  Schedule mode: รันทุกวันที่เวลาที่กำหนด (default เที่ยงคืน)
# ══════════════════════════════════════════════════════════════════════════════
def _seconds_until(target_hour: int, target_minute: int) -> float:
    """คำนวณจำนวนวินาทีที่ต้องรอจนถึงเวลา target_hour:target_minute รอบถัดไป"""
    from datetime import datetime, timedelta

    now  = datetime.now()
    next_run = now.replace(hour=target_hour, minute=target_minute,
                           second=0, microsecond=0)
    if next_run <= now:
        next_run += timedelta(days=1)

    wait = (next_run - now).total_seconds()
    return wait, next_run


def run_schedule(
    at_hour: int   = 0,
    at_minute: int = 0,
    dry_run: bool  = False,
) -> None:
    """
    รัน pipeline ทุกวันที่เวลา at_hour:at_minute (24h format)
    ค้างโปรแกรมไว้จนกว่าจะ Ctrl+C

    Parameters
    ----------
    at_hour   : ชั่วโมง 0–23  (default 0 = เที่ยงคืน)
    at_minute : นาที 0–59     (default 0)
    """
    log.info("🕛  Schedule mode: รันทุกวันเวลา %02d:%02d  (Ctrl+C เพื่อหยุด)",
             at_hour, at_minute)

    while True:
        wait_sec, next_run = _seconds_until(at_hour, at_minute)
        log.info("⏳  รอบถัดไป: %s  (อีก %.0f นาที)",
                 next_run.strftime("%Y-%m-%d %H:%M"), wait_sec / 60)
        time.sleep(wait_sec)

        log.info("🔔  ถึงเวลาแล้ว! เริ่มรัน pipeline...")
        run_once(dry_run=dry_run)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prediction Service — ส่ง predicted temp & hum ไปยัง ThingsBoard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--loop",  action="store_true",
                      help="รันวนซ้ำทุก --interval นาที")
    mode.add_argument("--once",  action="store_true",
                      help="รัน 1 ครั้งแล้วออก")

    parser.add_argument("--at",       type=str, default="00:00",
                        metavar="HH:MM",
                        help="เวลาที่รัน ใช้กับ --schedule  เช่น 00:00, 06:30")
    parser.add_argument("--interval", type=int, default=INTERVAL_MIN,
                        help="ช่วงเวลาระหว่างแต่ละรอบ (นาที) ใช้กับ --loop")
    parser.add_argument("--dry-run",  action="store_true",
                        help="ทำนายแต่ไม่ส่งไปยัง ThingsBoard จริง")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                        help="โฟลเดอร์ที่เก็บ .pkl")
    parser.add_argument("--hours",    type=int, default=FETCH_HOURS,
                        help="ดึงข้อมูลย้อนหลัง N ชั่วโมง")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # parse --at HH:MM
    try:
        at_h, at_m = [int(x) for x in args.at.split(":")]
    except ValueError:
        print(f"❌  --at ต้องเป็นรูปแบบ HH:MM  เช่น 00:00 หรือ 06:30")
        exit(1)

    # default: schedule mode (ค้างรอทุกเที่ยงคืน)
    use_loop     = args.loop
    use_once     = args.once
    use_schedule = not use_loop and not use_once   # default

    log.info("=" * 52)
    log.info("  PREDICTION SERVICE — Sensor EM-300")
    log.info("  ThingsBoard : mqtt://%s:%d  (topic: %s)", TB_HOST, TB_PORT, TB_TOPIC)
    log.info("  Mode        : %s",
             f"schedule (ทุกวัน {args.at})" if use_schedule
             else f"loop (ทุก {args.interval} นาที)" if use_loop
             else "one-shot")
    log.info("  Dry-run     : %s", args.dry_run)
    log.info("=" * 52)

    if use_loop:
        run_loop(interval_min=args.interval, dry_run=args.dry_run)
    elif use_once:
        success = run_once(dry_run=args.dry_run)
        exit(0 if success else 1)
    else:
        run_schedule(at_hour=at_h, at_minute=at_m, dry_run=args.dry_run)
