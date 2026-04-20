"""
generate_report_figures.py
รัน:  python generate_report_figures.py
สร้างรูป report-quality ทั้งหมดไว้ที่ reports/
"""
import sys, os, pickle, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Absolute paths ────────────────────────────────────────────────────────────
ROOT      = Path("/sessions/zealous-keen-johnson/mnt/Predict_temp_hum")
DATA_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUT_DIR   = ROOT / "reports"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# ── Style (report-quality) ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "font.family":         "DejaVu Sans",
    "font.size":           11,
    "axes.titlesize":      12,
    "axes.labelsize":      11,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.35,
    "legend.framealpha":   0.85,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.15,
})
COLORS = {
    "temp":    "#c0392b",
    "hum":     "#1a5276",
    "train":   "#27ae60",
    "val":     "#f39c12",
    "test":    "#e74c3c",
    "perfect": "#2c3e50",
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
def load_csv(name):
    df = pd.read_csv(DATA_DIR / name, index_col=0, parse_dates=True)
    return df[["temp","hum"]].dropna()

train_df = load_csv("train.csv")
val_df   = load_csv("val.csv")
test_df  = load_csv("test.csv")
print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

# ── Load model components ─────────────────────────────────────────────────────
print("Loading models...")
from src.trainnig.lstm_trainer import (
    add_time_features, make_sequences, compute_metrics,
    load_lstm_bundle, LOOKBACK_STEPS, N_HOURS_AHEAD, FEATURE_NAMES,
)

def evaluate(df, target):
    col_idx = 0 if target == "temp" else 1
    b = load_lstm_bundle(target, MODEL_DIR)
    raw = add_time_features(df)
    Xs  = b["scaler_X"].transform(raw)
    X, ys = make_sequences(Xs, col_idx)
    yp_s  = b["model"].predict(X, batch_size=256, verbose=0)
    sy = b["scaler_y"]
    yt = sy.inverse_transform(ys.reshape(-1,1)).reshape(ys.shape)
    yp = sy.inverse_transform(yp_s.reshape(-1,1)).reshape(yp_s.shape)
    return yt, yp

results = {}
for target in ["temp","hum"]:
    results[target] = {}
    for sname, df in [("train",train_df),("val",val_df),("test",test_df)]:
        yt, yp = evaluate(df, target)
        m = compute_metrics(yt.flatten(), yp.flatten())
        results[target][sname] = {"y_true":yt, "y_pred":yp, "metrics":m}
        print(f"  {target}/{sname}: RMSE={m['RMSE']:.3f}  R2={m['R2']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Overall Metrics Bar Chart (Train / Val / Test)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nFig 1: Overall metrics...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("BiLSTM Model Performance — Train / Val / Test Comparison",
             fontsize=14, fontweight="bold", y=1.01)

splits = ["train","val","test"]
x = np.arange(3)
w = 0.35
metrics_list = ["RMSE","MAE","MAPE","R2"]
ylabels = {"RMSE":"RMSE","MAE":"MAE","MAPE":"MAPE (%)","R2":"R²"}

for idx, metric in enumerate(metrics_list):
    ax = axes[idx//2][idx%2]
    vals_t = [results["temp"][s]["metrics"][metric] for s in splits]
    vals_h = [results["hum"][s]["metrics"][metric]  for s in splits]
    b1 = ax.bar(x-w/2, vals_t, w, color=COLORS["temp"], alpha=0.8, label="Temp (°C)", edgecolor="white")
    b2 = ax.bar(x+w/2, vals_h, w, color=COLORS["hum"],  alpha=0.8, label="Hum (%)",  edgecolor="white")
    for bar, v in [(b, v) for bars, vs in [(b1,vals_t),(b2,vals_h)] for bar, v in zip(bars,vs)]:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    if metric == "R2":
        ax.axhline(0, color="#e74c3c", ls="--", lw=1.2, alpha=0.7, label="R²=0")
        ax.axhline(0.7, color="#27ae60", ls=":", lw=1.2, alpha=0.7, label="R²=0.7 target")
    ax.set_xticks(x); ax.set_xticklabels(["Train","Val","Test"], fontsize=10)
    ax.set_title(ylabels[metric]); ax.set_ylabel(ylabels[metric])
    ax.legend(fontsize=9)

plt.tight_layout()
out = OUT_DIR / "fig1_overall_metrics.png"
plt.savefig(out); plt.close()
print(f"  Saved: {out.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Per-Horizon RMSE (h+1 → h+24)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 2: Per-horizon RMSE...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("RMSE per Forecast Horizon (Test Set)", fontsize=14, fontweight="bold")

h_axis = list(range(1, 25))
for ax, (target, color, unit) in zip(axes, [("temp",COLORS["temp"],"°C"),("hum",COLORS["hum"],"%")]):
    yt = results[target]["test"]["y_true"]
    yp = results[target]["test"]["y_pred"]
    rmse_h = [float(np.sqrt(np.mean((yt[:,h]-yp[:,h])**2))) for h in range(24)]
    mae_h  = [float(np.mean(np.abs(yt[:,h]-yp[:,h])))       for h in range(24)]

    ax.fill_between(h_axis, rmse_h, alpha=0.12, color=color)
    ax.plot(h_axis, rmse_h, lw=2.2, color=color, marker="o", ms=4, label="RMSE")
    ax.plot(h_axis, mae_h,  lw=1.8, color=color, marker="s", ms=3, ls="--", alpha=0.75, label="MAE")

    ax.axvline(6,  color="#95a5a6", ls=":", lw=1)
    ax.axvline(12, color="#95a5a6", ls=":", lw=1)
    ax.text(6.2,  ax.get_ylim()[1]*0.97, "+6h",  fontsize=8, color="#7f8c8d")
    ax.text(12.2, ax.get_ylim()[1]*0.97, "+12h", fontsize=8, color="#7f8c8d")

    ax.set_xlabel("Forecast Horizon (hours ahead)", fontsize=11)
    ax.set_ylabel(f"Error ({unit})", fontsize=11)
    ax.set_title(f"{'Temperature' if target=='temp' else 'Humidity'}")
    ax.set_xticks(range(1,25,2))
    ax.legend(fontsize=10)

plt.tight_layout()
out = OUT_DIR / "fig2_horizon_rmse.png"
plt.savefig(out); plt.close()
print(f"  Saved: {out.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Actual vs Predicted Time Series (Test set, h+1 & h+6)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 3: Actual vs Predicted...")
import matplotlib.dates as mdates

fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex="col")
fig.suptitle("Actual vs Predicted — Test Set", fontsize=14, fontweight="bold")

n_show = min(len(results["temp"]["test"]["y_true"]), 6*24*6)  # ~6 days of windows

for col, horizon in enumerate([0, 5]):   # h+1, h+6
    for row, (target, color, unit) in enumerate([("temp",COLORS["temp"],"°C"),
                                                   ("hum",COLORS["hum"],"%")]):
        ax = axes[row][col]
        yt = results[target]["test"]["y_true"][-n_show:, horizon]
        yp = results[target]["test"]["y_pred"][-n_show:, horizon]
        idx_arr = np.arange(len(yt))

        ax.fill_between(idx_arr, yt, alpha=0.15, color="#7f8c8d")
        ax.plot(idx_arr, yt, lw=1.5, color=COLORS["perfect"], label="Actual")
        ax.plot(idx_arr, yp, lw=1.2, color=color, alpha=0.9,
                label=f"Predicted +{horizon+1}h", ls="--")

        rmse = float(np.sqrt(np.mean((yt-yp)**2)))
        r2   = float(1 - np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2))
        ax.set_title(f"{'Temp' if target=='temp' else 'Hum'} | Horizon +{horizon+1}h  "
                     f"RMSE={rmse:.2f}{unit}  R²={r2:.3f}")
        ax.set_ylabel(unit, fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        if row == 1:
            ax.set_xlabel("Window index (test set)", fontsize=10)

plt.tight_layout()
out = OUT_DIR / "fig3_actual_vs_pred.png"
plt.savefig(out); plt.close()
print(f"  Saved: {out.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Scatter Plot + Residuals
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 4: Scatter + Residuals...")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Scatter Plot & Residual Analysis — Test Set", fontsize=14, fontweight="bold")

for row, (target, color, unit, label) in enumerate([
    ("temp", COLORS["temp"], "°C", "Temperature"),
    ("hum",  COLORS["hum"],  "%",  "Humidity"),
]):
    yt = results[target]["test"]["y_true"].flatten()
    yp = results[target]["test"]["y_pred"].flatten()
    res = yt - yp

    # Scatter
    ax = axes[row, 0]
    ax.scatter(yt, yp, alpha=0.05, s=3, color=color)
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo,hi],[lo,hi], lw=1.8, color=COLORS["perfect"], ls="--", label="Perfect (y=x)")
    sl, ic, r, _, _ = stats.linregress(yt, yp)
    xf = np.linspace(lo, hi, 200)
    ax.plot(xf, sl*xf+ic, lw=2, color="#e74c3c", label=f"Fit  r={r:.3f}")
    ax.set_xlabel(f"Actual ({unit})"); ax.set_ylabel(f"Predicted ({unit})")
    ax.set_title(f"{label} — Scatter")
    ax.legend(fontsize=9)
    ax.text(0.05, 0.90, f"R²={r**2:.3f}", transform=ax.transAxes,
            fontsize=10, bbox=dict(boxstyle="round", facecolor="#ffeaa7", alpha=0.85))

    # Residuals vs Predicted
    ax = axes[row, 1]
    ax.scatter(yp, res, alpha=0.05, s=3, color=color)
    ax.axhline(0, lw=1.8, color=COLORS["perfect"], ls="--")
    ax.axhline( res.std(), color="#e67e22", lw=1, ls=":", label="+1 SD")
    ax.axhline(-res.std(), color="#e67e22", lw=1, ls=":", label="-1 SD")
    ax.set_xlabel(f"Predicted ({unit})"); ax.set_ylabel(f"Residual ({unit})")
    ax.set_title(f"{label} — Residuals vs Predicted")
    ax.legend(fontsize=9)

    # Residuals Histogram
    ax = axes[row, 2]
    ax.hist(res, bins=80, color=color, alpha=0.72, edgecolor="white", lw=0.3)
    ax.axvline(0, lw=2, color=COLORS["perfect"], ls="--")
    ax.axvline(res.mean(), lw=1.8, color="#e74c3c",
               label=f"mean={res.mean():.3f}")
    # Normal fit overlay
    mu, sigma = res.mean(), res.std()
    xn = np.linspace(res.min(), res.max(), 300)
    ax2 = ax.twinx()
    ax2.plot(xn, stats.norm.pdf(xn, mu, sigma), lw=2, color=COLORS["perfect"], alpha=0.7, label="Normal fit")
    ax2.set_ylabel("Density", fontsize=9); ax2.legend(fontsize=8, loc="upper left")
    ax.set_xlabel(f"Residual ({unit})"); ax.set_ylabel("Count")
    ax.set_title(f"{label} — Residual Distribution\nμ={mu:.3f}  σ={sigma:.3f}")
    ax.legend(fontsize=9)

plt.tight_layout()
out = OUT_DIR / "fig4_scatter_residuals.png"
plt.savefig(out); plt.close()
print(f"  Saved: {out.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Error Heatmap (Hour-of-Day × Horizon)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 5: Error heatmap...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Mean Absolute Error — Hour of Prediction × Forecast Horizon",
             fontsize=14, fontweight="bold")

n_win = results["temp"]["test"]["y_true"].shape[0]
pred_hours = test_df.index[LOOKBACK_STEPS: LOOKBACK_STEPS + n_win].hour

for ax, (target, cmap, unit) in zip(axes, [("temp","YlOrRd","°C"),("hum","Blues","%")]):
    yt = results[target]["test"]["y_true"]
    yp = results[target]["test"]["y_pred"]
    ae = np.abs(yt - yp)

    heatmap = np.full((24, 24), np.nan)
    for h in range(24):
        mask = (pred_hours == h)
        if mask.sum() > 0:
            heatmap[h, :] = ae[mask].mean(axis=0)

    im = ax.imshow(heatmap, aspect="auto", cmap=cmap, origin="lower")
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(f"MAE ({unit})", fontsize=10)

    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"+{h+1}h" for h in range(0,24,2)], fontsize=8, rotation=45)
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0,24,2)], fontsize=8)
    ax.set_xlabel("Forecast Horizon (hours ahead)", fontsize=11)
    ax.set_ylabel("Hour of Prediction", fontsize=11)
    ax.set_title(f"{'Temperature' if target=='temp' else 'Humidity'} MAE ({unit})")

plt.tight_layout()
out = OUT_DIR / "fig5_error_heatmap.png"
plt.savefig(out); plt.close()
print(f"  Saved: {out.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Summary Dashboard (Metrics Table + R² per horizon)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 6: Summary dashboard...")
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
fig.suptitle("BiLSTM Performance Summary Dashboard", fontsize=15, fontweight="bold")

# (0,0)-(0,1) Metrics table
ax_tab = fig.add_subplot(gs[0, :2])
ax_tab.axis("off")
col_labels = ["Target","Split","RMSE","MAE","MAPE (%)","R²"]
rows_data  = []
for target in ["temp","hum"]:
    unit = "°C" if target=="temp" else "%"
    for split in ["train","val","test"]:
        m = results[target][split]["metrics"]
        rows_data.append([
            f"{target} ({unit})", split,
            f"{m['RMSE']:.3f}", f"{m['MAE']:.3f}",
            f"{m['MAPE']:.2f}", f"{m['R2']:.4f}"
        ])

row_colors = []
split_colors = {"train":"#d5f5e3","val":"#fdebd0","test":"#fadbd8"}
for r in rows_data:
    row_colors.append([split_colors[r[1]]] * len(col_labels))

tbl = ax_tab.table(cellText=rows_data, colLabels=col_labels,
                   cellLoc="center", loc="center",
                   cellColours=row_colors)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.7)
for (r,c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
ax_tab.set_title("Metrics Summary", fontsize=11, fontweight="bold", pad=10)

# (0,2) MAPE bar
ax_mape = fig.add_subplot(gs[0, 2])
mape_t = [results["temp"][s]["metrics"]["MAPE"] for s in ["train","val","test"]]
mape_h = [results["hum"][s]["metrics"]["MAPE"]  for s in ["train","val","test"]]
x = np.arange(3)
ax_mape.bar(x-0.2, mape_t, 0.38, color=COLORS["temp"], alpha=0.82, label="Temp")
ax_mape.bar(x+0.2, mape_h, 0.38, color=COLORS["hum"],  alpha=0.82, label="Hum")
ax_mape.set_xticks(x); ax_mape.set_xticklabels(["Train","Val","Test"])
ax_mape.set_ylabel("MAPE (%)"); ax_mape.set_title("MAPE by Split")
ax_mape.legend(fontsize=9)

# (1,0) Per-horizon RMSE
ax_hz = fig.add_subplot(gs[1, 0])
h_axis = list(range(1, 25))
for target, color, unit in [("temp",COLORS["temp"],"°C"),("hum",COLORS["hum"],"%")]:
    yt = results[target]["test"]["y_true"]
    yp = results[target]["test"]["y_pred"]
    rmse_h = [float(np.sqrt(np.mean((yt[:,h]-yp[:,h])**2))) for h in range(24)]
    ax_hz.plot(h_axis, rmse_h, lw=2, color=color, marker="o", ms=3,
               label=f"{target} ({unit})")
ax_hz.set_xlabel("Horizon (h ahead)"); ax_hz.set_ylabel("RMSE")
ax_hz.set_title("RMSE per Horizon — Test"); ax_hz.legend(fontsize=9)
ax_hz.set_xticks(range(1,25,3))

# (1,1) R² per horizon
ax_r2 = fig.add_subplot(gs[1, 1])
for target, color in [("temp",COLORS["temp"]),("hum",COLORS["hum"])]:
    yt = results[target]["test"]["y_true"]
    yp = results[target]["test"]["y_pred"]
    r2_h = [float(1-np.sum((yt[:,h]-yp[:,h])**2)/np.sum((yt[:,h]-yt[:,h].mean())**2))
            for h in range(24)]
    ax_r2.plot(h_axis, r2_h, lw=2, color=color, marker="o", ms=3, label=target)
ax_r2.axhline(0,   color="#e74c3c", ls="--", lw=1.2, alpha=0.7, label="R²=0")
ax_r2.axhline(0.7, color="#27ae60", ls=":",  lw=1.2, alpha=0.7, label="R²=0.7")
ax_r2.set_xlabel("Horizon (h ahead)"); ax_r2.set_ylabel("R²")
ax_r2.set_title("R² per Horizon — Test"); ax_r2.legend(fontsize=9)
ax_r2.set_xticks(range(1,25,3))

# (1,2) Train/Val/Test size bar
ax_sz = fig.add_subplot(gs[1, 2])
sizes = [len(train_df), len(val_df), len(test_df)]
pcts  = [s/sum(sizes)*100 for s in sizes]
bars  = ax_sz.bar(["Train","Val","Test"], sizes,
                   color=[COLORS["train"],COLORS["val"],COLORS["test"]],
                   alpha=0.85, edgecolor="white")
for bar, sz, pct in zip(bars, sizes, pcts):
    ax_sz.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
               f"{sz:,}\n({pct:.0f}%)", ha="center", fontsize=9, fontweight="bold")
ax_sz.set_ylabel("Number of Samples")
ax_sz.set_title("Dataset Split")

out = OUT_DIR / "fig6_summary_dashboard.png"
plt.savefig(out); plt.close()
print(f"  Saved: {out.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Training Loss History
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 7: Training history...")
hist_files = [MODEL_DIR/"lstm_history_temp.png", MODEL_DIR/"lstm_history_hum.png"]
if all(p.exists() for p in hist_files):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Loss History", fontsize=14, fontweight="bold")
    titles = ["Temperature Model", "Humidity Model"]
    for ax, p, title in zip(axes, hist_files, titles):
        img = plt.imread(str(p))
        ax.imshow(img); ax.axis("off"); ax.set_title(title, fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / "fig7_training_history.png"
    plt.savefig(out); plt.close()
    print(f"  Saved: {out.name}")
else:
    print("  Skipped (no history images)")

print("\n✅  All figures saved to:", OUT_DIR)
print("Files:")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
