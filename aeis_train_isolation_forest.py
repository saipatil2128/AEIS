"""
=============================================================
  AEIS — Autonomous Edge Immune System
  Model Training  |  aeis_train_isolation_forest.py
  -----------------------------------------------------------
  Model   : Isolation Forest  (unsupervised anomaly detection)
  Purpose : Detects zero-day / novel attacks without labels.
  -----------------------------------------------------------
  Usage   : python aeis_train_isolation_forest.py
  Outputs : outputs_if/
              model_isolation_forest.pkl
              iso_threshold.npy
              cm_isolation_forest.png
              roc_isolation_forest.png
              drift_robustness_if.png
=============================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, recall_score
import joblib

from aeis_utils import (
    BASE_FEATURES,
    engineer_features,
    optimal_threshold,
    print_metrics,
    save_cm,
    save_roc,
)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DATA_DIR     = "."
OUTPUT_DIR   = "outputs_if"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  AEIS — ISOLATION FOREST TRAINING")
print("=" * 60)
print("\n[1] Loading data...")

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

X_train.columns = BASE_FEATURES
X_test.columns  = BASE_FEATURES

print(f"   X_train : {X_train.shape}  |  X_test : {X_test.shape}")
print(f"   Train   → Normal: {(y_train==0).sum()}  "
      f"Attack: {(y_train==1).sum()}  "
      f"({y_train.mean()*100:.1f}% attack)")
print(f"   Test    → Normal: {(y_test==0).sum()}   "
      f"Attack: {(y_test==1).sum()}   "
      f"({y_test.mean()*100:.1f}% attack)")

# ─────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n[2] Engineering features (4 → 11)...")

X_train_fe = engineer_features(X_train)
X_test_fe  = engineer_features(X_test)

print(f"   Features : {list(X_train_fe.columns)}")

X_tr = X_train_fe.values.astype(np.float64)
X_te = X_test_fe.values.astype(np.float64)
y_tr = y_train.values.astype(int)
y_te = y_test.values.astype(int)

# ─────────────────────────────────────────────────────────────
# 3.  TRAIN ISOLATION FOREST
# ─────────────────────────────────────────────────────────────
print("\n[3] Training Isolation Forest...")
print("""
   Design choices
   --------------
   contamination  : Derived from actual normal fraction in training
                    data (not a fixed guess). This directly sets the
                    anomaly score boundary.
   n_estimators   : 300 for score stability on small datasets.
   Training set   : Full training set (normal + attack), because
                    83% of samples are attacks — training on
                    normal-only would discard most data. The
                    contamination parameter compensates.
   Threshold      : Optimised via Precision-Recall curve on training
                    scores (no test leakage), targeting max recall
                    to minimise missed attacks.
""")

# Data-driven contamination
contamination = round(float((y_tr == 0).mean()), 4)
print(f"   Contamination (data-driven) : {contamination:.4f}  "
      f"= {(y_tr==0).sum()} normals / {len(y_tr)} total")

iso = IsolationForest(
    n_estimators=300,
    contamination=contamination,
    max_features=1.0,
    max_samples="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
iso.fit(X_tr)

# ─────────────────────────────────────────────────────────────
# 4.  THRESHOLD OPTIMISATION
# ─────────────────────────────────────────────────────────────
print("\n[4] Optimising anomaly threshold (target: max recall)...")

# Score on training set to find threshold — no test leakage
iso_scores_tr = -iso.score_samples(X_tr)   # higher = more anomalous
iso_scores_te = -iso.score_samples(X_te)

threshold = optimal_threshold(y_tr, iso_scores_tr, metric="recall")
print(f"   Optimised threshold : {threshold:.5f}")
print(f"   (v1 used fixed binary predict — no threshold control)")

iso_pred = (iso_scores_te >= threshold).astype(int)

# ─────────────────────────────────────────────────────────────
# 5.  EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n[5] Evaluating on test set...")
print("\n" + "-" * 50)
result = print_metrics("Isolation Forest", y_te, iso_pred, iso_scores_te)
print("-" * 50)

# ── Overfitting check ────────────────────────────────────────
iso_pred_tr  = (iso_scores_tr >= threshold).astype(int)
train_f1     = f1_score(y_tr, iso_pred_tr, zero_division=0)
test_f1      = result["f1"]
train_recall = recall_score(y_tr, iso_pred_tr, zero_division=0)
test_recall  = result["recall"]

print(f"\n[6] Overfitting check:")
print(f"   {'Metric':<12} {'Train':>8} {'Test':>8} {'Gap':>8}")
print(f"   {'-'*40}")
print(f"   {'F1':<12} {train_f1:>8.4f} {test_f1:>8.4f} "
      f"{abs(train_f1 - test_f1):>8.4f}")
print(f"   {'Recall':<12} {train_recall:>8.4f} {test_recall:>8.4f} "
      f"{abs(train_recall - test_recall):>8.4f}")
if abs(train_f1 - test_f1) < 0.05:
    print("   ✅ No overfitting detected (gap < 0.05).")
else:
    print("   ⚠️  Gap > 0.05 — model may be overfit. "
          "Try reducing n_estimators.")

# ─────────────────────────────────────────────────────────────
# 7.  DRIFT / ROBUSTNESS TEST
# ─────────────────────────────────────────────────────────────
print("\n[7] Robustness test — simulating feature drift / sensor noise...")

rng         = np.random.default_rng(RANDOM_STATE)
noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
drift_rows  = []

for sigma in noise_levels:
    noise   = rng.normal(0, sigma, X_te.shape)
    X_noisy = X_te + noise
    scores  = -iso.score_samples(X_noisy)
    pred    = (scores >= threshold).astype(int)
    drift_rows.append({
        "noise_sigma": sigma,
        "F1":     f1_score(y_te, pred, zero_division=0),
        "Recall": recall_score(y_te, pred, zero_division=0),
    })

drift_df = pd.DataFrame(drift_rows)
print(f"\n   {'Noise σ':<12} {'F1':>8} {'Recall':>10}")
print(f"   {'-'*32}")
for _, row in drift_df.iterrows():
    print(f"   {row['noise_sigma']:<12.2f} {row['F1']:>8.4f} "
          f"{row['Recall']:>10.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(drift_df["noise_sigma"], drift_df["F1"],
        "o-", color="#1565C0", label="F1")
ax.plot(drift_df["noise_sigma"], drift_df["Recall"],
        "s--", color="#C62828", label="Recall")
ax.set_xlabel("Noise σ  (feature drift intensity)")
ax.set_ylabel("Score")
ax.set_title("Isolation Forest — Drift Robustness")
ax.set_ylim(0, 1.05)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "drift_robustness_if.png"), dpi=120)
plt.close()

# ─────────────────────────────────────────────────────────────
# 8.  SAVE PLOTS
# ─────────────────────────────────────────────────────────────
print("\n[8] Saving plots...")

save_cm(y_te, iso_pred,
        title="Isolation Forest — Confusion Matrix",
        output_dir=OUTPUT_DIR,
        fname="cm_isolation_forest.png")

save_roc(y_te, iso_scores_te,
         title="Isolation Forest — ROC Curve",
         output_dir=OUTPUT_DIR,
         fname="roc_isolation_forest.png")

# ─────────────────────────────────────────────────────────────
# 9.  SAVE MODEL
# ─────────────────────────────────────────────────────────────
print("\n[9] Saving model artefacts...")

joblib.dump(iso, os.path.join(OUTPUT_DIR, "model_isolation_forest.pkl"))
np.save(os.path.join(OUTPUT_DIR, "iso_threshold.npy"), threshold)

print(f"   ✅ outputs_if/model_isolation_forest.pkl")
print(f"   ✅ outputs_if/iso_threshold.npy  (value={threshold:.5f})")
print(f"   ✅ outputs_if/cm_isolation_forest.png")
print(f"   ✅ outputs_if/roc_isolation_forest.png")
print(f"   ✅ outputs_if/drift_robustness_if.png")

# ─────────────────────────────────────────────────────────────
# 10. INFERENCE SNIPPET
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  INFERENCE USAGE")
print("=" * 60)
print("""
  import joblib, numpy as np
  from aeis_utils import engineer_features
  import pandas as pd

  iso       = joblib.load("outputs_if/model_isolation_forest.pkl")
  threshold = float(np.load("outputs_if/iso_threshold.npy"))

  def detect(packets_per_min, avg_packet_size,
             activity_hour, dest_count):
      raw  = pd.DataFrame([[packets_per_min, avg_packet_size,
                             activity_hour, dest_count]],
                           columns=["packets_per_min","avg_packet_size",
                                    "activity_hour","dest_count"])
      feat  = engineer_features(raw).values
      score = -iso.score_samples(feat)[0]   # higher = more anomalous
      return {
          "attack"    : bool(score >= threshold),
          "iso_score" : round(float(score), 5),
          "threshold" : round(threshold, 5),
      }
""")

print("=" * 60)
print("  ISOLATION FOREST TRAINING COMPLETE")
print("=" * 60)
