"""
Experiment 3 – Statistical Significance Tests
EC2 (g5.xlarge) vs SageMaker (ml.g5.xlarge) – train_time_sec per model

Test: Welch's independent-samples t-test (equal_var=False, two-tailed)
Significance threshold α = 0.05
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[2]
EC2_FILE = ROOT / "Output" / "Experiment3_EC2.json"
SM_FILE  = ROOT / "Output" / "Experiment3_SM.json"
OUT_DIR  = ROOT / "outputs" / "experiment3"
OUT_FILE = OUT_DIR / "significance_tests.json"

ALPHA   = 0.05
MODELS  = ["mobilenetv3", "resnet50", "resnet101"]
LABELS  = {"mobilenetv3": "MobileNetV3", "resnet50": "ResNet50", "resnet101": "ResNet101"}

# ---------------------------------------------------------------------------
# Load & group by model
# ---------------------------------------------------------------------------
with open(EC2_FILE) as f:
    ec2_all = json.load(f)
with open(SM_FILE) as f:
    sm_all = json.load(f)

def by_model(runs: list[dict]) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = {}
    for r in runs:
        groups.setdefault(r["model_name"], []).append(r["train_time_sec"])
    return groups

ec2 = by_model(ec2_all)
sm  = by_model(sm_all)

# ---------------------------------------------------------------------------
# Welch–Satterthwaite degrees of freedom
# ---------------------------------------------------------------------------
def welch_df(a: list[float], b: list[float]) -> float:
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    n1, n2 = len(a), len(b)
    num   = (s1**2/n1 + s2**2/n2)**2
    denom = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
    return num / denom if denom > 0 else float("nan")

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
results: dict = {
    "metadata": {
        "experiment": "Experiment 3 – multi-model training time comparison",
        "ec2_instance": "g5.xlarge",
        "sm_instance":  "ml.g5.xlarge",
        "metric":       "train_time_sec",
        "test":         "Welch's independent-samples t-test (two-tailed, equal_var=False)",
        "alpha":        ALPHA,
        "data_sources": {
            "ec2": str(EC2_FILE.relative_to(ROOT)),
            "sm":  str(SM_FILE.relative_to(ROOT)),
        },
    },
    "models": {},
}

for model in MODELS:
    ec2_vals = ec2[model]
    sm_vals  = sm[model]

    ec2_mean = float(np.mean(ec2_vals))
    ec2_std  = float(np.std(ec2_vals, ddof=1))
    sm_mean  = float(np.mean(sm_vals))
    sm_std   = float(np.std(sm_vals, ddof=1))

    abs_diff_sec = ec2_mean - sm_mean
    rel_diff_pct = (abs_diff_sec / sm_mean) * 100

    t_stat, p_val = stats.ttest_ind(ec2_vals, sm_vals, equal_var=False, alternative="two-sided")
    df = welch_df(ec2_vals, sm_vals)
    significant = bool(p_val < ALPHA)

    results["models"][model] = {
        "label": LABELS[model],
        "ec2": {
            "n": len(ec2_vals),
            "values_sec": [round(v, 4) for v in ec2_vals],
            "mean_sec":   round(ec2_mean, 4),
            "std_sec":    round(ec2_std, 4),
        },
        "sagemaker": {
            "n": len(sm_vals),
            "values_sec": [round(v, 4) for v in sm_vals],
            "mean_sec":   round(sm_mean, 4),
            "std_sec":    round(sm_std, 4),
        },
        "comparison": {
            "abs_diff_sec": round(abs_diff_sec, 4),
            "rel_diff_pct": round(rel_diff_pct, 4),
        },
        "welch_t_test": {
            "t_statistic":        round(float(t_stat), 6),
            "p_value":            round(float(p_val), 8),
            "degrees_of_freedom": round(float(df), 4),
            "significant":        significant,
        },
    }

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=4)

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
SEP  = "=" * 100
sep2 = "-" * 100

print()
print(SEP)
print("  EXPERIMENT 3 – Training Time: EC2 (g5.xlarge) vs SageMaker (ml.g5.xlarge)")
print(f"  Metric: train_time_sec  |  Test: Welch's t-test (two-tailed)  |  α = {ALPHA}")
print(SEP)
print(f"  {'Model':<14}  {'EC2 mean±std (s)':>22}  {'SM mean±std (s)':>22}  "
      f"{'Δ (s)':>9}  {'Δ (%)':>8}  {'t':>8}  {'p-value':>10}  {'Significant'}")
print(sep2)

for model in MODELS:
    r   = results["models"][model]
    lbl = r["label"]
    e   = r["ec2"]
    s   = r["sagemaker"]
    cmp = r["comparison"]
    wt  = r["welch_t_test"]

    ec2_str = f"{e['mean_sec']:.1f} ± {e['std_sec']:.1f}"
    sm_str  = f"{s['mean_sec']:.1f} ± {s['std_sec']:.1f}"
    sig_str = "YES  ***" if wt["significant"] else "no"

    print(f"  {lbl:<14}  {ec2_str:>22}  {sm_str:>22}  "
          f"{cmp['abs_diff_sec']:>+9.1f}  {cmp['rel_diff_pct']:>+8.2f}  "
          f"{wt['t_statistic']:>+8.3f}  {wt['p_value']:>10.6f}  {sig_str}")

print(sep2)
print()
print("  Δ = EC2_mean − SM_mean  (negative → EC2 is faster)")
print(f"  Results saved to: {OUT_FILE.relative_to(ROOT)}")
print(SEP)
print()
