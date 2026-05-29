"""
Experiment 2 – Statistical Significance Tests
EC2 (g5.xlarge) vs SageMaker (ml.g5.xlarge) – ResNet50 per-epoch component timing

Tests per component:
  • Welch's independent-samples t-test  (equal_var=False, two-tailed)
  • Mann-Whitney U test                  (alternative='two-sided')

Additional metrics:
  • Mean ± std for each platform (seconds)
  • Absolute difference (ms)
  • Relative difference (%)
  • Share of total epoch time (abs_diff_sec / EC2_wall_clock_mean * 100)
  • Cohen's d effect size

Significance threshold α = 0.05
Practical significance flag: abs diff > 1 % of EC2 epoch wall-clock mean
"""

import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
EC2_FILE = ROOT / "Output" / "Experiment2-EC2.json"
SM_FILE  = ROOT / "Output" / "Experiment2_SM.json"
OUT_DIR  = ROOT / "outputs" / "experiment2"
OUT_FILE = OUT_DIR / "significance_tests.json"

ALPHA = 0.05

COMPONENTS = [
    "avg_data_loading_sec",
    "avg_forward_pass_sec",
    "avg_backward_pass_sec",
    "avg_optimizer_step_sec",
    "avg_platform_overhead_sec",
    "avg_epoch_wall_clock_sec",
]

COMPONENT_LABELS = {
    "avg_data_loading_sec":      "Data Loading",
    "avg_forward_pass_sec":      "Forward Pass",
    "avg_backward_pass_sec":     "Backward Pass",
    "avg_optimizer_step_sec":    "Optimizer Step",
    "avg_platform_overhead_sec": "Platform Overhead",
    "avg_epoch_wall_clock_sec":  "Epoch Wall-Clock",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(EC2_FILE) as f:
    ec2_runs = json.load(f)                        # already all ResNet50

with open(SM_FILE) as f:
    sm_runs = json.load(f)

assert len(ec2_runs) == 5, f"Expected 5 EC2 runs, got {len(ec2_runs)}"
assert len(sm_runs)  == 5, f"Expected 5 SM runs,  got {len(sm_runs)}"

# ---------------------------------------------------------------------------
# Helper: Cohen's d (pooled-std version, appropriate for unequal n)
# ---------------------------------------------------------------------------
def cohens_d(a: list[float], b: list[float]) -> float:
    na, nb = len(a), len(b)
    pooled_std = math.sqrt(
        ((na - 1) * np.std(a, ddof=1) ** 2 + (nb - 1) * np.std(b, ddof=1) ** 2)
        / (na + nb - 2)
    )
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

# ---------------------------------------------------------------------------
# Reference: EC2 epoch wall-clock mean (for practical-significance threshold)
# ---------------------------------------------------------------------------
ec2_wall_clock = [r["avg_epoch_wall_clock_sec"] for r in ec2_runs]
ec2_wall_clock_mean = float(np.mean(ec2_wall_clock))
practical_threshold_sec = 0.01 * ec2_wall_clock_mean   # 1 % of epoch time

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
results: dict = {
    "metadata": {
        "experiment": "Experiment 2 – ResNet50 training overhead",
        "ec2_instance":       "g5.xlarge",
        "sm_instance":        "ml.g5.xlarge",
        "ec2_n_runs":         len(ec2_runs),
        "sm_n_runs":          len(sm_runs),
        "alpha":              ALPHA,
        "practical_significance_threshold_pct_epoch": 1.0,
        "practical_significance_threshold_sec": round(practical_threshold_sec, 6),
        "ec2_wall_clock_mean_sec": round(ec2_wall_clock_mean, 6),
        "data_sources": {
            "ec2": str(EC2_FILE.relative_to(ROOT)),
            "sm":  str(SM_FILE.relative_to(ROOT)),
        },
        "notes": (
            "Both platforms use identical hardware (g5.xlarge GPU), batch size 4, 70 epochs."
        ),
    },
    "components": {},
}

for comp in COMPONENTS:
    ec2_vals = [r[comp] for r in ec2_runs]
    sm_vals  = [r[comp] for r in sm_runs]

    ec2_mean = float(np.mean(ec2_vals))
    ec2_std  = float(np.std(ec2_vals, ddof=1))
    sm_mean  = float(np.mean(sm_vals))
    sm_std   = float(np.std(sm_vals, ddof=1))

    abs_diff_sec = ec2_mean - sm_mean          # positive = EC2 faster
    abs_diff_ms  = abs_diff_sec * 1000
    rel_diff_pct = (abs_diff_sec / sm_mean) * 100 if sm_mean != 0 else float("nan")
    share_of_epoch_pct = (abs(abs_diff_sec) / ec2_wall_clock_mean) * 100

    d = cohens_d(ec2_vals, sm_vals)

    t_stat, t_pval = stats.ttest_ind(ec2_vals, sm_vals, equal_var=False, alternative="two-sided")
    # Welch–Satterthwaite degrees of freedom
    s1, s2, n1, n2 = ec2_std, sm_std, len(ec2_vals), len(sm_vals)
    welch_df = (s1**2/n1 + s2**2/n2)**2 / (
        (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
    )

    u_stat, u_pval = stats.mannwhitneyu(ec2_vals, sm_vals, alternative="two-sided")

    stat_sig  = bool(t_pval < ALPHA and u_pval < ALPHA)
    pract_sig = bool(abs(abs_diff_sec) > practical_threshold_sec)

    if pract_sig:
        significance_label = "PRACTICALLY SIGNIFICANT"
    else:
        significance_label = "practically negligible"

    results["components"][comp] = {
        "label": COMPONENT_LABELS[comp],
        "ec2": {
            "values_sec": [round(v, 6) for v in ec2_vals],
            "mean_sec":   round(ec2_mean, 6),
            "std_sec":    round(ec2_std, 6),
        },
        "sagemaker": {
            "values_sec": [round(v, 6) for v in sm_vals],
            "mean_sec":   round(sm_mean, 6),
            "std_sec":    round(sm_std, 6),
        },
        "comparison": {
            "abs_diff_sec":        round(abs_diff_sec, 6),
            "abs_diff_ms":         round(abs_diff_ms, 3),
            "rel_diff_pct":        round(rel_diff_pct, 4),
            "share_of_epoch_pct":  round(share_of_epoch_pct, 4),
            "cohens_d":            round(d, 6),
        },
        "welch_t_test": {
            "t_statistic":  round(float(t_stat), 6),
            "p_value":      round(float(t_pval), 8),
            "degrees_of_freedom": round(float(welch_df), 4),
            "significant":  bool(t_pval < ALPHA),
        },
        "mann_whitney_u_test": {
            "u_statistic":  float(u_stat),
            "p_value":      round(float(u_pval), 8),
            "significant":  bool(u_pval < ALPHA),
        },
        "overall": {
            "statistically_significant_both_tests": stat_sig,
            "practically_significant":              pract_sig,
            "significance_label":                   significance_label,
        },
    }

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=4)

# ---------------------------------------------------------------------------
# Print formatted summary table
# ---------------------------------------------------------------------------
SEP  = "=" * 110
sep2 = "-" * 110

print()
print(SEP)
print("  EXPERIMENT 2 – ResNet50 Component Timing: EC2 (g5.xlarge) vs SageMaker (ml.g5.xlarge)")
print(f"  n = {len(ec2_runs)} runs per platform  |  α = {ALPHA}  |  Practical threshold = 1% of epoch time "
      f"({practical_threshold_sec*1000:.1f} ms)")
print(SEP)
print(f"  {'Component':<22}  {'EC2 mean±std (s)':>20}  {'SM mean±std (s)':>20}  "
      f"{'Δ (ms)':>9}  {'Δ (%)':>7}  {'Epoch%':>6}  {'d':>6}  "
      f"{'t-p':>8}  {'MW-p':>8}  {'Flag'}")
print(sep2)

for comp in COMPONENTS:
    r   = results["components"][comp]
    lbl = r["label"]
    ec2 = r["ec2"]
    sm  = r["sagemaker"]
    cmp = r["comparison"]
    wt  = r["welch_t_test"]
    mw  = r["mann_whitney_u_test"]
    ov  = r["overall"]

    ec2_str = f"{ec2['mean_sec']:.3f} ± {ec2['std_sec']:.3f}"
    sm_str  = f"{sm['mean_sec']:.3f} ± {sm['std_sec']:.3f}"
    flag    = "*** PRACT. SIG." if ov["practically_significant"] else "    negligible"
    stat_mk = " *" if ov["statistically_significant_both_tests"] else "  "

    print(f"  {lbl:<22}  {ec2_str:>20}  {sm_str:>20}  "
          f"{cmp['abs_diff_ms']:>+9.1f}  {cmp['rel_diff_pct']:>+7.2f}  "
          f"{cmp['share_of_epoch_pct']:>6.2f}  {cmp['cohens_d']:>+6.2f}  "
          f"{wt['p_value']:>8.4f}  {mw['p_value']:>8.4f}  {flag}{stat_mk}")

print(sep2)
print()
print("  Column guide:")
print("    Δ (ms)   = EC2_mean − SM_mean  (positive → EC2 took longer)")
print("    Δ (%)    = relative to SM mean")
print("    Epoch%   = |Δ| as % of EC2 epoch wall-clock time")
print("    d        = Cohen's d  (EC2 − SM, pooled SD)")
print("    t-p      = Welch's t-test p-value (equal_var=False, two-tailed)")
print("    MW-p     = Mann-Whitney U p-value  (two-sided)")
print("    *        = statistically significant on BOTH tests (p < 0.05)")
print()
print("  Practical significance threshold: |Δ| > 1% of EC2 epoch wall-clock mean")
print(f"    = {practical_threshold_sec*1000:.2f} ms  (EC2 epoch mean = {ec2_wall_clock_mean:.3f} s)")
print()
print(f"  Results saved to: {OUT_FILE.relative_to(ROOT)}")
print(SEP)
print()
