"""
Experiment 2 — SageMaker Results Aggregator
Laddar ner alla training_results_*.json från S3,
beräknar mean, std dev och 95% CI över 5 runs.
"""

import boto3
import json
import math
import os
import tarfile
import tempfile

# ===== SETTINGS =====
BUCKET        = "sagemaker-eu-north-1-600889066998"
REGION        = "eu-north-1"
OUTPUT_FILE   = "sagemaker_exp2_aggregated.json"
TABLE_FILE    = "sagemaker_exp2_summary_table.txt"

# Lägg till job-namnen för alla 5 körningar här
JOB_NAMES = [
    "pytorch-training-2026-04-29-12-55-01-136",
    "pytorch-training-2026-04-29-16-42-32-777",
    "pytorch-training-2026-04-29-20-31-28-192",
    "pytorch-training-2026-04-30-00-19-48-404",
    "pytorch-training-2026-04-30-04-08-07-224",
]

# Metrics att aggregera
METRICS = [
    "train_time_sec",
    "training_cost_usd",
    "avg_wall_clock_ms",
    "avg_data_loading_ms",
    "avg_forward_pass_ms",
    "avg_backward_pass_ms",
    "avg_optimizer_step_ms",
    "avg_total_measured_ms",
    "avg_platform_overhead_ms",
    "avg_overhead_percentage",
    "avg_cost_per_epoch_usd",
    "best_val_accuracy",
    "test_accuracy",
]


# ===== HELPERS =====

def mean(values):
    return sum(values) / len(values)

def std_dev(values):
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)

def ci_95(values):
    return 1.96 * std_dev(values) / math.sqrt(len(values))

def download_results(s3, bucket, job_name):
    """Laddar ner output.tar.gz för ett job och extraherar training_results_*.json"""
    prefix = f"{job_name}/output/model.tar.gz"
    print(f"Downloading s3://{bucket}/{prefix} ...")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_tar = os.path.join(tmpdir, "output.tar.gz")
        s3.download_file(bucket, prefix, local_tar)

        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(tmpdir)

        # Hitta training_results_*.json
        for root, _, files in os.walk(tmpdir):
            for f in files:
                if f.startswith("training_results_") and f.endswith(".json"):
                    with open(os.path.join(root, f)) as fp:
                        return json.load(fp)

    raise FileNotFoundError(f"training_results_*.json not found in {job_name}")


# ===== MAIN =====

s3 = boto3.client("s3", region_name=REGION)

all_results = []
for job in JOB_NAMES:
    try:
        results = download_results(s3, BUCKET, job)
        all_results.append(results)
        print(f"  ✓ {job} — test_acc: {results.get('test_accuracy', 'N/A')}")
    except Exception as e:
        print(f"  ✗ {job} — ERROR: {e}")

if not all_results:
    print("Inga resultat hittades. Avbryter.")
    exit(1)

n = len(all_results)
print(f"\nAggregerar {n} runs...")

aggregated = {
    "experiment":   "experiment2_overhead_decomposition",
    "platform":     "sagemaker",
    "n_runs":       n,
    "job_names":    JOB_NAMES[:n],
}

# Beräkna stats för varje metric
for metric in METRICS:
    values = []
    for r in all_results:
        if metric in r:
            values.append(r[metric])

    if not values:
        print(f"  Warning: {metric} saknas i alla runs, hoppar över.")
        continue

    m   = mean(values)
    sd  = std_dev(values) if len(values) > 1 else 0.0
    ci  = ci_95(values)   if len(values) > 1 else 0.0

    aggregated[metric] = {
        "values":   [round(v, 6) for v in values],
        "mean":     round(m,  6),
        "std_dev":  round(sd, 6),
        "ci_95":    round(ci, 6),
    }

# Spara JSON
with open(OUTPUT_FILE, "w") as f:
    json.dump(aggregated, f, indent=4)
print(f"\nAggregated JSON saved to: {OUTPUT_FILE}")

# ===== LÄSBAR TABELL =====
lines = []
lines.append("=" * 72)
lines.append(f"SAGEMAKER EXP2 AGGREGATED RESULTS  (n={n} runs)")
lines.append("=" * 72)
lines.append(f"{'Metric':<35} {'Mean':>12} {'Std Dev':>10} {'95% CI':>10}")
lines.append("-" * 72)

for metric in METRICS:
    if metric not in aggregated:
        continue
    d = aggregated[metric]
    lines.append(
        f"{metric:<35} {d['mean']:>12.4f} {d['std_dev']:>10.4f} {d['ci_95']:>10.4f}"
    )

lines.append("=" * 72)

# Skriv ut och spara
table_str = "\n".join(lines)
print("\n" + table_str)

with open(TABLE_FILE, "w") as f:
    f.write(table_str)
print(f"\nSummary table saved to: {TABLE_FILE}")