"""
Experiment 2 — Stacked Bar Chart
Komponent-wise breakdown: EC2 vs SageMaker
Kör från: ~/Cloud-based-SLR-1/training/
Sparar: outputs/experiment2/exp2_stacked_bar.pdf och .png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ===== DATA (från aggregerade resultat) =====

components = {
    'Data Loading':   {'ec2': 51930.80,  'sm': 88638.30},
    'Forward Pass':   {'ec2': 19743.49,  'sm': 19802.79},
    'Backward Pass':  {'ec2': 36796.40,  'sm': 35919.53},
    'Optimizer Step': {'ec2':  4031.64,  'sm':  4456.42},
    'Platform\nOverhead': {'ec2': 656.06, 'sm': 534.56},
}

# Std dev för error bars på totalen
ec2_total_std = 745.94
sm_total_std  = 1396.37

# ===== FÄRGER =====
colors = {
    'Data Loading':       '#E63946',   # röd  — den som skiljer sig
    'Forward Pass':       '#457B9D',   # blå
    'Backward Pass':      '#1D3557',   # mörkblå
    'Optimizer Step':     '#A8DADC',   # ljusblå
    'Platform\nOverhead': '#F4A261',   # orange
}

# ===== PLOT =====
fig, ax = plt.subplots(figsize=(9, 6))

x       = np.array([0, 1])
width   = 0.5
labels  = ['EC2\n(g5.xlarge)', 'SageMaker\n(ml.g5.xlarge)']
bottoms = np.zeros(2)

bars_for_legend = []

for comp, vals in components.items():
    heights = np.array([vals['ec2'], vals['sm']]) / 1000  # ms → sekunder
    bar = ax.bar(
        x,
        heights,
        width,
        bottom=bottoms / 1000,
        color=colors[comp],
        edgecolor='white',
        linewidth=0.8,
        label=comp
    )
    bars_for_legend.append(bar)

    # Lägg till värde-label inuti stapeln om tillräckligt hög
    for i, (h, b) in enumerate(zip(heights, bottoms / 1000)):
        if h > 2:  # bara om stapeln är > 2s
            ax.text(
                x[i], b + h / 2,
                f'{h:.1f}s',
                ha='center', va='center',
                fontsize=8.5, color='white', fontweight='bold'
            )
    bottoms += np.array([vals['ec2'], vals['sm']])

# Error bars på toppen
totals = bottoms / 1000
ax.errorbar(
    x, totals,
    yerr=[ec2_total_std / 1000, sm_total_std / 1000],
    fmt='none',
    color='black',
    capsize=6,
    linewidth=1.5,
    label='_nolegend_'
)

# Total-värde ovanför varje stapel
for i, (xi, tot) in enumerate(zip(x, totals)):
    ax.text(
        xi, tot + 1.5,
        f'{tot:.0f}s',
        ha='center', va='bottom',
        fontsize=10, fontweight='bold', color='black'
    )

# ===== ANNOTERING: markera data loading skillnad =====
dl_ec2 = components['Data Loading']['ec2'] / 1000
dl_sm  = components['Data Loading']['sm']  / 1000

ax.text(
    1.35, dl_sm / 2,
    'Data Loading\ndifference:\n+70.7%\n(36.7s)',
    ha='left', va='center',
    fontsize=9,
    color=colors['Data Loading'],
    fontweight='bold'
)

# ===== STYLING =====
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Average Epoch Time (seconds)', fontsize=11)

ax.set_xlim(-0.5, 1.8)
ax.set_ylim(0, max(totals) * 1.15)
ax.yaxis.grid(True, alpha=0.4, linestyle='--')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
legend = ax.legend(
    loc='upper left',
    fontsize=9,
    framealpha=0.9,
    title='Component',
    title_fontsize=9
)

plt.tight_layout()

# ===== SPARA =====
out_dir = 'outputs/experiment2'
os.makedirs(out_dir, exist_ok=True)

pdf_path = os.path.join(out_dir, 'exp2_stacked_bar.pdf')
png_path = os.path.join(out_dir, 'exp2_stacked_bar.png')

plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
plt.savefig(png_path, dpi=300, bbox_inches='tight')

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
