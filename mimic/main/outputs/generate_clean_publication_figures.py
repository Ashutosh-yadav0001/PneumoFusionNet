r"""
generate_clean_publication_figures.py
------------------------------------
Generates crystal-clear, high-resolution, publication-ready figures for PneumoFusionNet
with PURE WHITE BACKGROUND, LARGE VISIBLE TYPOGRAPHY, and HIGH CONTRAST as requested by the professor.

Outputs saved to brain artifact directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = r"C:\Users\ashutosh\.gemini\antigravity\brain\7fe6dfc2-5088-4163-b4e1-5291a55d3093"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# GLOBAL PUBLICATION STYLE CONFIGURATION
# -----------------------------------------------------------------------------
plt.rcParams.update({
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor':   '#FFFFFF',
    'axes.edgecolor':   '#1E293B',
    'axes.linewidth':   1.4,
    'axes.labelcolor':  '#0F172A',
    'xtick.color':      '#0F172A',
    'ytick.color':      '#0F172A',
    'text.color':       '#0F172A',
    'grid.color':       '#E2E8F0',
    'grid.linewidth':   0.9,
    'grid.linestyle':   '--',
    'font.family':      'DejaVu Sans',
    'font.size':        12,
})

# =============================================================================
# FIGURE 3A: 6-WAY FULL PIPELINE COMPARISON (4 PANELS: AUC, ACC, SENS, SPEC)
# =============================================================================
def generate_fig3_6way():
    results = [
        dict(label="Phase 1\n(Half, ~1.8k)",    stage="Phase 1 (Image)",    ds="Half",    auc=0.8152, acc=0.7600, sens=0.6974, spec=0.8160, col="#2563EB", hatch="//"),
        dict(label="Phase 1\n(Scaleup, ~3.8k)", stage="Phase 1 (Image)",    ds="Scaleup", auc=0.8258, acc=0.7626, sens=0.7103, spec=0.8160, col="#DC2626", hatch="//"),
        dict(label="Phase 2\n(Half, ~2.0k)",    stage="Phase 2 (+Text)",   ds="Half",    auc=0.9490, acc=0.8863, sens=0.9137, spec=0.8625, col="#2563EB", hatch="\\\\"),
        dict(label="Phase 2\n(Scaleup, ~3.8k)", stage="Phase 2 (+Text)",   ds="Scaleup", auc=0.9460, acc=0.8780, sens=0.9030, spec=0.8905, col="#DC2626", hatch="\\\\"),
        dict(label="Phase 3\n(Half, ~1.8k)",    stage="Phase 3 (+Clin)",   ds="Half",    auc=0.9742, acc=0.9246, sens=0.9157, spec=0.9318, col="#2563EB", hatch=".."),
        dict(label="Phase 3c\n(Scaleup, ~3.8k)", stage="Phase 3c (+WBC)",  ds="Scaleup", auc=0.9711, acc=0.9292, sens=0.9225, spec=0.9359, col="#16A34A", hatch=""),
    ]

    labels = [r["label"] for r in results]
    colors = [r["col"] for r in results]
    hatches = [r["hatch"] for r in results]
    x = np.arange(len(labels))
    w = 0.58

    fig, axes = plt.subplots(1, 4, figsize=(24, 7.5), facecolor="#FFFFFF")
    fig.suptitle(
        "Figure 3: PneumoFusionNet Complete Multimodal Pipeline Progression\n"
        "Phase 1 (Chest X-Ray)  →  Phase 2 (X-Ray + Radiology Text)  →  Phase 3/3c (Triple Fusion with Clinical Biomarker)",
        fontsize=16, fontweight="bold", color="#0F172A", y=1.03
    )

    panels = [
        ("auc",  "Test ROC-AUC",        (0.75, 1.02),  "{:.4f}", (0.9711, "#16A34A", "Phase 3c SOTA (0.971)")),
        ("acc",  "Accuracy",            (0.70, 1.00),  "{:.2%}", (0.9292, "#16A34A", "Phase 3c Top (92.9%)")),
        ("sens", "Sensitivity (Recall)",(0.60, 1.03),  "{:.2%}", (0.9000, "#EF4444", "Clinical Target (≥90%)")),
        ("spec", "Specificity",         (0.70, 1.00),  "{:.2%}", (0.9359, "#16A34A", "Phase 3c Top (93.6%)")),
    ]

    for ax, (metric_key, title, ylim, fmt, ref_line) in zip(axes, panels):
        vals = [r[metric_key] for r in results]
        bars = ax.bar(x, vals, width=w, color=colors, edgecolor="#0F172A", linewidth=1.4, zorder=3)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
            bar.set_alpha(0.92)

        # Highlighting Phase 3c
        bars[-1].set_edgecolor("#15803D")
        bars[-1].set_linewidth(2.6)

        # Value annotations
        for bar, v in zip(bars, vals):
            text_str = fmt.format(v)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + (ylim[1] - ylim[0]) * 0.015,
                text_str,
                ha="center", va="bottom",
                fontsize=11.5, fontweight="bold", color="#0F172A"
            )

        # Reference target line
        if ref_line:
            val, col, desc = ref_line
            ax.axhline(val, color=col, linestyle="--", linewidth=1.8, alpha=0.85, label=desc, zorder=4)
            ax.legend(loc="lower right", fontsize=10.5, framealpha=0.95, facecolor="#F8FAFC", edgecolor="#CBD5E1")

        # Visual phase dividers
        ax.axvline(1.5, color="#94A3B8", linestyle=":", linewidth=1.5, alpha=0.9, zorder=2)
        ax.axvline(3.5, color="#94A3B8", linestyle=":", linewidth=1.5, alpha=0.9, zorder=2)

        # Header phase tags
        ax.text(0.5, ylim[1] - (ylim[1] - ylim[0]) * 0.04, "Phase 1: Image", ha="center", fontsize=10.5, fontweight="bold", color="#475569")
        ax.text(2.5, ylim[1] - (ylim[1] - ylim[0]) * 0.04, "Phase 2: +Text",  ha="center", fontsize=10.5, fontweight="bold", color="#475569")
        ax.text(4.5, ylim[1] - (ylim[1] - ylim[0]) * 0.04, "Phase 3: Triple", ha="center", fontsize=10.5, fontweight="bold", color="#15803D")

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12, color="#0F172A")
        ax.set_ylabel(title, fontsize=12, fontweight="bold", color="#0F172A")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10.5, fontweight="bold", rotation=22, ha="right", color="#0F172A")
        ax.set_ylim(ylim)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Professional bottom legend
    leg_half    = mpatches.Patch(facecolor="#2563EB", edgecolor="#0F172A", label="Half-Dataset Cohort (~1,857 samples)")
    leg_scale   = mpatches.Patch(facecolor="#DC2626", edgecolor="#0F172A", label="Scaleup Cohort (~3,763 samples)")
    leg_p3c     = mpatches.Patch(facecolor="#16A34A", edgecolor="#15803D", label="Phase 3c Final Model (WBC Point-of-Care Fusion)")
    leg_h_p1    = mpatches.Patch(facecolor="#E2E8F0", hatch="//", edgecolor="#0F172A", label="Hatch '//' = CXR Only")
    leg_h_p2    = mpatches.Patch(facecolor="#E2E8F0", hatch="\\\\", edgecolor="#0F172A", label="Hatch '\\\\' = CXR + Text")
    leg_h_p3    = mpatches.Patch(facecolor="#E2E8F0", hatch="..", edgecolor="#0F172A", label="Hatch '..' = CXR + Text + Labs")

    fig.legend(
        handles=[leg_half, leg_scale, leg_p3c, leg_h_p1, leg_h_p2, leg_h_p3],
        loc="lower center", ncol=6, fontsize=11, framealpha=0.98,
        facecolor="#F8FAFC", edgecolor="#CBD5E1",
        bbox_to_anchor=(0.5, -0.09)
    )

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "Fig3_Full_Pipeline_Comparison_Clean.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    print(f"Saved: {save_path}")
    plt.close(fig)


# =============================================================================
# FIGURE 3B: PHASE PROGRESSION TO WBC FUSION (CLEAN PUBLICATION REPLACEMENT)
# =============================================================================
def generate_fig3_progression_p3c():
    phases = [
        {"name": "Phase 1\nImage Only\n(DenseNet+CBAM)", "auc": 0.8258, "sens": 71.03, "spec": 81.60, "acc": 76.26, "color": "#3B82F6"},
        {"name": "Phase 2v2\nCXR + Text\n(Cross-Attention)", "auc": 0.9460, "sens": 90.30, "spec": 89.05, "acc": 87.80, "color": "#0D9488"},
        {"name": "Phase 3\nCXR + Text\n+ 17 Clin Feats", "auc": 0.9692, "sens": 89.10, "spec": 94.30, "acc": 91.70, "color": "#E11D48"},
        {"name": "Phase 3c (Ours)\nCXR + Text\n+ WBC Count Only", "auc": 0.9711, "sens": 92.25, "spec": 93.59, "acc": 92.92, "color": "#16A34A"},
    ]

    names  = [p["name"]  for p in phases]
    colors = [p["color"] for p in phases]
    x      = np.arange(len(phases))
    w      = 0.55

    fig, axes = plt.subplots(1, 4, figsize=(22, 6.8), facecolor="#FFFFFF")
    fig.suptitle(
        "Figure 3: PneumoFusionNet Multimodal Performance Progression  (Scaleup Cohort, N = 3,763)\n"
        "Comparison of Image-Only (Phase 1), Text-Fusion (Phase 2v2), Full Clinical Panel (Phase 3), and Point-of-Care WBC (Phase 3c)",
        fontsize=15, fontweight="bold", color="#0F172A", y=1.03
    )

    metrics = [
        ("auc",  "Test ROC-AUC",        (0.75, 1.02), "{:.4f}"),
        ("sens", "Sensitivity (%)",     (65.0, 102.0), "{:.1f}%"),
        ("spec", "Specificity (%)",     (75.0, 102.0), "{:.1f}%"),
        ("acc",  "Accuracy (%)",        (70.0, 102.0), "{:.1f}%"),
    ]

    for ax, (key, title, ylim, fmt) in zip(axes, metrics):
        vals = [p[key] for p in phases]
        bars = ax.bar(x, vals, width=w, color=colors, edgecolor="#0F172A", linewidth=1.3, zorder=3)
        bars[-1].set_edgecolor("#15803D")
        bars[-1].set_linewidth(2.8)

        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                v + (ylim[1] - ylim[0]) * 0.015,
                fmt.format(v),
                ha="center", va="bottom",
                fontsize=11.5, fontweight="bold", color="#0F172A"
            )

        if key == "sens":
            ax.axhline(90.0, color="#EF4444", linestyle="--", linewidth=1.8, alpha=0.85, label="Clinical Target (≥90%)")
            ax.legend(loc="lower right", fontsize=10.5, facecolor="#F8FAFC", edgecolor="#CBD5E1")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10.5, fontweight="bold", color="#0F172A")
        ax.set_ylim(ylim)
        ax.set_ylabel(title, fontsize=12, fontweight="bold", color="#0F172A")
        ax.set_title(title, fontsize=13.5, fontweight="bold", pad=10, color="#0F172A")
        ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "Fig3_Phase_Progression_P3c_Clean.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    print(f"Saved: {save_path}")
    plt.close(fig)


# =============================================================================
# FIGURE 3C: ORIGINAL 2-PANEL COMPARISON (FIGURE 3 IN COMBINED REPORT)
# =============================================================================
def generate_fig3_combined_p1_p2():
    fig3, axes3 = plt.subplots(1, 2, figsize=(18, 7.5), facecolor="#FFFFFF")
    fig3.suptitle(
        "Figure 3: PneumoFusionNet Multimodal Gain and Cohort Scaling Analysis\n"
        "Left: Progression from Image-Only (Phase 1) to Multimodal (Phase 2) | Right: Multi-Metric Cohort Comparison",
        fontsize=15, fontweight="bold", color="#0F172A", y=1.02
    )

    HALF_COL  = "#2563EB"  # Royal Blue
    SCALE_COL = "#DC2626"  # Crimson Red

    # Panel 1: AUC progression
    ax = axes3[0]
    phases = ["Phase 1\n(Half)", "Phase 1\n(Scaleup)", "Phase 2\n(Half)", "Phase 2\n(Scaleup)"]
    auc_vals = [0.8152, 0.8258, 0.9490, 0.9460]
    cols = [HALF_COL, SCALE_COL, HALF_COL, SCALE_COL]

    bars = ax.bar(range(4), auc_vals, color=cols, alpha=0.88, width=0.52, edgecolor="#0F172A", linewidth=1.3, zorder=3)
    ax.plot([0, 2], [0.8152, 0.9490], "o--", color=HALF_COL, lw=2.4, ms=9, label="Half Data progression (+16.4% AUC)")
    ax.plot([1, 3], [0.8258, 0.9460], "s--", color=SCALE_COL, lw=2.4, ms=9, label="Scaleup progression (+14.6% AUC)")

    for bar, v in zip(bars, auc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#0F172A")

    ax.set_xticks(range(4))
    ax.set_xticklabels(phases, fontsize=11, fontweight="bold", color="#0F172A")
    ax.set_ylim(0.76, 1.00)
    ax.set_ylabel("Test ROC-AUC", fontsize=13, fontweight="bold", color="#0F172A")
    ax.set_title("Test ROC-AUC Progression (Phase 1 → Phase 2)", fontsize=14, fontweight="bold", pad=12, color="#0F172A")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.legend(fontsize=11, loc="lower right", facecolor="#F8FAFC", edgecolor="#CBD5E1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: All metrics grouped
    ax = axes3[1]
    all_metrics = ["ROC-AUC", "Accuracy", "Sensitivity\n(Recall)", "Specificity"]
    ph1_half_m  = [0.8152, 0.7600, 0.6974, 0.8160]
    ph1_scale_m = [0.8258, 0.7626, 0.7103, 0.8160]
    ph2_half_m  = [0.9490, 0.8863, 0.9137, 0.8625]
    ph2_scale_m = [0.9460, 0.8780, 0.9030, 0.8905]

    x = np.arange(4)
    w = 0.20
    ax.bar(x - 1.5*w, ph1_half_m,  w, color=HALF_COL,  alpha=0.65, label="P1 Half (Image)", hatch="//", edgecolor="#0F172A", linewidth=1.1, zorder=3)
    ax.bar(x - 0.5*w, ph1_scale_m, w, color=SCALE_COL, alpha=0.65, label="P1 Scaleup (Image)", hatch="//", edgecolor="#0F172A", linewidth=1.1, zorder=3)
    ax.bar(x + 0.5*w, ph2_half_m,  w, color=HALF_COL,  alpha=0.95, label="P2 Half (Multimodal)", edgecolor="#0F172A", linewidth=1.3, zorder=3)
    ax.bar(x + 1.5*w, ph2_scale_m, w, color=SCALE_COL, alpha=0.95, label="P2 Scaleup (Multimodal)", edgecolor="#0F172A", linewidth=1.3, zorder=3)

    for i in range(4):
        for offset, vals in [(-1.5*w, ph1_half_m), (-0.5*w, ph1_scale_m), (+0.5*w, ph2_half_m), (+1.5*w, ph2_scale_m)]:
            v = vals[i]
            ax.text(x[i] + offset, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#0F172A")

    ax.set_xticks(x)
    ax.set_xticklabels(all_metrics, fontsize=11, fontweight="bold", color="#0F172A")
    ax.set_ylim(0.60, 1.02)
    ax.set_ylabel("Score", fontsize=13, fontweight="bold", color="#0F172A")
    ax.set_title("Performance Metrics Across Datasets (P1 Hatched, P2 Solid)", fontsize=14, fontweight="bold", pad=12, color="#0F172A")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.legend(fontsize=10.5, ncol=2, loc="lower right", facecolor="#F8FAFC", edgecolor="#CBD5E1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "Fig3_P1_P2_Combined_Comparison_Clean.png")
    fig3.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    print(f"Saved: {save_path}")

    # Overwrite combined_comparison.png in brain
    fig3.savefig(os.path.join(OUT_DIR, "combined_comparison.png"), dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig3)

if __name__ == "__main__":
    print("Generating Clean, Publication-Ready Figures for Professor...")
    generate_fig3_6way()
    generate_fig3_progression_p3c()
    generate_fig3_combined_p1_p2()
    print("All Figures Successfully Generated with Clean White Background & High Legibility!")
