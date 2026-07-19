"""
Generate standalone 6-notebook comparison chart.
Can be run directly OR pasted as the final cell in any Phase 3 notebook.
Results for Phase 1-2 both datasets + Phase 3 half are hardcoded from saved outputs.
Phase 3 scaleup result read from phase3_results.json if available, else uses placeholder.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import json, os

# ── Load Phase 3 scaleup results if JSON saved ───────────────────────────────
SAVE_DIR_SCALEUP = r'C:\2026\PneumoFusionNet\mimic\main\outputs\Phase_3_triple_fusion_scaleup'
P3S_JSON = os.path.join(SAVE_DIR_SCALEUP, 'phase3_results.json')

if os.path.exists(P3S_JSON):
    with open(P3S_JSON) as f:
        p3s = json.load(f)
    p3s_auc  = p3s['results']['test_auc']
    p3s_acc  = p3s['results']['youden_acc']
    p3s_sens = p3s['results']['youden_sensitivity']
    p3s_spec = p3s['results']['youden_specificity']
    print(f'Phase 3 Scaleup loaded from JSON: AUC={p3s_auc}')
else:
    # Fallback placeholder — replace with actual after training
    p3s_auc, p3s_acc, p3s_sens, p3s_spec = None, None, None, None
    print('Phase 3 Scaleup JSON not found — bar will show 0. Run Cell 17 first.')

# ── All 6 results ─────────────────────────────────────────────────────────────
RESULTS = [
    dict(label='Phase 1\nImage Only\n~1,857',
         ds='Half (~1,857)', auc=0.8152, acc=0.7600, sens=0.6974, spec=None,
         color='#4A90D9', hatch='//'),

    dict(label='Phase 1\nImage Only\n~3,763',
         ds='Scaleup (~3,763)', auc=0.8258, acc=0.7626, sens=0.7103, spec=None,
         color='#E85D5D', hatch='//'),

    dict(label='Phase 2\nImg+Text\n~1,857',
         ds='Half (~1,857)', auc=0.9490, acc=0.8863, sens=0.9137, spec=0.8625,
         color='#4A90D9', hatch=''),

    dict(label='Phase 2\nImg+Text\n~3,763',
         ds='Scaleup (~3,763)', auc=0.9460, acc=0.8780, sens=0.9030, spec=0.8905,
         color='#E85D5D', hatch=''),

    dict(label='Phase 3\nImg+Text+Meta\n~1,857',
         ds='Half (~1,857)', auc=0.9841, acc=0.9470, sens=0.9490, spec=0.9450,
         color='#4A90D9', hatch='..'),

    dict(label='Phase 3\nImg+Text+Meta\n~3,763',
         ds='Scaleup (~3,763)', auc=p3s_auc or 0, acc=p3s_acc or 0,
         sens=p3s_sens or 0, spec=p3s_spec or 0,
         color='#F5A623', hatch='..'),
]

labels  = [r['label'] for r in RESULTS]
aucs    = [r['auc']  for r in RESULTS]
accs    = [r['acc']  for r in RESULTS]
senss   = [r['sens'] for r in RESULTS]
specs   = [r['spec'] if r['spec'] else 0 for r in RESULTS]
colors  = [r['color']  for r in RESULTS]
hatches = [r['hatch']  for r in RESULTS]
x = np.arange(len(labels))
w = 0.6

# ── Style ─────────────────────────────────────────────────────────────────────
BG      = '#0F1117'
CARD_BG = '#1A1D27'
GRID    = '#2A2D3A'
TEXT    = '#E8EAF0'
ACCENT  = '#7B61FF'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD_BG,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': GRID,
    'grid.linewidth': 0.5, 'font.family': 'DejaVu Sans',
})

fig = plt.figure(figsize=(26, 10), facecolor=BG)
fig.suptitle(
    'PneumoFusionNet · Complete Pipeline Comparison\n'
    'Phase 1 (Image Only)  ·  Phase 2 (Image + Text)  ·  Phase 3 (Image + Text + Clinical Metadata)',
    fontsize=15, fontweight='bold', color=TEXT, y=1.01
)

gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)
axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

def draw_panel(ax, vals, title, ylim, ylabel, fmt='.4f',
               target=None, na_indices=None):
    bars = ax.bar(x, vals, width=w, color=colors,
                  edgecolor='#2A2D3A', linewidth=1.0, zorder=3)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
        bar.set_alpha(0.92)

    for bar, v, idx in zip(bars, vals, range(len(vals))):
        if na_indices and idx in na_indices:
            ax.text(bar.get_x() + bar.get_width()/2,
                    ylim[0] + (ylim[1]-ylim[0])*0.05,
                    'N/A', ha='center', va='bottom',
                    fontsize=8, color='#888888', style='italic')
        elif v > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + (ylim[1]-ylim[0])*0.012,
                    format(v, fmt), ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color=TEXT)

    if target:
        ax.axhline(target[0], color=target[1], ls='--', lw=1.8,
                   alpha=0.85, label=target[2], zorder=4)
        ax.legend(fontsize=7.5, loc='lower right',
                  facecolor=CARD_BG, edgecolor=GRID)

    # Phase separators
    ax.axvline(1.5, color=GRID, lw=1.2, ls=':', alpha=0.7, zorder=2)
    ax.axvline(3.5, color=GRID, lw=1.2, ls=':', alpha=0.7, zorder=2)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10, color=TEXT)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.8, rotation=20, ha='right', color=TEXT)
    ax.set_ylim(ylim)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Phase group labels at top
    for xpos, lbl in [(0.5, 'Phase 1'), (2.5, 'Phase 2'), (4.5, 'Phase 3')]:
        ax.text(xpos, ylim[1] - (ylim[1]-ylim[0])*0.03, lbl,
                ha='center', va='top', fontsize=7.5,
                color='#888888', style='italic')

spec_na = [i for i, r in enumerate(RESULTS) if r['spec'] is None]
p3s_na  = [] if p3s_auc else [5]

draw_panel(axes[0], aucs,  'Test AUC',           (0.75, 1.02), 'AUC',
           target=(0.9841, '#5EE87A', 'P3 Half best'))
draw_panel(axes[1], accs,  'Accuracy\n(Youden)', (0.72, 1.00), 'Accuracy',
           na_indices=p3s_na)
draw_panel(axes[2], senss, 'Sensitivity\n(Youden)', (0.60, 1.02), 'Sensitivity',
           target=(0.90, '#FF6B6B', '90% clinical target'),
           na_indices=p3s_na)
draw_panel(axes[3], specs, 'Specificity\n(Youden)', (0.70, 1.00), 'Specificity',
           na_indices=spec_na + p3s_na)

# ── Legend ────────────────────────────────────────────────────────────────────
p_half  = mpatches.Patch(facecolor='#4A90D9', label='Half dataset  (~1,857 images)', edgecolor=GRID)
p_scale = mpatches.Patch(facecolor='#E85D5D', label='Scaleup dataset (~3,763 images)', edgecolor=GRID)
p_p3s   = mpatches.Patch(facecolor='#F5A623', label='Phase 3 Scaleup (current run)',  edgecolor=GRID)
p_img   = mpatches.Patch(facecolor='grey', hatch='//', label='Hatched // = Image only (Phase 1)', edgecolor=TEXT)
p_tri   = mpatches.Patch(facecolor='grey', hatch='..', label='Dotted .. = Triple fusion (Phase 3)', edgecolor=TEXT)

fig.legend(handles=[p_half, p_scale, p_p3s, p_img, p_tri],
           loc='lower center', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, -0.07),
           facecolor=CARD_BG, edgecolor=GRID, framealpha=0.9)

plt.tight_layout()
out_path = os.path.join(SAVE_DIR_SCALEUP, 'full_pipeline_6way_comparison.png')
os.makedirs(SAVE_DIR_SCALEUP, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print('\n' + '='*85)
print('  PneumoFusionNet · COMPLETE PIPELINE COMPARISON (All 6 Notebooks)')
print('='*85)
print(f'  {"Model":<40} {"Dataset":<20} {"AUC":>7} {"Acc":>7} {"Sens":>7} {"Spec":>7}')
print('  ' + '-'*82)
for r in RESULTS:
    spec_str = f"{r['spec']:.4f}" if r['spec'] else '   N/A'
    p3s_flag = ' *' if r['label'].startswith('Phase 3') and '3,763' in r['ds'] and not p3s_auc else ''
    lbl = r['label'].replace('\n', ' ')
    print(f"  {lbl:<40} {r['ds']:<20} {r['auc']:>7.4f} {r['acc']:>7.4f} {r['sens']:>7.4f} {spec_str:>7}{p3s_flag}")
print('='*85)
if not p3s_auc:
    print('  * Phase 3 Scaleup result pending — run Cell 17 (Save JSON) first')

# ── Delta table ───────────────────────────────────────────────────────────────
print('\n  IMPROVEMENT: Phase 3 vs Phase 2 (same dataset)')
print(f'  {"Comparison":<45} {"ΔAUC":>8} {"ΔSens":>8} {"ΔSpec":>8}')
print('  ' + '-'*72)

# P3 Half vs P2 Half
d_auc  = 0.9841 - 0.9490
d_sens = 0.9490 - 0.9137
d_spec = 0.9450 - 0.8625
print(f'  {"P3 Half vs P2 Half  (1,857 images)":<45} {d_auc:>+8.4f} {d_sens:>+8.4f} {d_spec:>+8.4f}')

# P3 Scaleup vs P2 Scaleup
if p3s_auc:
    d_auc2  = p3s_auc  - 0.9460
    d_sens2 = p3s_sens - 0.9030
    d_spec2 = p3s_spec - 0.8905
    print(f'  {"P3 Scaleup vs P2 Scaleup (3,763 images)":<45} {d_auc2:>+8.4f} {d_sens2:>+8.4f} {d_spec2:>+8.4f}')

print(f'\n  Chart saved: {out_path}')
