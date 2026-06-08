"""
patch_phase2_notebook.py
Programmatically patches Phase-2-FINAL_multimodal_classifier.ipynb to:
  1. Use dataset_v3.csv instead of v2
  2. Drop impression columns in Step 3 (leakage guard)
  3. Use 'findings' column in MultimodalDataset instead of 'impression'
"""
import json, re, os

NB_PATH = r'C:\2026\PneumoFusionNet\mimic\mimic_pilot_139\Notebooks\Phase-2-FINAL_multimodal_classifier.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

patched = 0

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue

    src = ''.join(cell['source'])

    # ── PATCH 1: CSV path v2 → v3 ────────────────────────────────────────────
    if 'mimic_multimodal_dataset_v2.csv' in src:
        cell['source'] = [
            line.replace('mimic_multimodal_dataset_v2.csv',
                         'mimic_multimodal_dataset_v3.csv')
            for line in cell['source']
        ]
        patched += 1
        print('[PATCH 1] CSV path: v2 → v3')

    # ── PATCH 2: Step 3 — drop impression, use findings ──────────────────────
    if "df['impression'] = df['impression'].fillna" in src:
        new_lines = []
        for line in cell['source']:
            new_lines.append(line)
            # After the image-path fix line, inject leakage guard
            if "df['image_path'] = df['image_path'].str.replace" in line:
                new_lines += [
                    "\n",
                    "# ── LEAKAGE GUARD ──────────────────────────────────────────────────────────\n",
                    "# 'impression' / 'impression_orig' state the diagnosis directly (HIGH leakage).\n",
                    "# Drop them NOW so they CANNOT be accidentally fed to the model.\n",
                    "# 'findings' is the only text used for model training.\n",
                    "LEAKY_COLS = [c for c in df.columns if 'impression' in c.lower()]\n",
                    "if LEAKY_COLS:\n",
                    "    print(f'Dropping leaky columns: {LEAKY_COLS}')\n",
                    "    df = df.drop(columns=LEAKY_COLS)\n",
                    "\n",
                    "# Fill missing findings (21/139 reports had no FINDINGS section)\n",
                    "df['findings'] = df['findings'].fillna('No radiological findings documented.').str.strip()\n",
                    "df.loc[df['findings'] == '', 'findings'] = 'No radiological findings documented.'\n",
                ]
            # Remove old impression-fill lines
            if "df['impression'] = df['impression'].fillna" in line:
                new_lines.pop()  # remove the line just added (was the impression line)
            if "df.loc[df['impression'] == '', 'impression']" in line:
                new_lines.pop()

        # Fix sample display: impression → findings
        new_lines = [
            line.replace("'Sample impressions:'", "'Sample FINDINGS (no leakage):'")
                .replace('row[\\"impression\\"]', 'row[\\"findings\\"]')
                .replace("row['impression']", "row['findings']")
                .replace('[:80]}', '[:90]}')
            for line in new_lines
        ]

        # Add column check line before df.head()
        final = []
        for line in new_lines:
            if 'df.head()' in line:
                final += [
                    "print('Columns available:', list(df.columns))\n",
                    "print('Impression column present:', any('impression' in c for c in df.columns))  # Must be False!\n",
                ]
            final.append(line)
        cell['source'] = final
        patched += 1
        print('[PATCH 2] Step 3: Drop impression, fill findings, update display')

    # ── PATCH 3: MultimodalDataset — impression → findings ───────────────────
    if "row['impression']" in src and 'class MultimodalDataset' in src:
        new_lines = []
        for line in cell['source']:
            # Update docstring comment
            line = line.replace(
                '# ── Text (IMPRESSION section)',
                '# ── Text: FINDINGS section (NOT impression)'
            )
            # Change the column read
            line = line.replace(
                "str(row['impression']) if pd.notna(row['impression']) else 'No findings reported.'",
                "str(row['findings']) if pd.notna(row['findings']) else 'No radiological findings documented.'"
            )
            new_lines.append(line)

        # Add leakage assert inside __init__
        patched_lines = []
        for line in new_lines:
            patched_lines.append(line)
            if 'self.df        = df.reset_index(drop=True)' in line:
                # Insert assert before self.df = ...
                patched_lines.insert(
                    len(patched_lines) - 1,
                    "        # Leakage guard: impression must not be present\n"
                )
                patched_lines.insert(
                    len(patched_lines) - 1,
                    "        assert not any('impression' in c for c in df.columns), \\\n"
                )
                patched_lines.insert(
                    len(patched_lines) - 1,
                    "            'LEAKAGE: impression column present! Drop it in Step 3 first.'\n"
                )

        cell['source'] = patched_lines
        patched += 1
        print('[PATCH 3] MultimodalDataset: impression → findings + leakage assert')

print(f'\nTotal patches applied: {patched}')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Notebook saved: {NB_PATH}')
print('Done.')
