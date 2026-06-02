"""
build_multimodal_dataset.py
----------------------------
Reads mimic_dataset.csv (image + label), extracts matching radiology
report .txt files from the zip, saves them to mimic_pilot/reports/,
and builds mimic_multimodal_dataset.csv with report_path column.

Output: mimic_multimodal_dataset.csv
Columns:
  subject_id | study_id | image_path | label | label_name | report_path | impression
"""

import os
import re
import zipfile
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CSV_PATH     = os.path.join(BASE_DIR, 'mimic_pilot', 'mimic_dataset.csv')
ZIP_PATH     = os.path.join(BASE_DIR, 'mimic_pilot', 'reports', 'mimic-cxr-reports.zip')
REPORTS_DIR  = os.path.join(BASE_DIR, 'mimic_pilot', 'reports', 'txt')
OUT_PATH     = os.path.join(BASE_DIR, 'mimic_pilot', 'mimic_multimodal_dataset.csv')

os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Load existing image+label dataset ────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Loaded          : {len(df)} rows")

# ── Helper: extract IMPRESSION from report text ───────────────────────────────
def extract_impression(text):
    text = text.strip()
    match = re.search(
        r'IMPRESSION[S]?\s*:?\s*\n?(.*?)(?:\n\s*\n|\Z)',
        text, re.IGNORECASE | re.DOTALL
    )
    if match:
        return re.sub(r'\s+', ' ', match.group(1).strip())
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if paragraphs:
        return re.sub(r'\s+', ' ', paragraphs[-1])
    return text[:300]

# ── Extract reports from zip ──────────────────────────────────────────────────
report_paths = []
impressions  = []
found = 0; not_found = 0

print("Opening zip file...")

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    zip_contents = set(z.namelist())
    print(f"Zip files       : {len(zip_contents):,}")
    print(f"Extracting {len(df)} reports to: {REPORTS_DIR}\n")

    for _, row in df.iterrows():
        subject_id = int(row['subject_id'])
        study_id   = int(row['study_id'])

        p_folder = f"p{str(subject_id)[:2]}"
        zip_path = f"files/{p_folder}/p{subject_id}/s{study_id}.txt"

        # Save path: reports/txt/p10000032_s50414267.txt
        save_name = f"p{subject_id}_s{study_id}.txt"
        save_path = os.path.join(REPORTS_DIR, save_name)

        if zip_path in zip_contents:
            # Extract if not already done
            if not os.path.exists(save_path):
                with z.open(zip_path) as f:
                    content = f.read().decode('utf-8', errors='replace')
                with open(save_path, 'w', encoding='utf-8') as out:
                    out.write(content)
            else:
                with open(save_path, 'r', encoding='utf-8') as f:
                    content = f.read()

            report_paths.append(save_path)
            impressions.append(extract_impression(content))
            found += 1
        else:
            report_paths.append('')
            impressions.append('')
            not_found += 1
            print(f"  NOT FOUND: {zip_path}")

# ── Build final dataset ───────────────────────────────────────────────────────
df['report_path'] = report_paths
df['impression']  = impressions

df_final = df[df['report_path'] != ''].reset_index(drop=True)
df_final.to_csv(OUT_PATH, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print('=' * 50)
print('  MULTIMODAL DATASET BUILT')
print('=' * 50)
print(f'  Reports found : {found} / {len(df)}')
print(f'  Not found     : {not_found}')
print(f'  Saved to      : {OUT_PATH}')
print('=' * 50)
print()
print('Columns:', list(df_final.columns))
print()
print('Sample row:')
r = df_final.iloc[0]
print(f"  image_path  : {r['image_path']}")
print(f"  report_path : {r['report_path']}")
print(f"  label       : {r['label_name']}")
print(f"  impression  : {r['impression']}")
