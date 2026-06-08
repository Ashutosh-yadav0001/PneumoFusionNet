"""
rebuild_dataset_findings.py
----------------------------
Re-extracts FINDINGS and INDICATION sections from existing .txt reports
to replace the leaky IMPRESSION column in mimic_multimodal_dataset_v2.csv.

Output: mimic_multimodal_dataset_v3.csv
Columns:
  subject_id | study_id | image_path | label | label_name |
  report_path | impression (kept for reference) | findings | indication

LEAKAGE ANALYSIS:
  IMPRESSION  → HIGH leakage  (diagnosis stated directly)
  FINDINGS    → MEDIUM leakage (observational — describes what is SEEN)
  INDICATION  → LOW leakage   (written BEFORE radiologist reads image)
  findings_only → recommended for Phase 2 training
"""

import os
import re
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = r'C:\2026\PneumoFusionNet\mimic\mimic_pilot_139'
CSV_V2     = os.path.join(BASE_DIR, 'dataset_139', 'mimic_multimodal_dataset_v2.csv')
CSV_OUT    = os.path.join(BASE_DIR, 'dataset_139', 'mimic_multimodal_dataset_v3.csv')
REPORT_DIR = os.path.join(BASE_DIR, 'reports', 'txt')


# ── Section extractor ─────────────────────────────────────────────────────────
def extract_section(text, section_names):
    """
    Extract a named section from a radiology report.
    section_names: list of possible header variants to try.
    Returns the section text or '' if not found.
    """
    # Build a regex pattern that matches any of the section names
    pattern_str = '|'.join(re.escape(s) for s in section_names)
    pattern = re.compile(
        rf'(?:{pattern_str})\s*:?\s*\n?(.*?)(?:\n\s*\n[A-Z]|\Z)',
        re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(text)
    if match:
        raw = match.group(1).strip()
        # Collapse whitespace
        cleaned = re.sub(r'\s+', ' ', raw)
        # Remove trailing section header noise (e.g., "IMPRESSION: ...")
        cleaned = re.sub(
            r'\s*(IMPRESSION|CONCLUSION|RECOMMENDATION)[S]?\s*:.*$',
            '', cleaned, flags=re.IGNORECASE
        ).strip()
        return cleaned
    return ''


def extract_all_sections(report_path):
    """
    Read a .txt report and extract FINDINGS, INDICATION, and IMPRESSION sections.
    Returns dict with keys: findings, indication, impression
    """
    if not os.path.exists(report_path):
        return {'findings': '', 'indication': '', 'impression': ''}

    with open(report_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    findings = extract_section(text, [
        'FINDINGS', 'FINDING'
    ])
    indication = extract_section(text, [
        'INDICATION', 'HISTORY', 'CLINICAL INFORMATION',
        'CLINICAL INDICATION', 'REASON FOR EXAM', 'REASON FOR EXAMINATION'
    ])
    impression = extract_section(text, [
        'IMPRESSION', 'IMPRESSIONS', 'CONCLUSION', 'CONCLUSIONS'
    ])

    return {
        'findings'  : findings   or 'No findings documented.',
        'indication': indication or 'No indication documented.',
        'impression': impression or 'No impression documented.',
    }


# ── Main ──────────────────────────────────────────────────────────────────────
print(f'Loading: {CSV_V2}')
df = pd.read_csv(CSV_V2)
print(f'Rows: {len(df)}')

findings_list   = []
indication_list = []
impression_list = []

n_no_findings = 0

# Fix report_path: v2 CSV has old mimic_pilot path, actual files are in mimic_pilot_139
OLD_REPORT = r'C:\2026\PneumoFusionNet\mimic\mimic_pilot\reports\txt'
NEW_REPORT = REPORT_DIR
df['report_path'] = df['report_path'].str.replace(OLD_REPORT, NEW_REPORT, regex=False)

for idx, row in df.iterrows():
    rp = str(row['report_path'])
    sections = extract_all_sections(rp)

    findings_list.append(sections['findings'])
    indication_list.append(sections['indication'])
    impression_list.append(sections['impression'])

    if sections['findings'] == 'No findings documented.':
        n_no_findings += 1
        print(f'  [WARN] No FINDINGS in: {os.path.basename(rp)}')

# Build v3
df['impression_orig'] = df['impression']       # keep old IMPRESSION for reference
df['impression']      = impression_list         # re-extracted (same data, cleaner)
df['findings']        = findings_list           # ← use this for Phase 2 training
df['indication']      = indication_list         # ← background clinical context

df.to_csv(CSV_OUT, index=False)

print('=' * 60)
print('DATASET v3 BUILT')
print('=' * 60)
print(f'Total rows     : {len(df)}')
print(f'Missing FINDINGS: {n_no_findings}')
print(f'Saved to       : {CSV_OUT}')
print()

# ── Show leakage comparison examples ──────────────────────────────────────────
print('LEAKAGE COMPARISON (first 5 rows)')
print('=' * 60)
for _, row in df.head(5).iterrows():
    print(f'\n[{row["label_name"]:9s}] study_id={row["study_id"]}')
    print(f'  INDICATION : {str(row["indication"])[:100]}')
    print(f'  FINDINGS   : {str(row["findings"])[:100]}')
    print(f'  IMPRESSION : {str(row["impression"])[:100]}')

print('\nRecommendation for Phase 2:')
print('  -> Use "findings" column -- observational, less conclusory than IMPRESSION')
print('  -> "indication" can be appended as additional context')
