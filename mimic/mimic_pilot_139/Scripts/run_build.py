import os, re, zipfile, pandas as pd

BASE_DIR    = r'C:\2026\PneumoFusionNet\mimic'
CSV_PATH    = os.path.join(BASE_DIR, 'mimic_pilot', 'mimic_dataset.csv')
ZIP_PATH    = os.path.join(BASE_DIR, 'mimic_pilot', 'reports', 'mimic-cxr-reports.zip')
REPORTS_DIR = os.path.join(BASE_DIR, 'mimic_pilot', 'reports', 'txt')
OUT_PATH    = os.path.join(BASE_DIR, 'mimic_pilot', 'mimic_multimodal_dataset_v2.csv')

os.makedirs(REPORTS_DIR, exist_ok=True)
df = pd.read_csv(CSV_PATH)

def extract_impression(text):
    m = re.search(r'IMPRESSION[S]?\s*:?\s*\n?(.*?)(?:\n\s*\n|\Z)', text, re.IGNORECASE|re.DOTALL)
    if m:
        return re.sub(r'\s+', ' ', m.group(1).strip())
    parts = [p.strip() for p in text.split('\n\n') if p.strip()]
    return re.sub(r'\s+', ' ', parts[-1]) if parts else text[:300]

report_paths = []
impressions  = []
found = 0
not_found = 0

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    zset = set(z.namelist())
    for _, row in df.iterrows():
        pid = int(row['subject_id'])
        sid = int(row['study_id'])
        prefix = str(pid)[:2]
        zip_key = "files/p" + prefix + "/p" + str(pid) + "/s" + str(sid) + ".txt"
        save_path = os.path.join(REPORTS_DIR, "p" + str(pid) + "_s" + str(sid) + ".txt")

        if zip_key in zset:
            if not os.path.exists(save_path):
                with z.open(zip_key) as f:
                    content = f.read().decode('utf-8', 'replace')
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

df['report_path'] = report_paths
df['impression']  = impressions
df_final = df[df['report_path'] != ''].reset_index(drop=True)
df_final.to_csv(OUT_PATH, index=False)

print("Found    :", found, "/", len(df))
print("Saved to :", OUT_PATH)
print("Columns  :", list(df_final.columns))
print()
r = df_final.iloc[0]
print("image_path  :", r['image_path'])
print("report_path :", r['report_path'])
print("label       :", r['label_name'])
print("impression  :", r['impression'])
