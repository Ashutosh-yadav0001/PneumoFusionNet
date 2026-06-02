"""
MIMIC-CXR Pilot — Build Dataset CSV
=====================================
Scans all JPG images in MIMIC_CXR_JPG_P1/files/p10/
Links each study to CheXpert binary labels:
  Pneumonia = 1.0  → label 1
  No Finding = 1.0 → label 0
  All others       → excluded

Output: mimic_pilot/mimic_dataset.csv
"""

import csv, os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P10_DIR    = os.path.join(BASE_DIR, "MIMIC_CXR_JPG_P1", "p10")
CHEXPERT   = os.path.join(BASE_DIR, "mimic-cxr-2.0.0-chexpert.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "mimic_pilot", "mimic_dataset.csv")

print("=" * 55)
print("  Building MIMIC-CXR Dataset CSV")
print("=" * 55)

# 1. Load CheXpert labels
print("Loading CheXpert labels...")
chexpert = {}
with open(CHEXPERT, "r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        key = (row["subject_id"].strip(), row["study_id"].strip())
        chexpert[key] = row
print(f"  {len(chexpert):,} studies loaded")

# 2. Scan P10 folder
print("Scanning images...")
records, seen = [], set()
stats = {"pneumonia": 0, "normal": 0, "excluded": 0, "no_chex": 0}

for patient in sorted(os.listdir(P10_DIR)):
    ppath = os.path.join(P10_DIR, patient)
    if not os.path.isdir(ppath): continue
    subject_id = patient[1:]  # strip 'p'

    for study in sorted(os.listdir(ppath)):
        spath = os.path.join(ppath, study)
        if not os.path.isdir(spath): continue
        study_id = study[1:]  # strip 's'
        key = (subject_id, study_id)
        if key in seen: continue

        jpgs = sorted([f for f in os.listdir(spath) if f.endswith(".jpg")])
        if not jpgs: continue

        if key not in chexpert:
            stats["no_chex"] += 1
            continue

        row    = chexpert[key]
        pneum  = row.get("Pneumonia",  "").strip()
        nofind = row.get("No Finding", "").strip()

        if pneum == "1.0":
            label, label_name = 1, "Pneumonia"
            stats["pneumonia"] += 1
        elif nofind == "1.0":
            label, label_name = 0, "Normal"
            stats["normal"] += 1
        else:
            stats["excluded"] += 1
            continue

        seen.add(key)
        records.append({
            "subject_id": subject_id,
            "study_id":   study_id,
            "image_path": os.path.join(spath, jpgs[0]),
            "label":      label,
            "label_name": label_name,
        })

# 3. Save
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["subject_id","study_id","image_path","label","label_name"])
    w.writeheader()
    w.writerows(records)

print()
print("=" * 55)
print("  DATASET SUMMARY")
print("=" * 55)
print(f"  Pneumonia (label=1) : {stats['pneumonia']}")
print(f"  Normal    (label=0) : {stats['normal']}")
print(f"  Excluded (uncertain): {stats['excluded']}")
print(f"  Not in CheXpert     : {stats['no_chex']}")
print(f"  Total saved         : {len(records)}")
print(f"  Positive rate       : {stats['pneumonia']/max(len(records),1)*100:.1f}%")
print(f"  Saved to            : {OUTPUT_CSV}")
print("=" * 55)
