import pandas as pd

# Load the 1,000 subset CSV
print("Loading subset cohort...")
df = pd.read_csv('1mimic_subset_1000.csv')
subset_studies = set(df['study_id'].astype(str))

matched = []
print("Scanning IMAGE_FILENAMES to match study IDs...")
# Load the full manifest and extract paths matching the subset
with open('IMAGE_FILENAMES', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('/')
        if len(parts) >= 4:
            study_part = parts[3]
            if study_part.startswith('s'):
                study_id = study_part[1:]
                if study_id in subset_studies:
                    matched.append(line)

# Write matching paths to a text file
output_file = 'subset_image_paths.txt'
with open(output_file, 'w') as f_out:
    f_out.write('\n'.join(matched))

print(f"Done! Successfully wrote {len(matched)} image paths to: {output_file}")
