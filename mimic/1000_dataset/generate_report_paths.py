import pandas as pd

# Load the 1,000 subset CSV
print("Loading subset cohort...")
df = pd.read_csv('1mimic_subset_1000.csv')

report_paths = []
print("Generating report file paths...")
for _, row in df.iterrows():
    sub_id = str(int(row['subject_id']))
    std_id = str(int(row['study_id']))
    prefix = sub_id[:2]
    
    # Reports are located at: files/pXX/pXXXXXXXX/sXXXXXXXX.txt
    path = f"files/p{prefix}/p{sub_id}/s{std_id}.txt"
    report_paths.append(path)

# Write to a text file
output_file = 'subset_report_paths.txt'
with open(output_file, 'w') as f_out:
    f_out.write('\n'.join(report_paths))

print(f"Done! Successfully wrote {len(report_paths)} report paths to: {output_file}")
