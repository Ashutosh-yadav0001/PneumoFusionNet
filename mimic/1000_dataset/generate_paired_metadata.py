import os
import pandas as pd
from tqdm import tqdm

def main():
    print("Starting metadata pairing process...")
    
    # Define paths
    base_dir = "restricted_Dowloaded_dataset"
    images_dir = os.path.join(base_dir, "image_subset")
    reports_dir = os.path.join(base_dir, "subset_reports")
    metadata_csv_path = os.path.join(base_dir, "mimic-cxr-2.0.0-metadata.csv")
    chexpert_csv_path = "1mimic_subset_1000_images.csv"
    output_csv_path = "mimic_paired_dataset.csv"
    
    # Check if files/directories exist
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        return
    if not os.path.exists(metadata_csv_path):
        print(f"Error: Metadata CSV not found at {metadata_csv_path}")
        return
        
    # Load metadata CSV
    print("Loading MIMIC-CXR metadata CSV...")
    df_meta = pd.read_csv(metadata_csv_path)
    print(f"Loaded metadata for {len(df_meta):,} images.")
    
    # Index metadata by dicom_id for fast lookup
    df_meta.set_index("dicom_id", inplace=True)
    
    # Load chexpert labels if available to propagate Pneumonia and No Finding labels
    chexpert_labels = {}
    if os.path.exists(chexpert_csv_path):
        print(f"Loading chexpert labels from {chexpert_csv_path}...")
        df_labels = pd.read_csv(chexpert_csv_path)
        # Create a lookup dictionary using (subject_id, study_id) as key
        for _, row in df_labels.iterrows():
            sub_id = int(row["subject_id"])
            std_id = int(row["study_id"])
            chexpert_labels[(sub_id, std_id)] = {
                "Pneumonia": row.get("Pneumonia", None),
                "No Finding": row.get("No Finding", None)
            }
    
    # List all images in image_subset
    print("Scanning image subset...")
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images in {images_dir}.")
    
    records = []
    missing_metadata = 0
    missing_reports = 0
    
    for filename in tqdm(image_files):
        # Extract dicom_id (which is the filename without extension)
        dicom_id, _ = os.path.splitext(filename)
        
        # Look up in metadata
        if dicom_id not in df_meta.index:
            missing_metadata += 1
            continue
            
        row_meta = df_meta.loc[dicom_id]
        
        # If there are duplicate dicom_ids, grab the first one
        if isinstance(row_meta, pd.DataFrame):
            row_meta = row_meta.iloc[0]
            
        subject_id = int(row_meta["subject_id"])
        study_id = int(row_meta["study_id"])
        view_position = row_meta["ViewPosition"]
        
        # Build report path
        # Reports are in: restricted_Dowloaded_dataset/subset_reports/pXX/pXXXXXXXX/sXXXXXXXX.txt
        sub_id_str = f"{subject_id}"
        prefix = f"p{sub_id_str[:2]}"
        report_filename = f"s{study_id}.txt"
        
        # Relative report path
        rel_report_path = os.path.join(reports_dir, prefix, f"p{subject_id}", report_filename)
        
        # Standardize path separators to forward slashes for cross-platform compatibility
        rel_report_path_normalized = rel_report_path.replace("\\", "/")
        rel_image_path = os.path.join(images_dir, filename).replace("\\", "/")
        
        # Check if report exists
        report_exists = os.path.exists(rel_report_path)
        if not report_exists:
            missing_reports += 1
            report_path_val = None
        else:
            report_path_val = rel_report_path_normalized
            
        # Get labels from CheXpert mapping if available
        pneumonia = None
        no_finding = None
        if (subject_id, study_id) in chexpert_labels:
            pneumonia = chexpert_labels[(subject_id, study_id)]["Pneumonia"]
            no_finding = chexpert_labels[(subject_id, study_id)]["No Finding"]
            
        records.append({
            "patient_id": f"p{subject_id}",
            "subject_id": subject_id,
            "study_id": study_id,
            "image_name": filename,
            "image_path": rel_image_path,
            "report_path": report_path_val,
            "view_position": view_position,
            "pneumonia_label": pneumonia,
            "no_finding_label": no_finding
        })
        
    df_output = pd.DataFrame(records)
    df_output.to_csv(output_csv_path, index=False)
    
    print("\nProcessing complete!")
    print(f"Total processed images: {len(records)}")
    print(f"Saved dataset CSV to: {output_csv_path}")
    print(f"Missing metadata in CSV: {missing_metadata}")
    print(f"Missing reports locally: {missing_reports}")
    
if __name__ == "__main__":
    main()
