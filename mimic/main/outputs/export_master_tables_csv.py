r"""
export_master_tables_csv.py
--------------------------
Exports all experimental results and comparison tables to CSV files
so the user can send them directly to the professor or open them in Excel.
"""

import os
import pandas as pd

OUT_DIR = r"C:\Users\ashutosh\.gemini\antigravity\brain\7fe6dfc2-5088-4163-b4e1-5291a55d3093"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. TABLE 1: Comprehensive Modality Progression across both cohorts
table1_data = [
    {
        "Phase": "Phase 1 (Image Only)",
        "Cohort": "Half (~1,854)",
        "Inputs": "Chest X-Ray (PA view)",
        "Architecture": "DenseNet-121 + CBAM (5-fold CV)",
        "ROC_AUC": 0.8152,
        "ROC_AUC_95CI": "[0.798, 0.832]",
        "Accuracy": 0.7600,
        "Sensitivity": 0.6974,
        "Specificity": 0.8160,
        "Threshold_Rule": "Tuned per fold",
        "Clinical_Turnaround": "5-10 mins"
    },
    {
        "Phase": "Phase 1 (Image Only)",
        "Cohort": "Scaleup (~3,763)",
        "Inputs": "Chest X-Ray (PA view)",
        "Architecture": "DenseNet-121 + CBAM (5-fold CV)",
        "ROC_AUC": 0.8258,
        "ROC_AUC_95CI": "[0.812, 0.840]",
        "Accuracy": 0.7626,
        "Sensitivity": 0.7103,
        "Specificity": 0.8160,
        "Threshold_Rule": "Tuned per fold",
        "Clinical_Turnaround": "5-10 mins"
    },
    {
        "Phase": "Phase 2v2 (Multimodal)",
        "Cohort": "Half (~1,989)",
        "Inputs": "CXR + Radiology Text",
        "Architecture": "Bio_ClinicalBERT + 8-Head Cross-Attention",
        "ROC_AUC": 0.9490,
        "ROC_AUC_95CI": "[0.938, 0.960]",
        "Accuracy": 0.8863,
        "Sensitivity": 0.9137,
        "Specificity": 0.8625,
        "Threshold_Rule": "Youden-J (tau=0.499)",
        "Clinical_Turnaround": "15-20 mins"
    },
    {
        "Phase": "Phase 2v2 (Multimodal)",
        "Cohort": "Scaleup (~3,763)",
        "Inputs": "CXR + Radiology Text",
        "Architecture": "Bio_ClinicalBERT + 8-Head Cross-Attention",
        "ROC_AUC": 0.9460,
        "ROC_AUC_95CI": "[0.935, 0.957]",
        "Accuracy": 0.8780,
        "Sensitivity": 0.9030,
        "Specificity": 0.8905,
        "Threshold_Rule": "Youden-J (tau=0.568)",
        "Clinical_Turnaround": "15-20 mins"
    },
    {
        "Phase": "Phase 3 (Triple Fusion - 17 Feats)",
        "Cohort": "Scaleup (~3,763)",
        "Inputs": "CXR + Text + 17 Clinical Variables",
        "Architecture": "Cross-Attn + MLP(17->128->64)",
        "ROC_AUC": 0.9692,
        "ROC_AUC_95CI": "[0.960, 0.978]",
        "Accuracy": 0.9170,
        "Sensitivity": 0.8910,
        "Specificity": 0.9430,
        "Threshold_Rule": "Youden-J (tau=0.550)",
        "Clinical_Turnaround": "1-4 hours (vitals/labs)"
    },
    {
        "Phase": "Phase 3 (Triple Fusion - 11 Feats)",
        "Cohort": "Scaleup (~3,763)",
        "Inputs": "CXR + Text + 11 Clean Features",
        "Architecture": "Cross-Attn + MLP(11->128->64)",
        "ROC_AUC": 0.9718,
        "ROC_AUC_95CI": "[0.963, 0.980]",
        "Accuracy": 0.9204,
        "Sensitivity": 0.8838,
        "Specificity": 0.9573,
        "Threshold_Rule": "Youden-J (tau=0.532)",
        "Clinical_Turnaround": "1-2 hours"
    },
    {
        "Phase": "Phase 3c (Point-of-Care Main Model)",
        "Cohort": "Scaleup (~3,763)",
        "Inputs": "CXR + Text + WBC Count Only",
        "Architecture": "Cross-Attn + WBC MLP(1->128->128->64)",
        "ROC_AUC": 0.9711,
        "ROC_AUC_95CI": "[0.963, 0.978]",
        "Accuracy": 0.9292,
        "Sensitivity": 0.9225,
        "Specificity": 0.9359,
        "Threshold_Rule": "Youden-J Optimal (tau=0.559)",
        "Clinical_Turnaround": "< 30 mins (STAT CBC)"
    },
]
df_table1 = pd.DataFrame(table1_data)
df_table1.to_csv(os.path.join(OUT_DIR, "Table1_Modality_Progression_Master.csv"), index=False)

# 2. TABLE 2: Phase 1 Fold-by-Fold Cross-Validation Metrics
table2_data = [
    {"Cohort": "Half (~1,854)", "Fold": "Fold 1", "Train_N": 1190, "Val_N": 293, "Test_N": 371, "Val_AUC": 0.8293, "Test_Acc": 0.7035, "Test_AUC": 0.7951, "TTA_AUC": 0.7981, "Tuned_Acc": 0.7439, "Pn_Recall": 0.6410},
    {"Cohort": "Half (~1,854)", "Fold": "Fold 2", "Train_N": 1189, "Val_N": 294, "Test_N": 371, "Val_AUC": 0.8043, "Test_Acc": 0.7574, "Test_AUC": 0.8411, "TTA_AUC": 0.8445, "Tuned_Acc": 0.7763, "Pn_Recall": 0.7950},
    {"Cohort": "Half (~1,854)", "Fold": "Fold 3", "Train_N": 1189, "Val_N": 294, "Test_N": 371, "Val_AUC": 0.7740, "Test_Acc": 0.7547, "Test_AUC": 0.8355, "TTA_AUC": 0.8349, "Tuned_Acc": 0.7682, "Pn_Recall": 0.7060},
    {"Cohort": "Half (~1,854)", "Fold": "Fold 4", "Train_N": 1189, "Val_N": 294, "Test_N": 371, "Val_AUC": 0.8332, "Test_Acc": 0.7278, "Test_AUC": 0.8112, "TTA_AUC": 0.8204, "Tuned_Acc": 0.7655, "Pn_Recall": 0.7360},
    {"Cohort": "Half (~1,854)", "Fold": "Fold 5", "Train_N": 1189, "Val_N": 294, "Test_N": 370, "Val_AUC": 0.7604, "Test_Acc": 0.7027, "Test_AUC": 0.7933, "TTA_AUC": 0.8001, "Tuned_Acc": 0.7459, "Pn_Recall": 0.6080},
    {"Cohort": "Half (~1,854)", "Fold": "MEAN",   "Train_N": 1189, "Val_N": 294, "Test_N": 371, "Val_AUC": 0.8002, "Test_Acc": 0.7292, "Test_AUC": 0.8152, "TTA_AUC": 0.8196, "Tuned_Acc": 0.7600, "Pn_Recall": 0.6974},
    {"Cohort": "Scaleup (~3,763)", "Fold": "Fold 1", "Train_N": 2408, "Val_N": 602, "Test_N": 753, "Val_AUC": 0.8651, "Test_Acc": 0.6796, "Test_AUC": 0.8206, "TTA_AUC": 0.8270, "Tuned_Acc": 0.7637, "Pn_Recall": 0.7840},
    {"Cohort": "Scaleup (~3,763)", "Fold": "Fold 2", "Train_N": 2408, "Val_N": 602, "Test_N": 753, "Val_AUC": 0.8338, "Test_Acc": 0.7236, "Test_AUC": 0.8088, "TTA_AUC": 0.8105, "Tuned_Acc": 0.7423, "Pn_Recall": 0.7220},
    {"Cohort": "Scaleup (~3,763)", "Fold": "Fold 3", "Train_N": 2408, "Val_N": 602, "Test_N": 753, "Val_AUC": 0.8288, "Test_Acc": 0.7543, "Test_AUC": 0.8365, "TTA_AUC": 0.8392, "Tuned_Acc": 0.7757, "Pn_Recall": 0.6830},
    {"Cohort": "Scaleup (~3,763)", "Fold": "Fold 4", "Train_N": 2408, "Val_N": 602, "Test_N": 753, "Val_AUC": 0.8473, "Test_Acc": 0.6622, "Test_AUC": 0.8133, "TTA_AUC": 0.8166, "Tuned_Acc": 0.7610, "Pn_Recall": 0.6540},
    {"Cohort": "Scaleup (~3,763)", "Fold": "Fold 5", "Train_N": 2409, "Val_N": 602, "Test_N": 752, "Val_AUC": 0.8210, "Test_Acc": 0.7380, "Test_AUC": 0.8500, "TTA_AUC": 0.8515, "Tuned_Acc": 0.7701, "Pn_Recall": 0.7090},
    {"Cohort": "Scaleup (~3,763)", "Fold": "MEAN",   "Train_N": 2408, "Val_N": 602, "Test_N": 753, "Val_AUC": 0.8392, "Test_Acc": 0.7115, "Test_AUC": 0.8258, "TTA_AUC": 0.8290, "Tuned_Acc": 0.7626, "Pn_Recall": 0.7103},
]
df_table2 = pd.DataFrame(table2_data)
df_table2.to_csv(os.path.join(OUT_DIR, "Table2_Phase1_CrossVal_Breakdown.csv"), index=False)

# 3. TABLE 3: Phase 3c Clinical Operating Threshold Deployments
table3_data = [
    {
        "Strategy": "Emergency Triage (Default)",
        "Threshold_tau": 0.5000,
        "Accuracy": 0.9239,
        "Sensitivity": 0.9472,
        "Specificity": 0.9004,
        "Clinical_Goal": "High-Sensitivity Screening: Ensures < 5.3% false negatives in emergency departments."
    },
    {
        "Strategy": "Balanced Operating (Youden-J)",
        "Threshold_tau": 0.5590,
        "Accuracy": 0.9292,
        "Sensitivity": 0.9225,
        "Specificity": 0.9359,
        "Clinical_Goal": "Optimal Diagnostic Utility: Maximises J = TPR - FPR with 92.9% overall accuracy."
    },
    {
        "Strategy": "Clinical Specificity Target",
        "Threshold_tau": 0.5752,
        "Accuracy": 0.9274,
        "Sensitivity": 0.9120,
        "Specificity": 0.9431,
        "Clinical_Goal": "False-Positive Reduction: Reaches 94.3% specificity to curb antibiotic overuse."
    },
]
df_table3 = pd.DataFrame(table3_data)
df_table3.to_csv(os.path.join(OUT_DIR, "Table3_Phase3c_Operating_Thresholds.csv"), index=False)

# 4. TABLE 4: Feature Panel Ablation Analysis
table4_data = [
    {
        "Configuration": "Phase 1: CXR Image Only",
        "Features_Included": "Radiograph pixels (224x224)",
        "Missingness": "0.0%",
        "Turnaround_Time": "5 - 10 min",
        "Test_AUC": 0.8258,
        "Sensitivity": 0.7103,
        "Specificity": 0.8160,
        "Clinical_Tradeoff": "Baseline visual triage; misses atypical presentations."
    },
    {
        "Configuration": "Phase 2v2: CXR + Report Text",
        "Features_Included": "CXR + FINDINGS/HISTORY tokens",
        "Missingness": "0.0%",
        "Turnaround_Time": "15 - 20 min",
        "Test_AUC": 0.9460,
        "Sensitivity": 0.9030,
        "Specificity": 0.8905,
        "Clinical_Tradeoff": "Major AUC jump (+12 pp); dependent on preliminary radiologist notes."
    },
    {
        "Configuration": "Phase 3: CXR + Text + 17 EHR Feats",
        "Features_Included": "CXR + Text + 6 vitals + 11 lab/demog",
        "Missingness": "78.3% (vitals absent in non-ICU)",
        "Turnaround_Time": "1 - 4 hours",
        "Test_AUC": 0.9692,
        "Sensitivity": 0.8910,
        "Specificity": 0.9430,
        "Clinical_Tradeoff": "Imputation of 78% vitals creates flat noise; delays decision by hours."
    },
    {
        "Configuration": "Phase 3: CXR + Text + 11 Lab Feats",
        "Features_Included": "CXR + Text + Age/Sex/Mort + 7 Labs",
        "Missingness": "< 20% (lab coverage ~80%)",
        "Turnaround_Time": "1 - 2 hours",
        "Test_AUC": 0.9718,
        "Sensitivity": 0.8838,
        "Specificity": 0.9573,
        "Clinical_Tradeoff": "Removes vital noise; high specificity, but still requires full lab workup."
    },
    {
        "Configuration": "Phase 3c: CXR + Text + WBC Point-of-Care",
        "Features_Included": "CXR + Text + Single WBC Scalar",
        "Missingness": "15.0% (train-median imputed)",
        "Turnaround_Time": "< 30 min (STAT CBC)",
        "Test_AUC": 0.9711,
        "Sensitivity": 0.9225,
        "Specificity": 0.9359,
        "Clinical_Tradeoff": "Highest sensitivity (92.3%-94.7%); available at bed-side within 30 min."
    },
]
df_table4 = pd.DataFrame(table4_data)
df_table4.to_csv(os.path.join(OUT_DIR, "Table4_Clinical_Feature_Ablation.csv"), index=False)

print("Master CSV Tables Exported Successfully:")
print(f"1. {os.path.join(OUT_DIR, 'Table1_Modality_Progression_Master.csv')}")
print(f"2. {os.path.join(OUT_DIR, 'Table2_Phase1_CrossVal_Breakdown.csv')}")
print(f"3. {os.path.join(OUT_DIR, 'Table3_Phase3c_Operating_Thresholds.csv')}")
print(f"4. {os.path.join(OUT_DIR, 'Table4_Clinical_Feature_Ablation.csv')}")
