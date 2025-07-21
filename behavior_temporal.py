"""
This script temporally evaluates a Random Forest model to predict all-cause
mortality using NHANES 1999–2018 data and public-use Linked Mortality Files.
Author: Matthew Sanders
"""

# Expected structure: raw NHANES .xpt files and MORTSTAT .dat files in ./data/
# Output: evaluation metrics and ROC curves saved in ./results/

import pandas as pd
import os
os.makedirs("results", exist_ok=True)
from joblib import load
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss

DATA_DIR = "data"
suffixes = ["_F", "_G", "_H", "_I", "_J"]  # 2009–2018 cycles

demo_frames = []

# Load behavioral NHANES files
for idx, suffix in enumerate(suffixes):
        demo = pd.read_sas(os.path.join(DATA_DIR, f"DEMO{suffix}.xpt"), format="xport")
        smq  = pd.read_sas(os.path.join(DATA_DIR, f"SMQ{suffix}.xpt"), format="xport")
        alq  = pd.read_sas(os.path.join(DATA_DIR, f"ALQ{suffix}.xpt"), format="xport")
        paq  = pd.read_sas(os.path.join(DATA_DIR, f"PAQ{suffix}.xpt"), format="xport")
        bmx  = pd.read_sas(os.path.join(DATA_DIR, f"BMX{suffix}.xpt"), format="xport")

        merged = demo.merge(smq, on="SEQN", how="inner") \
                     .merge(alq, on="SEQN", how="inner") \
                     .merge(paq, on="SEQN", how="inner") \
                     .merge(bmx, on="SEQN", how="inner")

        merged["cycle_year"] = 1999 + 2 * (idx + 5)  # suffix _F = cycle 2009
        demo_frames.append(merged)
    
full_df = pd.concat(demo_frames, ignore_index=True)

# Build feature DataFrame
model_df = pd.DataFrame({
    "SEQN": full_df["SEQN"],
    "age": full_df["RIDAGEYR"],
    "sex": full_df["RIAGENDR"],
    "race_ethnicity": full_df["RIDRETH1"],
    "education_level": full_df["DMDEDUC2"],
    "income_poverty_ratio": full_df["INDFMPIR"],
    "current_smoker": full_df["SMQ040"],
    "ever_smoked": full_df["SMQ020"],
    "alcohol_freq": full_df["ALQ130"],
    "bmi": full_df["BMXBMI"],
    "cycle_year": full_df["cycle_year"]
})
model_df = model_df[model_df["age"] >= 20]

# Harmonize binary encodings
model_df["ever_smoked"] = model_df["ever_smoked"].replace({1: 1, 2: 0})

# Clean current_smoker (SMQ040): 1=Yes, 2=No, 3=Some days → 1
model_df["current_smoker"] = model_df["current_smoker"].replace({1: 1, 2: 0, 3: 1})

# Drop refused (7) or don't know (9)
model_df = model_df[model_df["current_smoker"].isin([0, 1])]

# Remove invalid alcohol frequency responses (777 = refused, 999 = don't know)
model_df = model_df[model_df["alcohol_freq"] < 777]

# Load and process mortality linkage
mort_files = [
    "NHANES_2009_2010_MORT_2019_PUBLIC.dat",
    "NHANES_2011_2012_MORT_2019_PUBLIC.dat",
    "NHANES_2013_2014_MORT_2019_PUBLIC.dat",
    "NHANES_2015_2016_MORT_2019_PUBLIC.dat",
    "NHANES_2017_2018_MORT_2019_PUBLIC.dat",
]

mort_dfs = []
colspecs = [(0, 5), (14, 15), (15, 16)]
colnames = ["SEQN", "ELIGSTAT", "MORTSTAT"]

for fname in mort_files:
        df = pd.read_fwf(os.path.join(DATA_DIR, fname), colspecs=colspecs, names=colnames)
        mort_dfs.append(df)

mortality_df = pd.concat(mort_dfs, ignore_index=True)
mortality_df["ELIGSTAT"] = mortality_df["ELIGSTAT"].astype(str).str.strip()
mortality_df = mortality_df[mortality_df["ELIGSTAT"] == "1"]
mortality_df = mortality_df[mortality_df["MORTSTAT"].isin(["0", "1"])]
mortality_df["MORTSTAT"] = pd.to_numeric(mortality_df["MORTSTAT"], errors="coerce")

# Merge and clean
model_df["SEQN"] = model_df["SEQN"].astype(int)
mortality_df["SEQN"] = mortality_df["SEQN"].astype(int)
model_df = model_df.merge(mortality_df[["SEQN", "MORTSTAT"]], on="SEQN", how="inner")
model_df["mortality"] = (model_df["MORTSTAT"] == 1).astype(int)
model_df.drop(columns=["MORTSTAT"], inplace=True)

# Final features
features = [
    "age", "sex", "race_ethnicity", "education_level", "income_poverty_ratio",
    "current_smoker", "alcohol_freq", "bmi"
]

model_df = model_df.dropna(subset=features + ["mortality"])
X_test = model_df[features]
y_test = model_df["mortality"]

# Load model and predict
clf = load("results/behavior_model.joblib")
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("\nTemporal Test Results (2009–2018):")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# Brier Score
brier = brier_score_loss(y_test, y_prob)
print(f"Brier Score: {brier:.4f}")

with open("results/brier_score.txt", "a") as f:
    f.write(f"Temporal Brier Score: {brier:.4f}\n")

# Confusion Matrix
from sklearn.metrics import confusion_matrix, roc_curve
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix:\n{cm}")
print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")

# ROC Curve CSV export
fpr, tpr, _ = roc_curve(y_test, y_prob)
pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv("results/roc_test.csv", index=False)

# Summary statistics by outcome
model_df["mortality"] = y_test  # Just to ensure it’s grouped properly
model_df[features + ["mortality"]].groupby("mortality").describe().to_csv("results/descriptive_stats_by_outcome_test.csv")
