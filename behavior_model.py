"""
This script trains a Random Forest model to predict all-cause mortality 
using NHANES 1999–2008 data and public-use Linked Mortality Files.
Author: Matthew Sanders
"""

# Expected structure: raw NHANES .xpt files and MORTSTAT .dat files in ./data/
# Output: evaluation metrics and ROC curves saved in ./results/

import pandas as pd
import os
os.makedirs("results", exist_ok=True)
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
suffixes = ["", "_B", "_C", "_D", "_E"]

demo_frames = []

# Load and merge core NHANES datasets

for idx, suffix in enumerate(suffixes):
    
        demo = pd.read_sas(os.path.join(DATA_DIR, f"DEMO{suffix}.xpt"), format="xport")
        smq  = pd.read_sas(os.path.join(DATA_DIR, f"SMQ{suffix}.xpt"), format="xport")
        alq  = pd.read_sas(os.path.join(DATA_DIR, f"ALQ{suffix}.xpt"), format="xport")
        paq  = pd.read_sas(os.path.join(DATA_DIR, f"PAQ{suffix}.xpt"), format="xport")
        bmx  = pd.read_sas(os.path.join(DATA_DIR, f"BMX{suffix}.xpt"), format="xport")

        # Inner join on SEQN to keep only complete cases
        merged = demo.merge(smq, on="SEQN", how="inner") \
                     .merge(alq, on="SEQN", how="inner") \
                     .merge(paq, on="SEQN", how="inner") \
                     .merge(bmx, on="SEQN", how="inner")

        merged["cycle_year"] = 1999 + 2 * idx
        demo_frames.append(merged)

# Combine all cycles
full_df = pd.concat(demo_frames, ignore_index=True)

# Clean and extract features

model_df = pd.DataFrame({
    "SEQN": full_df["SEQN"],
    "age": full_df["RIDAGEYR"],
    "sex": full_df["RIAGENDR"],
    "race_ethnicity": full_df["RIDRETH1"],
    "education_level": full_df["DMDEDUC2"] if "DMDEDUC2" in full_df.columns else pd.NA,
    "income_poverty_ratio": full_df["INDFMPIR"] if "INDFMPIR" in full_df.columns else pd.NA,
    "current_smoker": full_df["SMQ040"] if "SMQ040" in full_df.columns else pd.NA,
    "ever_smoked": full_df["SMQ020"] if "SMQ020" in full_df.columns else pd.NA,
    "alcohol_freq": full_df["ALQ130"] if "ALQ130" in full_df.columns else pd.NA,
    "alcohol_binge": full_df["ALQ151"] if "ALQ151" in full_df.columns else pd.NA,
    "vigorous_activity": full_df["PAQ605"] if "PAQ605" in full_df.columns else pd.NA,
    "moderate_activity": full_df["PAQ620"] if "PAQ620" in full_df.columns else pd.NA,
    "recreational_vigorous": full_df["PAQ635"] if "PAQ635" in full_df.columns else pd.NA,
    "recreational_moderate": full_df["PAQ650"] if "PAQ650" in full_df.columns else pd.NA,
    "bmi": full_df["BMXBMI"] if "BMXBMI" in full_df.columns else pd.NA,
    "cycle_year": full_df["cycle_year"]
})
model_df = model_df[model_df["age"] >= 20]

# Drop rows missing only the core predictors
required_cols = ["age", "sex", "race_ethnicity", "education_level", "income_poverty_ratio", "current_smoker"]
model_df = model_df.dropna(subset=required_cols)

# Harmonize categorical features (1=Yes, 2=No)
binary_vars = [
    'ever_smoked', 'alcohol_binge',
    'vigorous_activity', 'moderate_activity',
    'recreational_vigorous', 'recreational_moderate'
]

for var in binary_vars:
    model_df[var] = model_df[var].replace({1: 1, 2: 0})

# Clean current_smoker (SMQ040): 1=Yes, 2=No, 3=Some days → 1
model_df["current_smoker"] = model_df["current_smoker"].replace({1: 1, 2: 0, 3: 1})

# Drop refused (7) or don't know (9)
model_df = model_df[model_df["current_smoker"].isin([0, 1])]

# Remove invalid alcohol frequency responses (777 = refused, 999 = don't know)
model_df = model_df[model_df["alcohol_freq"] < 777]

# Load and merge mortality files
mort_files = [
    "NHANES_1999_2000_MORT_2019_PUBLIC.dat",
    "NHANES_2001_2002_MORT_2019_PUBLIC.dat",
    "NHANES_2003_2004_MORT_2019_PUBLIC.dat",
    "NHANES_2005_2006_MORT_2019_PUBLIC.dat",
    "NHANES_2007_2008_MORT_2019_PUBLIC.dat",
    ]

mort_dfs = []
colspecs = [(0, 5), (14, 15), (15, 16)]
colnames = ["SEQN", "ELIGSTAT", "MORTSTAT"]

for fname in mort_files:
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_fwf(path, colspecs=colspecs, names=colnames)
    mort_dfs.append(df)

mortality_df = pd.concat(mort_dfs, ignore_index=True)

# Keep only eligible participants
mortality_df["ELIGSTAT"] = mortality_df["ELIGSTAT"].astype(str).str.strip()
mortality_df = mortality_df[mortality_df["ELIGSTAT"] == '1']

# Keep only rows with mortality status 0 or 1
mortality_df = mortality_df[mortality_df["MORTSTAT"].isin(['0', '1'])]

# Convert MORTSTAT to integer
mortality_df["MORTSTAT"] = mortality_df["MORTSTAT"].astype(int)

# Convert MORTSTAT: '.' → NaN, '0'/'1' → ints
mortality_df["MORTSTAT"] = pd.to_numeric(mortality_df["MORTSTAT"], errors="coerce")

model_df["SEQN"] = model_df["SEQN"].astype(int)
mortality_df["SEQN"] = mortality_df["SEQN"].astype(int)

# Merge mortality into main dataset
model_df = model_df.merge(mortality_df[["SEQN", "MORTSTAT"]], on="SEQN", how="inner")

# Assign binary mortality outcome: 1 = dead, 0 = alive
model_df["mortality"] = (model_df["MORTSTAT"] == 1).astype(int)
model_df.drop(columns=["MORTSTAT"], inplace=True)

# Build training model
features = [
    'age', 'sex', 'race_ethnicity', 'education_level', 'income_poverty_ratio',
    'current_smoker', 'alcohol_freq','bmi'
]

# Drop rows with missing values
model_df = model_df.dropna(subset=features + ["mortality"])

# Define features and target
X = model_df[features]
y = model_df["mortality"]

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train baseline model (RandomForest)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save trained model
from joblib import dump
dump(clf, "results/behavior_model.joblib")

# Evaluate model
from sklearn.metrics import classification_report, roc_auc_score
y_pred = clf.predict(X_val)
y_prob = clf.predict_proba(X_val)[:, 1]

print(classification_report(y_val, y_pred))
print("ROC AUC:", roc_auc_score(y_val, y_prob))

# Confusion Matrix
from sklearn.metrics import confusion_matrix, roc_curve
cm = confusion_matrix(y_val, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix:\n{cm}")
print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")

# ROC Curve CSV export
fpr, tpr, _ = roc_curve(y_val, y_prob)
pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv("results/roc_train.csv", index=False)

# Feature Importance
importances = clf.feature_importances_
feat_names = X.columns
importance_df = pd.DataFrame({"feature": feat_names, "importance": importances})
importance_df = importance_df.sort_values(by="importance", ascending=False)
print("\nFeature Importances:\n", importance_df)
importance_df.to_csv("results/feature_importance.csv", index=False)

# Summary statistics by outcome
model_df["mortality"] = y  # Just to ensure the variable is there for grouping
summary = model_df.groupby("mortality")[features].agg(['mean', 'std']).round(2)
summary.to_csv("results/descriptive_stats_by_outcome_train_clean.csv", index=False)

