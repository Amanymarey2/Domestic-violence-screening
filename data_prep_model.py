#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -------------------------------------------------
# Configuration
# -------------------------------------------------
DB_PATH = "egypt_uhis_ews.db"
OUTPUT_PATH = "modeling_dataset.parquet"   # or .csv

conn = sqlite3.connect(DB_PATH)

print("Loading and joining tables...")

# -------------------------------------------------
# 1. Base Join (Visit level)
# -------------------------------------------------
query = """
SELECT 
    v.*,
    i.sex,
    i.age_at_registration,
    i.marital_status,
    i.education_level,
    i.employment_status,
    i.relationship_to_head,
    i.chronic_conditions,
    h.income_level,
    h.income_source,
    h.household_size,
    h.crowding_index,
    h.has_substance_use_flag,
    h.education_mismatch_flag,
    h.step_parent_or_orphan_flag,
    h.disability_in_household,
    h.geographic_isolation_score,
    f.governorate,
    f.markaz,
    f.facility_id,
    f.urbanization
FROM visits v
LEFT JOIN individuals i ON v.individual_id = i.individual_id
LEFT JOIN households h ON v.household_id = h.household_id
LEFT JOIN facilities f ON v.facility_id = f.facility_id
"""

df = pd.read_sql(query, conn)
print(f"Base dataset shape: {df.shape}")

# -------------------------------------------------
# 2. Basic Cleaning
# -------------------------------------------------
df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")

# Fill simple missing values
flag_cols = [
    "injury_present", "burn_present", "malnutrition_flag", "alopecia_flag",
    "hypopigmentation_flag", "neglected_chronic_flag", "anxiety_flag",
    "depression_flag", "sleep_disturbance_flag", "hymen_exam_requested",
    "reported_abuse_in_notes", "neurologic_or_psych_disorder",
    "referral_token_issued", "has_substance_use_flag", "education_mismatch_flag",
    "step_parent_or_orphan_flag", "disability_in_household"
]

for col in flag_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)
df["days_since_injury"] = df["days_since_injury"].fillna(9999)

# -------------------------------------------------
# 3. Visit-level Derived Features
# -------------------------------------------------
print("Creating visit-level features...")

df["is_severe_injury"] = (
    (df["injury_severity"] == "Severe") | (df["injury_type"] == "Assault")
).astype(int)

df["has_mental_flag"] = (
    (df["depression_flag"] == 1) | (df["anxiety_flag"] == 1)
).astype(int)

df["has_nutritional_dermal"] = (
    (df["malnutrition_flag"] == 1) | 
    (df["alopecia_flag"] == 1) | 
    (df["hypopigmentation_flag"] == 1)
).astype(int)

df["tier3_signal"] = (
    (df["reported_abuse_in_notes"] == 1) |
    (df["hymen_exam_requested"] == 1) |
    (df["injury_type"] == "Assault") |
    (df["is_severe_injury"] == 1)
).astype(int)

# -------------------------------------------------
# 4. Longitudinal Features (Very Important)
# -------------------------------------------------
print("Creating longitudinal features...")

df = df.sort_values(["household_id", "visit_date"])

def add_longitudinal_features(group):
    group = group.sort_values("visit_date").copy()
    
    # Rolling counts (previous visits only)
    group["prev_injury_count_6m"] = (
        group.rolling(window="180D", on="visit_date", closed="left")["injury_present"]
        .sum()
        .fillna(0)
)
    
    group["prev_mental_count_6m"] = (
        group.rolling(window="180D", on="visit_date", closed="left")['has_mental_flag']
        .sum()
        .fillna(0)
    )
    
    group["prev_malnutrition_count_6m"] = (
        group.rolling(window="180D", on="visit_date", closed="left")["malnutrition_flag"]
        .sum()
        .fillna(0)
    )
    
    group["prev_tier3_count"] = (
        group.rolling(window="365D", on="visit_date", closed="left")["tier3_signal"]
        .sum()
        .fillna(0)
    )
    
    # Time since last concerning visit
    concerning = group["injury_present"] == 1
    group["days_since_last_injury"] = (
        group["visit_date"] - group.loc[concerning, "visit_date"].shift(1)
    ).dt.days
    
    return group

df = df.groupby("household_id", group_keys=False).apply(add_longitudinal_features)

# Fill remaining NaNs
df["days_since_last_injury"] = df["days_since_last_injury"].fillna(999)
df[["prev_injury_count_6m", "prev_mental_count_6m", 
    "prev_malnutrition_count_6m", "prev_tier3_count"]] = \
    df[["prev_injury_count_6m", "prev_mental_count_6m", 
        "prev_malnutrition_count_6m", "prev_tier3_count"]].fillna(0)

# -------------------------------------------------
# 5. Life Event Features (Tier-2)
# -------------------------------------------------
print("Adding life event features...")

life_events = pd.read_sql("SELECT * FROM life_events", conn)
life_events["event_date"] = pd.to_datetime(life_events["event_date"], errors="coerce")

# Merge nearest previous life event
df = df.sort_values("visit_date")
life_events = life_events.sort_values("event_date")

df = pd.merge_asof(
    df,
    life_events[["household_id", "event_date", "event_type"]].rename(
        columns={"event_date": "last_life_event_date", "event_type": "last_life_event_type"}
    ),
    left_on="visit_date",
    right_on="last_life_event_date",
    by="household_id",
    direction="backward"
)

df["days_since_life_event"] = (df["visit_date"] - df["last_life_event_date"]).dt.days
df["recent_life_event_6m"] = (df["days_since_life_event"] <= 180).astype(int)
df["days_since_life_event"] = df["days_since_life_event"].fillna(999)
df["recent_life_event_6m"] = df["recent_life_event_6m"].fillna(0).astype(int)

# -------------------------------------------------
# 6. Household Aggregated Risk Features
# -------------------------------------------------
print("Creating household-level aggregations...")

hh_agg = df.groupby("household_id").agg(
    total_visits=("visit_id", "count"),
    total_injuries=("injury_present", "sum"),
    total_tier3=("tier3_signal", "sum"),
    total_mental=("has_mental_flag", "sum"),
    total_malnutrition=("malnutrition_flag", "sum"),
    avg_bmi=("bmi", "mean")
).reset_index()

hh_agg["injury_rate"] = hh_agg["total_injuries"] / hh_agg["total_visits"]
hh_agg["tier3_rate"] = hh_agg["total_tier3"] / hh_agg["total_visits"]

df = df.merge(hh_agg, on="household_id", how="left")

# -------------------------------------------------
# 7. Simple Target Variable (Initial Version)
# -------------------------------------------------
print("Creating initial target variable...")

conditions = [
    (df["tier3_signal"] == 1) | (df["prev_tier3_count"] >= 1),
    (df["prev_injury_count_6m"] >= 2) & 
    ((df["prev_mental_count_6m"] >= 1) | (df["prev_malnutrition_count_6m"] >= 1)),
    (df["prev_injury_count_6m"] >= 3) | (df["injury_rate"] > 0.4)
]

choices = [2, 1, 1]  # 2 = High, 1 = Medium

df["risk_label"] = np.select(conditions, choices, default=0)
df["risk_label"] = df["risk_label"].astype(int)

# Binary version
df["high_risk"] = (df["risk_label"] >= 1).astype(int)

print("\nRisk Label Distribution:")
print(df["risk_label"].value_counts().sort_index())

# -------------------------------------------------
# 8. Final Selection of Modeling Columns
# -------------------------------------------------
modeling_cols = [
    # IDs (keep for reference)
    "visit_id", "individual_id", "household_id", "visit_date", "governorate",
    
    # Target
    "risk_label", "high_risk",
    
    # Structured flags
    "injury_present", "is_severe_injury", "burn_present",
    "malnutrition_flag", "has_nutritional_dermal",
    "has_mental_flag", "tier3_signal",
    
    # Continuous
    "bmi", "weight_change_since_last", "mental_health_score",
    "days_since_injury", "days_since_last_injury",
    
    # Longitudinal
    "prev_injury_count_6m", "prev_mental_count_6m",
    "prev_malnutrition_count_6m", "prev_tier3_count",
    
    # Life events
    "recent_life_event_6m", "days_since_life_event",
    
    # Household
    "income_level", "has_substance_use_flag", "education_mismatch_flag",
    "step_parent_or_orphan_flag", "disability_in_household",
    "geographic_isolation_score", "crowding_index", "household_size",
    "injury_rate", "tier3_rate", "total_visits",
    
    # Demographics
    "sex", "age_at_registration", "marital_status", "education_level"
]

# Keep only existing columns
modeling_cols = [c for c in modeling_cols if c in df.columns]
df_model = df[modeling_cols].copy()

# -------------------------------------------------
# 9. Save
# -------------------------------------------------
df_model.to_parquet(OUTPUT_PATH, index=False)
# df_model.to_csv("modeling_dataset.csv", index=False)  # alternative

print(f"\nModeling dataset saved to: {OUTPUT_PATH}")
print(f"Final shape: {df_model.shape}")
print("\nColumns:")
print(df_model.columns.tolist())

conn.close()
print("\nDone.")

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# Load the dataset
# -------------------------------------------------
df = pd.read_parquet("modeling_dataset.parquet")

print("=" * 70)
print("1. BASIC OVERVIEW")
print("=" * 70)
print(f"Shape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

# -------------------------------------------------
# 2. Missing Values
# -------------------------------------------------
print("\n" + "=" * 70)
print("2. MISSING VALUES")
print("=" * 70)
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    print(missing)
else:
    print("No missing values found.")

# -------------------------------------------------
# 3. Target Variable Distribution
# -------------------------------------------------
print("\n" + "=" * 70)
print("3. TARGET DISTRIBUTION")
print("=" * 70)

print("\nRisk Label (0=Low, 1=Medium, 2=High):")
print(df["risk_label"].value_counts().sort_index())
print("\nNormalized:")
print(df["risk_label"].value_counts(normalize=True).sort_index().round(3))

print("\nHigh Risk (binary):")
print(df["high_risk"].value_counts())
print(df["high_risk"].value_counts(normalize=True).round(3))

# -------------------------------------------------
# 4. Key Categorical Features
# -------------------------------------------------
print("\n" + "=" * 70)
print("4. KEY CATEGORICAL FEATURES")
print("=" * 70)

categorical_cols = ["sex", "income_level", "marital_status", "education_level", "governorate"]

for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).head(10))

# -------------------------------------------------
# 5. Key Binary / Flag Features
# -------------------------------------------------
print("\n" + "=" * 70)
print("5. FLAG FEATURES (Mean = Prevalence)")
print("=" * 70)

flag_cols = [
    "injury_present", "is_severe_injury", "burn_present",
    "malnutrition_flag", "has_mental_flag", "tier3_signal",
    "has_substance_use_flag", "education_mismatch_flag",
    "step_parent_or_orphan_flag", "recent_life_event_6m", "high_risk"
]

existing_flags = [c for c in flag_cols if c in df.columns]
print(df[existing_flags].mean().sort_values(ascending=False).round(3))

# -------------------------------------------------
# 6. Numerical Features Summary
# -------------------------------------------------
print("\n" + "=" * 70)
print("6. NUMERICAL FEATURES SUMMARY")
print("=" * 70)

numerical_cols = [
    "bmi", "weight_change_since_last", "mental_health_score",
    "prev_injury_count_6m", "prev_mental_count_6m",
    "prev_malnutrition_count_6m", "prev_tier3_count",
    "days_since_last_injury", "injury_rate", "tier3_rate",
    "household_size", "crowding_index", "geographic_isolation_score"
]

existing_num = [c for c in numerical_cols if c in df.columns]
print(df[existing_num].describe().round(2))

# -------------------------------------------------
# 7. Correlation with Target
# -------------------------------------------------
print("\n" + "=" * 70)
print("7. CORRELATION WITH HIGH_RISK TARGET")
print("=" * 70)

corr_cols = existing_flags + existing_num
corr_cols = [c for c in corr_cols if c in df.columns and c != "high_risk"]

correlations = df[corr_cols + ["high_risk"]].corr()["high_risk"].drop("high_risk")
correlations = correlations.sort_values(key=abs, ascending=False)

print(correlations.round(3))

# -------------------------------------------------
# 8. Simple Correlation Heatmap (Optional)
# -------------------------------------------------
print("\n" + "=" * 70)
print("8. TOP CORRELATIONS HEATMAP")
print("=" * 70)

top_features = correlations.head(12).index.tolist() + ["high_risk"]
corr_matrix = df[top_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Top Features Correlation with High Risk")
plt.tight_layout()
plt.show()

print("\nInspection completed.")

