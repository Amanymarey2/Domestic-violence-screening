#!/usr/bin/env python
# coding: utf-8

# In[3]:


import re
import sqlite3
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# -------------------------------------------------
# Configuration
# -------------------------------------------------
DB_PATH = "egypt_uhis_ews.db"
OUTPUT_TABLE = "visit_nlp_scores"

# -------------------------------------------------
# 1. Load Notes Only
# -------------------------------------------------
conn = sqlite3.connect(DB_PATH)

df = pd.read_sql(
    """
    SELECT 
        visit_id,
        individual_id,
        household_id,
        visit_date,
        provider_notes_short AS note
    FROM visits
    WHERE provider_notes_short IS NOT NULL
      AND TRIM(provider_notes_short) != ''
""",
    conn,
)

print(f"Loaded notes: {len(df)}")


# -------------------------------------------------
# 2. Text Cleaning
# -------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    
    # ---- 1. Strip Out Cardiac False Positives ----
    # Removes variants of "heartbeats", "heart rate", or "heart patient"
 
    text = re.sub(r'ضربات\s*(الـ?|بـ?)?قلب', ' ', text)
    
    # ---- 2. Strip Out Routine Boilerplate Discharge Text ----
    # Removes variants of "Good, follow up after..."
    text = re.sub(r'كويسة\s+مراجعة(\s+بعد)?', ' ', text)
    text = re.sub(r'الحالة\s+كويسة', ' ', text)
    
    # ---- 3. Standard Text Standardization ----
    text = re.sub(r'[^\w\s\u0600-\u06FF.,;!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()



df["clean_note"] = df["note"].apply(clean_text)
df = df[df["clean_note"].str.len() > 3].copy()
print(f"Notes after cleaning: {len(df)}")



# -------------------------------------------------
# 3. Rule-Based Silver Labeling
# -------------------------------------------------

KEYWORDS = {
    "high": {
        "weight": 1.0,
        "terms": [
            "اعتداء",
            "ضرب",
            "عنف",
            "كدمات متعددة",
            "إصابات متعددة",
            "قصة غير واضحة",
            "قصة غير متطابقة",
            "رواية مختلفة",
            "خوف شديد",
            "مراحل شفاء",
            "intentional",
            "assault",
            "multiple bruises",
            "inconsistent history",
        ],
    },
    "medium": {
        "weight": 0.55,
        "terms": [
            "متكرر",
            "مرة أخرى",
            "حوادث منزلية",
            "حادث منزلي",
            "سقوط متكرر",
            "سقوط في المنزل",
            "وقع في البيت",
            "وزن نزل",
            "نقص وزن",
            "سوء تغذية",
            "مزاج",
            "قلق",
            "اكتئاب",
            "تعبانة",
            "أرق",
            "repeated",
            "recurrent",
            "household accident",
            "home accident",
            "weight loss",
            "low mood",
        ],
    },
    "injury": {
        "weight": 0.25,
        "terms": [
            "كدمة",
            "كدمات",
            "إصابة",
            "سقوط",
            "حرق",
            "جرح",
            "bruise",
            "burn",
            "fall",
            "trauma",
            "injury",
        ],
    },
    "vague": {
        "weight": 0.20,
        "terms": ["بتقول", "حسب قول", "قالت إن", "رواية", "قصة", "claimed", "according to", "history"],
    },
}


def rule_based_labeler(text):
    score = 0.0
    high_hits = 0
    medium_hits = 0
    injury_hits = 0
    vague_hits = 0

    for group_name, group in KEYWORDS.items():
        count = sum(1 for term in group["terms"] if term in text)
        score += count * group["weight"]
        if group_name == "high":
            high_hits = count
        elif group_name == "medium":
            medium_hits = count
        elif group_name == "injury":
            injury_hits = count
        elif group_name == "vague":
            vague_hits = count

    severity_score = min(score / 3.0, 1.0)

    # Classifying target categories: 0=Low, 1=Medium, 2=High
    if high_hits >= 1 or severity_score >= 0.75:
        return 2, severity_score
    elif medium_hits >= 1 or (injury_hits >= 1 and vague_hits >= 1) or severity_score >= 0.35:
        return 1, severity_score
    else:
        return 0, severity_score


# Generate tracking data and targets
labels_and_scores = [rule_based_labeler(text) for text in df["clean_note"]]
df["target_level"] = [x[0] for x in labels_and_scores]
df["silver_score"] = [x[1] for x in labels_and_scores]

# -------------------------------------------------
# 4. Train Machine Learning Pipeline
# -------------------------------------------------
print("\nSplitting and Training Model...")

X = df["clean_note"]
y = df["target_level"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Text ML Pipeline: Combines word n-grams & character n-grams to handle Arabic root structures perfectly
# Text ML Pipeline with stronger regularization (C=0.5 or C=0.1)
model_pipeline = make_pipeline(
    TfidfVectorizer(
        ngram_range=(1, 3), 
        analyzer="word", 
        min_df=2, 
        sublinear_tf=True
    ),
    # Lowering C from 1.0 to 0.5 prevents the model from over-focusing on rare words
    LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=42),
)

# Train the model on the notes text only
model_pipeline.fit(X_train, y_train)

from sklearn.metrics import classification_report

# Predict on validation data
val_preds = model_pipeline.predict(X_val)

# Print a breakdown for Low (0), Medium (1), and High (2)
print("\nDetailed Validation Performance:")
print(classification_report(y_val, val_preds, target_names=["Low", "Medium", "High"]))


# Evaluation
train_acc = model_pipeline.score(X_train, y_train)
val_acc = model_pipeline.score(X_val, y_val)
print(f"Training Accuracy: {train_acc:.3f}")
print(f"Validation Accuracy: {val_acc:.3f}")

# -------------------------------------------------
# 5. Apply Model to Complete Dataset
# -------------------------------------------------
print("\nGenerating predictions with trained model...")

# Predict actual probability profiles for continuous scoring
prob_matrix = model_pipeline.predict_proba(df["clean_note"])

# We construct a weighted scalar score based on the risk spectrum classes (0, 1, 2)
# prob_medium * 0.5 + prob_high * 1.0 gives a continuous risk score between 0 and 1
df["nlp_severity_score"] = np.round((prob_matrix[:, 1] * 0.5) + (prob_matrix[:, 2] * 1.0), 4)

# Predict the multi-class discrete value
predicted_classes = model_pipeline.predict(df["clean_note"])
level_mapping = {0: "Low", 1: "Medium", 2: "High"}
df["nlp_severity_level"] = [level_mapping[cls] for cls in predicted_classes]

# Extract feature hits for historical reference tracking
df["kw_high_hits"] = [sum(1 for term in KEYWORDS["high"]["terms"] if term in t) for t in df["clean_note"]]
df["kw_medium_hits"] = [sum(1 for term in KEYWORDS["medium"]["terms"] if term in t) for t in df["clean_note"]]
df["kw_injury_hits"] = [sum(1 for term in KEYWORDS["injury"]["terms"] if term in t) for t in df["clean_note"]]
df["kw_vague_hits"] = [sum(1 for term in KEYWORDS["vague"]["terms"] if term in t) for t in df["clean_note"]]

# -------------------------------------------------
# 6. Save Results
# -------------------------------------------------
output_cols = [
    "visit_id",
    "individual_id",
    "household_id",
    "visit_date",
    "nlp_severity_score",
    "nlp_severity_level",
    "kw_high_hits",
    "kw_medium_hits",
    "kw_injury_hits",
    "kw_vague_hits",
]

results = df[output_cols].copy()
results.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

print("\nTrained model level distribution:")
print(results["nlp_severity_level"].value_counts())

print("\nTrained model continuous score summary:")
print(results["nlp_severity_score"].describe().round(3))

conn.close()
print(f"\nResults successfully saved to table: {OUTPUT_TABLE}")


# In[4]:


# Extract the components from the text pipeline
vectorizer = model_pipeline.named_steps["tfidfvectorizer"]
classifier = model_pipeline.named_steps["logisticregression"]

# Get all words/phrases matched by the vectorizer
feature_names = np.array(vectorizer.get_feature_names_out())

# Index 2 corresponds to the 'High' severity class coefficients
high_class_coefficients = classifier.coef_[2]

# Sort features by their weight impact
top_high_indices = np.argsort(high_class_coefficients)[-15:][::-1]

print("\n=== TOP 15 PHRASES DRIVING 'HIGH' RISK SEVERITY ===")
for rank, idx in enumerate(top_high_indices, 1):
    print(f"{rank}. Token: '{feature_names[idx]}' -> Weight: {high_class_coefficients[idx]:.4f}")


# In[5]:


import re
import sqlite3
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Or your specific logical core count


# -------------------------------------------------
# Configuration
# -------------------------------------------------
DB_PATH = "egypt_uhis_ews.db"
OUTPUT_TABLE = "visit_final_predictions"

# -------------------------------------------------
# 1. Load Combined Data Structure
# -------------------------------------------------
conn = sqlite3.connect(DB_PATH)

# Selecting text alongside structured indicators
# Assumes days_since_injury and injury_present columns exist in the same source table
df = pd.read_sql("""
    SELECT 
        visit_id,
        individual_id,
        household_id,
        visit_date,
        injury_present,
        days_since_injury,
        provider_notes_short AS note
    FROM visits
    WHERE provider_notes_short IS NOT NULL
      AND TRIM(provider_notes_short) != ''
""", conn)

print(f"Loaded records: {len(df)}")

# -------------------------------------------------
# 2. Text Preprocessing & Structured Feature Engineering
# -------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    
    # Clean cardiac terms and follow-up boilerplate
    text = re.sub(r'ضربات\s*(الـ?|بـ?)?قلب', ' ', text)
    text = re.sub(r'كويسة\s+مراجعة(\s+بعد)?', ' ', text)
    text = re.sub(r'الحالة\s+كويسة', ' ', text)
    
    text = re.sub(r'[^\w\s\u0600-\u06FF.,;!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df["clean_note"] = df["note"].apply(clean_text)
df = df[df["clean_note"].str.len() > 3].copy()

# --- FIX SECTIONS FOR DATAFRAME ROLLING WINDOWS ---
# 1. Force explicit conversion to datetime before running rolling calculations
df["visit_date"] = pd.to_datetime(df["visit_date"])

# 2. Sort values ascending by timeline (Required by pandas rolling windows)
df = df.sort_values("visit_date")

# 3. Safe engineering of structured features
df["prev_injury_count_6m"] = (
    df.rolling(window="180D", on="visit_date", closed="left")["injury_present"]
    .sum()
    .fillna(0)
)



KEYWORDS = {
    "high": {"weight": 1.0, "terms": ["اعتداء", "ضرب", "عنف", "كدمات متعددة", "قصة غير واضحة", "assault", "multiple bruises"]},
    "medium": {"weight": 0.55, "terms": ["متكرر", "حوادث منزلية", "سقوط في المنزل", "repeated", "household accident"]},
    "injury": {"weight": 0.25, "terms": ["كدمة", "إصابة", "سقوط", "جرح", "bruise", "fall", "injury"]},
    "vague": {"weight": 0.20, "terms": ["بتقول", "قصة", "claimed", "history"]}
}

def rule_based_labeler(text):
    score = 0.0
    high_hits = 0
    medium_hits = 0
    for group_name, group in KEYWORDS.items():
        count = sum(1 for term in group["terms"] if term in text)
        score += count * group["weight"]
        if group_name == "high": high_hits = count
        elif group_name == "medium": medium_hits = count
    
    severity_score = min(score / 3.0, 1.0)
    if high_hits >= 1 or severity_score >= 0.75: return 2
    elif medium_hits >= 1 or severity_score >= 0.35: return 1
    return 0

df["target_level"] = df["clean_note"].apply(rule_based_labeler)

# -------------------------------------------------
# 3. Step 1: Fit Text Pipeline & Inspect Words
# -------------------------------------------------
X_text = df["clean_note"]
y = df["target_level"]

model_pipeline = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 3), analyzer="word", min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=42)
)
model_pipeline.fit(X_text, y)

# Word Weight Inspection
vectorizer = model_pipeline.named_steps["tfidfvectorizer"]
classifier = model_pipeline.named_steps["logisticregression"]
feature_names = np.array(vectorizer.get_feature_names_out())
high_coefficients = classifier.coef_[2]
top_high_indices = np.argsort(high_coefficients)[-10:][::-1]

print("\n=== TOP 10 RISK PHRASES INTERCEPTED ===")
for rank, idx in enumerate(top_high_indices, 1):
    print(f"  {rank}. '{feature_names[idx]}' (Weight: {high_coefficients[idx]:.3f})")

# Generate continuous text scores for the whole dataset
prob_matrix = model_pipeline.predict_proba(df["clean_note"])
df["nlp_severity_score"] = np.round((prob_matrix[:, 1] * 0.5) + (prob_matrix[:, 2] * 1.0), 4)

# -------------------------------------------------
# 4. Step 2: Final Combined Classifier
# -------------------------------------------------
print("\nTraining Final Combined Model...")

# Features matrix (Tabular + Meta Text Score)
# HistGradientBoosting natively tracks NaN values in days_since_injury without imputation!
features = ["prev_injury_count_6m", "days_since_injury", "nlp_severity_score"]
X_final = df[features]

X_train, X_val, y_train, y_val = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)

final_model = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
final_model.fit(X_train, y_train)

# Validate combined capability
val_preds = final_model.predict(X_val)
print("\nFinal Combined Model Validation Performance:")
print(classification_report(y_val, val_preds, target_names=["Low", "Medium", "High"]))

# -------------------------------------------------
# 5. Predict and Export Everything
# -------------------------------------------------
df["final_predicted_level_int"] = final_model.predict(X_final)
level_mapping = {0: "Low", 1: "Medium", 2: "High"}
df["final_severity_level"] = df["final_predicted_level_int"].map(level_mapping)

# Calculate final unified continuous probability
final_probs = final_model.predict_proba(X_final)
df["final_risk_score"] = np.round((final_probs[:, 1] * 0.5) + (final_probs[:, 2] * 1.0), 4)

output_cols = [
    "visit_id", "individual_id", "household_id", "visit_date",
    "prev_injury_count_6m", "days_since_injury", "nlp_severity_score",
    "final_risk_score", "final_severity_level"
]

results = df[output_cols].copy()
results.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

print("\nFinal Multi-Modal Distribution Saved:")
print(results["final_severity_level"].value_counts())

conn.close()
print(f"\nExecution completely finished. Database updated at: {OUTPUT_TABLE}")


# In[5]:


import os
import sqlite3
import joblib
import pandas as pd

# -------------------------------------------------
# Configuration
# -------------------------------------------------
DB_PATH = "egypt_uhis_ews.db"
MODEL_DIR = "saved_models"
AUDIT_CSV_PATH = "high_risk_clinical_audit_sample.csv"

# Create the directory for models if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

print("=== TASK 1: EXPORTING TRAINED MODELS ===")

# --- 1. Save your trained models ---
# (Make sure to run this right after fitting final_model and model_pipeline in your training script)
try:
    # Save the Text NLP Pipeline (TfidfVectorizer + LogisticRegression)
    joblib.dump(model_pipeline, os.path.join(MODEL_DIR, "text_nlp_pipeline.joblib"))
    print("✅ Successfully saved 'text_nlp_pipeline.joblib' to", MODEL_DIR)

    # Save the Final Combined Classifier (HistGradientBoostingClassifier)
    joblib.dump(final_model, os.path.join(MODEL_DIR, "final_combined_classifier.joblib"))
    print("✅ Successfully saved 'final_combined_classifier.joblib' to", MODEL_DIR)
    
    print("\n💡 To reload these in a separate production script later, just use:")
    print("   nlp_model = joblib.load('saved_models/text_nlp_pipeline.joblib')")
except NameError:
    print("⚠️ Notice: To auto-save models via joblib, place the joblib.dump commands")
    print("   directly at the end of your main training execution script where those variables exist.")

print("\n" + "="*50 + "\n")
print("=== TASK 2: EXTRACTING HIGH-RISK CLINICAL AUDIT SAMPLE ===")

# --- 2. Extract High-Risk Audit Cases ---
conn = sqlite3.connect(DB_PATH)

# Fetch data from the generated output tables along with raw provider notes
# Joining the final prediction results back to the original source text
audit_df = pd.read_sql("""
    SELECT 
        p.visit_id,
        p.individual_id,
        p.visit_date,
        v.provider_notes_short AS raw_clinical_note,
        p.prev_injury_count_6m,
        p.days_since_injury,
        p.nlp_severity_score,
        p.final_risk_score,
        p.final_severity_level
    FROM visit_final_predictions p
    JOIN visits v ON p.visit_id = v.visit_id
    WHERE p.final_severity_level = 'High'
    ORDER BY p.final_risk_score DESC
""", conn)

conn.close()

if len(audit_df) == 0:
    print("❌ No 'High' severity cases found in the database table to extract.")
else:
    print(f"Total high-risk cases available in database: {len(audit_df)}")
    
    # Take a diverse audit sample: the top 15 most severe cases + a random sample of 15 other high cases
    top_severe = audit_df.head(15)
    
    if len(audit_df) > 15:
        random_severe = audit_df.iloc[15:].sample(n=min(15, len(audit_df) - 15), random_state=42)
        final_audit_sample = pd.concat([top_severe, random_severe])
    else:
        final_audit_sample = top_severe

    # Export to a structured CSV file
    final_audit_sample.to_csv(AUDIT_CSV_PATH, index=False, encoding="utf-8-sig") # utf-8-sig preserves Arabic characters in Excel
    
    print(f"✅ Successfully exported an audit sample of {len(final_audit_sample)} cases to: {AUDIT_CSV_PATH}")
    print("\n--- SAMPLE PREVIEW ---")
    print(final_audit_sample[["visit_id", "prev_injury_count_6m", "days_since_injury", "final_risk_score"]].head())


# In[8]:


import os
import re
import joblib
import pandas as pd
import numpy as np

# -------------------------------------------------
# Configuration
# -------------------------------------------------
MODEL_DIR = "saved_models"

# Text cleaning logic aligned exactly with your optimized training engine
def clean_new_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    
    # Strip out cardiac traps and follow-up boilerplate
    text = re.sub(r'ضربات\s*(الـ?|بـ?)?قلب', ' ', text)
    text = re.sub(r'كويسة\s+مراجعة(\s+بعد)?', ' ', text)
    text = re.sub(r'الحالة\s+كويسة', ' ', text)
    
    # Strip out repetitive structural frameworks
    boilerplate_patterns = [
        r'مريض\s+جاي(\s+بـ)?', r'المريض\s+جاي(\s+بـ)?',
        r'العيان\s+جاي(\s+بـ)?', r'مريضة\s+جايه(\s+بـ)?'
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, ' ', text)
        
    text = re.sub(r'[^\w\s\u0600-\u06FF.,;!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -------------------------------------------------
# Real-Time EWS Scoring Logic
# -------------------------------------------------
def pipeline_early_warning_score(raw_note, prev_injury_count_6m, days_since_injury):
    """
    Takes an active clinic entry and returns the multi-modal risk tier.
    """
    # 1. Load the binary deployment files
    nlp_pipeline = joblib.load(os.path.join(MODEL_DIR, "text_nlp_pipeline.joblib"))
    combined_classifier = joblib.load(os.path.join(MODEL_DIR, "final_combined_classifier.joblib"))
    
    # 2. Process text through the structural normalization engine
    clean_note = clean_new_text(raw_note)
    
    # 3. Generate continuous NLP probability vector
    # Pass input as an iterable series
    nlp_prob_matrix = nlp_pipeline.predict_proba([clean_note])
    nlp_severity_score = np.round((nlp_prob_matrix[0, 1] * 0.5) + (nlp_prob_matrix[0, 2] * 1.0), 4)
    
    # 4. Form structured feature payload matrix
    input_data = pd.DataFrame([{
        "prev_injury_count_6m": prev_injury_count_6m,
        "days_since_injury": days_since_injury,
        "nlp_severity_score": nlp_severity_score
    }])
    
    # 5. Extract ultimate blended multi-modal prediction
    predicted_class_int = combined_classifier.predict(input_data)[0]
    final_probabilities = combined_classifier.predict_proba(input_data)[0]
    final_risk_score = np.round((final_probabilities[1] * 0.5) + (final_probabilities[2] * 1.0), 4)
    
    level_mapping = {0: "Low", 1: "Medium", 2: "High"}
    
    return {
        "Cleaned Text Focus": clean_note,
        "Isolated Text Risk Score": nlp_severity_score,
        "Unified Multi-Modal Risk Score": final_risk_score,
        "Assigned Action Tier": level_mapping[predicted_class_int]
    }

# -------------------------------------------------
# Production Test Scenario Verification
# -------------------------------------------------
if __name__ == "__main__":
    print("=== TESTING CASUAL CLINIC VISIT DATA PROFILE ===")
    
    # Simulated Entry: Patient has a history of injury, and the text has explicit inconsistencies
    test_note = "العيانة جت عندها كدمات زرقا في دراعها، وبتقول وقعت من السرير بس التاريخ مش راكب مع شكل الإصابة."
    
    results = pipeline_early_warning_score(
        raw_note=test_note,
        prev_injury_count_6m=2,    # Multi-visit pattern
        days_since_injury=1        # Recent trauma
    )
    
    for key, val in results.items():
        print(f"{key:<30}: {val}")


# In[ ]:



