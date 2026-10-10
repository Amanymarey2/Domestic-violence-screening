# Domestic-violence-screening
# Egypt UHIS Early Warning System (EWS): Multi-Modal Vulnerability & Domestic Violence Screening

[![Python Version](https://shields.io)](https://python.org)
[![ML Framework](https://shields.io)](https://scikit-learn.org)
[![BI Tool](https://shields.io)](https://microsoft.com)

## 📌 Introduction

In recent years, Egypt has taken massive strides toward digitizing its healthcare sector. The rollout of the **Universal Health Insurance System (UHIS - التأمين الصحي الشامل)** represents a paradigm shift, moving the nation toward comprehensive, centralized, and electronic health record (EHR) systems. By transforming raw paper logs into structured digital data, Egypt has unlocked a historic opportunity: utilizing machine learning to catch systemic social and clinical risks early.

Domestic violence and household trauma remain highly underreported challenges that are often masked within busy primary care workflows. Frontline doctors in rural clinics frequently capture signs of abuse through subtle phrasing in their text notes or through recurrent injury frequencies, rather than direct disclosures. 

This project builds an **Early Warning System (EWS)** that flags hidden household vulnerabilities by analyzing unstructured clinical narratives alongside longitudinal patient tracking records. This system is designed as a data-driven decision-making engine for national development frameworks like **Hayah Karima (حياة كريمة)**. By aggregating risk indices at the Governorate and Markaz levels, national campaigns can dynamically deploy social worker caravans, legal clinics, and economic aid directly to the regions with the highest intercepted trauma densities.

---

## 📂 Repository File Directory

```unset
├── saved_models/                      # Serialized, production-ready ML model binaries (.joblib)
├── Database schema.png                # Visual static Entity-Relationship Diagram (ERD)
├── coordinates.xlsx                   # Latitude/Longitude keys for Egypt's Governorates and Markazes
├── db_creation.py                     # Script initializing the SQLite database, schema constraints, and DDL
├── schema.sql                         # Standard SQL production layout file detailing table structures
├── synthetic_data_openai.py           # LLM data generator with dynamic context injection
├── data_prep_model.py                 # Advanced feature engineering script (180-day rolling lookbacks)
├── nlp_modeling.py                    # Multi-tiered stacked NLP text and ensemble modeling pipeline
├── data_prep_powerbi.py               # SQL consolidation script extracting the flat reporting view
├── egypt_uhis_ews.db                  # Local production SQLite database database file
├── hayah_karima_decision_support_data.xlsx # Formatted geographical data pipeline file fed into Power BI
├── high_risk_clinical_audit_sample.xlsx    # High-severity case extractions exported for clinical review
├── structure.md                       # High-level markdown mapping out pipeline data flows
└── domestic violence screening.pbix   # Core interactive Power BI executive dashboard
```

---
Here is the updated markdown, structured as a clean, scannable table for your README, followed by sample code blocks demonstrating how these locations can be dynamically selected or sampled.

## 🌍 Introduction
This project utilizes a robust framework for generating highly realistic synthetic data modeled after authentic regional distributions. To ensure data diversity, the framework incorporates geographic structures from major Egyptian regions, specifically targeting the Suhag, Assuit, and Sharqia governorates along with their respective markazes (administrative districts).

## 📊 Geographic Baselines
The table below outlines the specific regional entities utilized as structural examples for the synthetic data generation:

| Governorate | Markaz / City Examples |
|---|---|
| Suhag | Suhag, Akhmim, Girga, Tahta, Markz Al Monshah, Markz Mallawi, Markz Maghagha, Markz Bani Mazar, Markz Samalut, Markz Abu Qurgas, Markz Dir Mawas |
| Assuit | Assuit, Manfalut, Abnub, Dayrout, Al-Qusia |
| Sharqia | Sharqia, Zaqaziq, Bilbis, Abu Hammad, Minya al-Qamh, Faqus |

------------------------------


---
## ⚙️ Data Engineering & Pipeline Workflow

```unset
[ Data Generation (OpenAI) ] ➔ [ SQLite Relational DB ] ➔ [ 180-Day Rolling Windows ]
                                                                       │
[ Power BI Decision Support ] 📍 [ Stacked ML Prediction ] 🩻 [ Text Regularization ]
```

### 1. Adversarial Synthetic Data Generation (`synthetic_data_openai.py`)
To emulate realistic Egyptian EHR clinic logs without compromising actual patient privacy, data was synthesized across ~7,000 clinic charts using `gpt-4o-mini` with strict behavioral prompts:
* **Dynamic Context Injection:** To prevent the model from falling into a "vocabulary loop" where it memorizes simple templates, the script injects randomized non-abuse primary care complaints (e.g., *acute back pain from agricultural work, cold and flu, routine diabetic prescription renewals*) directly into baseline records.
* **Anti-Boilerplate Constraints:** Forbids standard LLM shortcut markers (*كل حاجة تمام, لا شكاوى*). It forces the AI to open with shuffled clinical shorthand variations (`مريض جاي بـ`, `العيان جالي الوحدة يشكي من`) to accurately mimic a busy clinic environment.

### 2. Relational Modeling (`db_creation.py`, `schema.sql`)
The database normalizes information across 4 connected layers to enforce structural integrity:
$$\text{Facilities} \longrightarrow \text{Households} \longrightarrow \text{Individuals} \longrightarrow \text{Visits}$$
It also logs concurrent socio-economic shocks (*Divorce, Orphaned, Partner Migration*) within a standalone `life_events` ledger.

### 3. Feature Preprocessing & Engineering (`data_prep_model.py`)
* **Text Regularization:** Cleans Arabic text fields and drops clinical false-positive traps. For instance, it isolates lookahead strings matching heartbeats (`ضربات\s*(الـ?|بـ?)?قلب`) so the system doesn't confuse a normal pulse reading with references to physical blunt trauma (*blows/strikes*).
* **Temporal Cross-Sectional Lookbacks:** Rather than evaluating patient checkups as isolated incidents, the script sorts data chronologically and applies a **180-day time-series rolling lookback window** to extract historical injury patterns (`prev_injury_count_6m`) on the fly:
  ```python
  df.rolling(window="180D", on="visit_date", closed="left")["injury_present"].sum()
  ```

---

## 🧠 Machine Learning Architecture (`nlp_modeling.py`)

The system deploys a **Two-Tier Stacked ML Engine** designed to handle highly imbalanced health data safely:

1. **Tier 1 (NLP Feature Extractor):** Processes cleaned Arabic-English medical strings through a `TfidfVectorizer` and a regularized `LogisticRegression` classifier (`C=0.5`). This converts messy clinical notes into a continuous text probability vector (`nlp_severity_score`).
2. **Tier 2 (Unified Ensemble):** Merges the `nlp_severity_score` with longitudinal tracking metrics (`prev_injury_count_6m`, `days_since_injury`) into a final **`HistGradientBoostingClassifier`**. This handles non-linear risk interactions and processes missing timeline fields natively without introducing data imputation bias.

### 📊 Model Performance Evaluation

The model achieved high performance metrics on the hidden validation split, eliminating the vocabulary dependencies common in standard text classifiers:

```unset
Final Combined Model Validation Performance:
              precision    recall  f1-score   support

         Low       1.00      1.00      1.00      1367
      Medium       0.81      0.89      0.85        19
        High       0.91      0.88      0.89        24

    accuracy                           0.99      1410
   macro avg       0.91      0.92      0.91      1410
weighted avg       0.99      0.99      0.99      1410
```

* **High Class Precision (0.91):** When the system fires a severe alert, it is correct **91% of the time**, dramatically reducing false-alarm fatigue for clinicians.
* **High Class Recall (0.88):** Successfully intercepts **88% of hidden high-severity records** in the validation pool.

### 🔍 Interpretability: Top Intercepted Risk Phrases
Extracting the model's coefficients shows that it bypasses generic clinical formatting and targets structural markers of risk and clinical narrative ambiguity:
1. `مش` (Not) — *Indicating narrative contradictions*
2. `متعددة` (Multiple) — *Indicating distributed physical injuries*
3. `مش راكب` (Does not add up) — *Highlighting clinical history mismatches*
4. `القصة غير واضحة` (The history is unclear) — *Flagging defensive explanations*
5. `كدمات متعددة` (Multiple bruises) — *Objective trauma documentation*

---

## 📊 Analytics & Decision Support Dashboard (`domestic violence screening.pbix`)

`data_prep_powerbi.py` joins model predictions back to regional coordinates to fuel an executive **Power BI Decision Support Dashboard** designed for social welfare resource planning.

### 🎥 Dashboard Figures

**Page 1**
![image](https://github.com/Amanymarey2/Domestic-violence-screening/blob/main/page1%20dashboard.png)

**Page 2**
![image](https://github.com/Amanymarey2/Domestic-violence-screening/blob/main/page1%20dashboard.png)

### 🎛️ Global Slicers & Banner KPIs
Every page features global slicers for **Governorate**, **Markaz**, and **Date Range**, displaying three real-time KPI metrics:
* **Total Screened Visits:** Total baseline clinical volume.
* **Identified High-Risk Clusters:** Total volume of severe flags needing active intervention.
* **Average Regional Vulnerability Score:** A continuous risk indicator calculated across patient histories.

### 📑 Dashboard Page Layout

#### Page 1: Regional Hotspots & Socio-Environmental Drivers
* **Geographical Risk Densities (ArcGIS Map):** Maps the volume of `High` and `Medium` risk levels across Upper Egypt Markazes. This enables **Hayah Karima** planners to identify localized trauma hotspots instantly.
* **Socio-Vulnerability Tree Map:** Displays the percentage of elevated risk categories broken down by region and household substance abuse flags, isolating the strongest community risk correlations.

#### Page 2: Demographic Profiles & Trend Analysis
* **Vulnerability Trajectories (Line Chart):** Tracks the average continuous risk metrics over several years to show whether regional trends are increasing or stabilizing.
* **Age Risk Distributions (Bar Chart):** Displays the average age of patients across different risk categories, helping child protective systems distinguish pediatric risks from adult indicators.
* **Income Matrix Profile (Stacked Bar Chart):** Breaks down income level percentages within each risk category to map out the impacts of economic poverty on household stability.
* **Socio-Economic Stress Flows (Sankey Chart):** Models the complex relationships between household substance abuse and income profiles to trace systemic vulnerability paths.
* **Employment Stability Streams (Ribbon Stacked Bar Chart):** Tracks shifts in high-risk categories across different employment statuses, illustrating how labor disruptions affect household stress markers.


