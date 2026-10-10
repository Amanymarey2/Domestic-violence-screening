## 🏗️ Early Warning System (EWS) Architecture & Data Pipeline

The project implements a stacked, multi-modal machine learning pipeline designed to identify household vulnerabilities and trauma trends from unstructured clinical narratives and longitudinal patient tracking records.

[  1. Synthetic Generation Core  ] ──> OpenAI GPT-4o-Mini with Dynamic Context Injection
                 │
                 ▼
[   2. Relational Schema Layer   ] ──> Normalized SQLite DB Engine (Facilities ➔ HH ➔ Ind ➔ Visits)
                 │
                 ▼
[  3. Text Normalization Pipeline ] ──> Regex Boilerplate Eraser & Clinical Noise Filter
                 │
                 ▼
[  4. Longitudinal Feature Eng.  ] ──> 180-Day Rolling Window Cross-Sectional Aggregators
                 │
                 ▼
[     5. Stacked ML Classifier   ] ──> Text TF-IDF Refiner ➔ Tabular HistGradientBoosting Engine
                 │
                 ▼
[  6. Decision Support Delivery  ] ──> Regional Vulnerability View Export ➔ Hayah Karima Power BI
