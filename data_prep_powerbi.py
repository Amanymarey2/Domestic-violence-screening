#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sqlite3
import pandas as pd

DB_PATH = "egypt_uhis_ews.db"
OUTPUT_BI_TABLE = "bi_vulnerability_analytics"
CSV_EXPORT_PATH = "hayah_karima_decision_support_data.csv"

print("🔄 Generating unified analytics view for Power BI...")

conn = sqlite3.connect(DB_PATH)

# FIXED: Joined facilities (f) using households (h.facility_id) instead of p.facility_id
query = """
    SELECT 
        p.visit_id,
        p.visit_date,
        p.final_risk_score,
        p.final_severity_level,
        p.prev_injury_count_6m,
        p.days_since_injury,
        p.nlp_severity_score,
        i.sex,
        i.age_at_registration,
        i.relationship_to_head,
        i.education_level,
        i.employment_status,
        h.income_level,
        h.income_source,
        h.crowding_index,
        h.geographic_isolation_score,
        h.has_substance_use_flag,
        h.step_parent_or_orphan_flag,
        f.facility_name,
        f.markaz,
        f.governorate,
        f.urbanization
    FROM visit_final_predictions p
    JOIN individuals i ON p.individual_id = i.individual_id
    JOIN households h ON p.household_id = h.household_id
    JOIN facilities f ON h.facility_id = f.facility_id
"""

bi_df = pd.read_sql(query, conn)

# 1. Save to SQLite database for storage
bi_df.to_sql(OUTPUT_BI_TABLE, conn, if_exists="replace", index=False)
print(f"✅ Saved analytics view to SQL table: '{OUTPUT_BI_TABLE}'")

# 2. Export to CSV for easy Power BI import
bi_df.to_csv(CSV_EXPORT_PATH, index=False, encoding="utf-8-sig")
print(f"✅ Exported flat analytical data to: '{CSV_EXPORT_PATH}'")

print("\n--- GEOGRAPHICAL RISK DISTRIBUTION PREVIEW ---")
geo_summary = bi_df.groupby(['governorate', 'final_severity_level']).size().unstack(fill_value=0)
print(geo_summary)

conn.close()


# In[ ]:



