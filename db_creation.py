#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sqlite3
import pandas as pd
from pathlib import Path
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import uuid
from tqdm import tqdm

# -------------------------------------------------
# 1. Create database and schema
# -------------------------------------------------
db_path = Path("egypt_uhis_ews.db")
conn = sqlite3.connect(db_path)

cursor = conn.cursor()

# Enable foreign keys
cursor.execute("PRAGMA foreign_keys = ON;")


# -------------------------------------------------
# Configuration - Adjust these numbers as needed
# -------------------------------------------------
DB_PATH = "egypt_uhis_ews.db"

GOVERNORATES = {
    "Minya": ["Mallawi", "Maghagha", "Bani Mazar", "Samalut", "Abu Qurqas", "Deir Mawas"],
    "Sohag": ["Sohag", "Akhmim", "Girga", "Tahta", "Gerga"],
    "Assiut": ["Assiut", "Manfalut", "Abnub", "Dayrout", "El Qusiya"],
    "Sharqia": ["Zagazig", "Bilbeis", "Abu Hammad", "Minya al-Qamh", "Faqous"]
}

FACILITIES_PER_GOV = 6          # Total facilities ≈ 24
HOUSEHOLDS_PER_FACILITY = 18    # Total households ≈ 430
AVG_VISITS_PER_HOUSEHOLD = 18   # Total visits ≈ 7,500–9,000

START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2025, 12, 31)

fake = Faker()
np.random.seed(42)
random.seed(42)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Clear previous data (optional – comment if you want to keep old Minya data)
# for table in ["visits", "life_events", "individuals", "households", "facilities"]:
#     cursor.execute(f"DELETE FROM {table}")
# conn.commit()

print("Starting expanded synthetic data generation...\n")

# -------------------------------------------------
# 1. Facilities
# -------------------------------------------------
facilities = []
fac_counter = 1

for gov, markaz_list in GOVERNORATES.items():
    for i in range(FACILITIES_PER_GOV):
        markaz = markaz_list[i % len(markaz_list)]
        fac_id = f"PHCU-{gov[:3].upper()}-{fac_counter:03d}"
        
        facilities.append({
            "facility_id": fac_id,
            "facility_name": f"وحدة طب أسرة - {markaz}",
            "facility_type": "Family Medicine Unit",
            "village_code": f"VIL-{gov[:3].upper()}-{fac_counter:03d}",
            "markaz": markaz,
            "governorate": gov,
            "is_uhis_digitized": 1,
            "urbanization": "Rural"
        })
        fac_counter += 1

df_fac = pd.DataFrame(facilities)
df_fac.to_sql("facilities", conn, if_exists="append", index=False)
print(f"Created {len(df_fac)} facilities")

# -------------------------------------------------
# 2. Households
# -------------------------------------------------
households = []
hh_counter = 1

for fac in tqdm(facilities, desc="Creating households"):
    for _ in range(HOUSEHOLDS_PER_FACILITY):
        hh_id = f"HH-{fac['governorate'][:3].upper()}-{hh_counter:05d}"
        
        # 30-40% of households are higher risk
        is_vulnerable = random.random() < 0.38
        
        households.append({
            "household_id": hh_id,
            "facility_id": fac["facility_id"],
            "head_of_household_id": None,
            "registration_date": fake.date_between(start_date="-4y", end_date="-6m"),
            "household_size": random.randint(3, 9),
            "housing_type": random.choice(["Owned", "Rented", "Shared"]),
            "crowding_index": round(random.uniform(1.1, 4.8), 2),
            "income_level": "Low" if is_vulnerable else random.choice(["Low", "Lower-middle", "Middle"]),
            "income_source": random.choice(["Informal", "Agriculture", "Remittance", "Formal"]),
            "has_substance_use_flag": 1 if (is_vulnerable and random.random() < 0.42) else 0,
            "education_mismatch_flag": 1 if (is_vulnerable and random.random() < 0.28) else 0,
            "step_parent_or_orphan_flag": 1 if (is_vulnerable and random.random() < 0.22) else 0,
            "disability_in_household": 1 if random.random() < 0.13 else 0,
            "geographic_isolation_score": round(random.uniform(0.15, 0.95), 2),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        hh_counter += 1

df_hh = pd.DataFrame(households)
df_hh.to_sql("households", conn, if_exists="append", index=False)
print(f"Created {len(df_hh)} households")

# -------------------------------------------------
# 3. Individuals
# -------------------------------------------------
individuals = []
individual_ids_by_hh = {}

for hh in tqdm(households, desc="Creating individuals"):
    hh_id = hh["household_id"]
    size = hh["household_size"]
    members = []
    
    # Head (mostly male)
    head_id = str(uuid.uuid4())[:10]
    members.append({
        "individual_id": head_id,
        "household_id": hh_id,
        "national_id_hash": fake.sha1()[:14],
        "relationship_to_head": "Head",
        "sex": "Male",
        "date_of_birth": fake.date_of_birth(minimum_age=30, maximum_age=58),
        "age_at_registration": random.randint(30, 58),
        "marital_status": "Married",
        "education_level": random.choice(["None", "Primary", "Preparatory", "Secondary"]),
        "employment_status": random.choice(["Working", "Non-working"]),
        "is_head_of_household": 1,
        "chronic_conditions": random.choice([None, "Hypertension", "Diabetes", "Diabetes,Hypertension"]),
        "allergies": None,
        "vaccination_status": "Up to date",
        "registration_date": hh["registration_date"],
        "is_active": 1
    })
    
    # Spouse
    if size >= 2:
        spouse_id = str(uuid.uuid4())[:10]
        members.append({
            "individual_id": spouse_id,
            "household_id": hh_id,
            "national_id_hash": fake.sha1()[:14],
            "relationship_to_head": "Spouse",
            "sex": "Female",
            "date_of_birth": fake.date_of_birth(minimum_age=22, maximum_age=52),
            "age_at_registration": random.randint(22, 52),
            "marital_status": "Married",
            "education_level": random.choice(["None", "Primary", "Preparatory", "Secondary", "Higher"]),
            "employment_status": random.choice(["Non-working", "Working"]),
            "is_head_of_household": 0,
            "chronic_conditions": None,
            "allergies": None,
            "vaccination_status": "Up to date",
            "registration_date": hh["registration_date"],
            "is_active": 1
        })
    
    # Children + others
    while len(members) < size:
        child_id = str(uuid.uuid4())[:10]
        age = random.randint(0, 18)
        members.append({
            "individual_id": child_id,
            "household_id": hh_id,
            "national_id_hash": fake.sha1()[:14],
            "relationship_to_head": "Child",
            "sex": random.choice(["Female", "Male"]),
            "date_of_birth": fake.date_of_birth(minimum_age=age, maximum_age=age),
            "age_at_registration": age,
            "marital_status": "Unmarried",
            "education_level": "None" if age < 6 else random.choice(["Primary", "Preparatory"]),
            "employment_status": "Child",
            "is_head_of_household": 0,
            "chronic_conditions": None,
            "allergies": None,
            "vaccination_status": "Up to date",
            "registration_date": hh["registration_date"],
            "is_active": 1
        })
    
    individuals.extend(members)
    individual_ids_by_hh[hh_id] = [m["individual_id"] for m in members]
    
    # Update head of household
    cursor.execute(
        "UPDATE households SET head_of_household_id = ? WHERE household_id = ?",
        (head_id, hh_id)
    )

df_ind = pd.DataFrame(individuals)
df_ind.to_sql("individuals", conn, if_exists="append", index=False)
print(f"Created {len(df_ind)} individuals")

# -------------------------------------------------
# 4. Visits (with controlled risk patterns)
# -------------------------------------------------
visits = []
visit_counter = 1

for hh in tqdm(households, desc="Creating visits"):
    hh_id = hh["household_id"]
    members = individual_ids_by_hh[hh_id]
    
    is_high_risk = (
        hh["has_substance_use_flag"] == 1 or
        hh["education_mismatch_flag"] == 1 or
        hh["step_parent_or_orphan_flag"] == 1 or
        hh["income_level"] == "Low"
    )
    
    n_visits = random.randint(14, 28) if is_high_risk else random.randint(6, 14)
    
    for _ in range(n_visits):
        person_id = random.choice(members)
        visit_date = fake.date_between(start_date=START_DATE, end_date=END_DATE)
        
        # Default values
        injury_present = 0
        injury_type = None
        injury_severity = None
        burn_present = 0
        malnutrition_flag = 0
        depression_flag = 0
        anxiety_flag = 0
        alopecia_flag = 0
        reported_abuse = 0
        hymen_exam = 0
        
        # Inject patterns
        if is_high_risk and random.random() < 0.37:
            injury_present = 1
            injury_type = random.choice(["Household accident", "Fall", "Burn", "Household accident"])
            injury_severity = random.choice(["Mild", "Moderate", "Moderate"])
            
            if injury_type == "Burn":
                burn_present = 1
            
            if random.random() < 0.32:
                malnutrition_flag = 1
            if random.random() < 0.28:
                depression_flag = 1
                anxiety_flag = 1
            if random.random() < 0.15:
                alopecia_flag = 1
        
        # Rare Tier-3
        if is_high_risk and random.random() < 0.035:
            injury_present = 1
            injury_type = "Assault"
            injury_severity = "Severe"
            reported_abuse = 1
            if random.random() < 0.45:
                hymen_exam = 1
        
        visits.append({
            "visit_id": f"VIS-{visit_counter:06d}",
            "individual_id": person_id,
            "household_id": hh_id,
            "facility_id": hh["facility_id"],
            "visit_date": visit_date,
            "visit_type": random.choice(["Routine", "Follow-up", "Acute", "Preventive"]),
            "chief_complaint": injury_type if injury_present else "General check",
            "injury_present": injury_present,
            "injury_type": injury_type,
            "injury_severity": injury_severity,
            "injury_location": random.choice(["Upper limb", "Head/Face/Neck", "Trunk", "Lower limb"]) if injury_present else None,
            "days_since_injury": random.randint(0, 12) if injury_present else None,
            "history_consistency_flag": random.choice(["Consistent", "Inconsistent", "Changed"]) if injury_present else None,
            "burn_present": burn_present,
            "burn_degree": random.choice(["1st", "2nd"]) if burn_present else None,
            "weight_kg": round(random.uniform(42, 92), 1),
            "bmi": round(random.uniform(17.5, 34), 1),
            "weight_change_since_last": round(random.uniform(-5.5, 2.5), 1),
            "malnutrition_flag": malnutrition_flag,
            "alopecia_flag": alopecia_flag,
            "hypopigmentation_flag": 1 if (alopecia_flag and random.random() < 0.4) else 0,
            "neglected_chronic_flag": 1 if random.random() < 0.07 else 0,
            "mental_health_score": random.randint(3, 6) if depression_flag else random.randint(0, 2),
            "anxiety_flag": anxiety_flag,
            "depression_flag": depression_flag,
            "sleep_disturbance_flag": 1 if depression_flag else 0,
            "hymen_exam_requested": hymen_exam,
            "reported_abuse_in_notes": reported_abuse,
            "neurologic_or_psych_disorder": 1 if random.random() < 0.04 else 0,
            "referral_token_issued": 1 if injury_severity == "Severe" else 0,
            "referral_destination": "Secondary" if injury_severity == "Severe" else None,
            "diagnosis_codes": None,
            "provider_notes_short": None,          # Will be filled later with LLM notes
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        visit_counter += 1

df_vis = pd.DataFrame(visits)
df_vis.to_sql("visits", conn, if_exists="append", index=False)
print(f"Created {len(df_vis)} visits")

# -------------------------------------------------
# 5. Life Events
# -------------------------------------------------
life_events = []
event_counter = 1

high_risk_households = [hh for hh in households if hh["income_level"] == "Low" or hh["has_substance_use_flag"] == 1]
selected_hh = random.sample(high_risk_households, min(120, len(high_risk_households)))

for hh in selected_hh:
    members = individual_ids_by_hh[hh["household_id"]]
    person_id = members[1] if len(members) > 1 else members[0]
    
    life_events.append({
        "event_id": f"EVT-{event_counter:05d}",
        "individual_id": person_id,
        "household_id": hh["household_id"],
        "event_type": random.choice(["Marriage", "Divorce", "Orphaned", "Remarriage", "Partner_migration"]),
        "event_date": fake.date_between(start_date=START_DATE, end_date=datetime(2024, 8, 1)),
        "related_individual_id": None
    })
    event_counter += 1

df_le = pd.DataFrame(life_events)
df_le.to_sql("life_events", conn, if_exists="append", index=False)
print(f"Created {len(df_le)} life events")

conn.commit()
conn.close()

print("\n" + "="*50)
print("Expanded structured data generation completed!")
print("="*50)


# In[ ]:


conn = sqlite3.connect("egypt_uhis_ews.db")
print(pd.read_sql("SELECT governorate, COUNT(*) FROM facilities GROUP BY governorate", conn))
print(pd.read_sql("SELECT COUNT(*) AS households FROM households", conn))
print(pd.read_sql("SELECT COUNT(*) AS individuals FROM individuals", conn))
print(pd.read_sql("SELECT COUNT(*) AS visits FROM visits", conn))
conn.close()


# In[ ]:


DB_PATH = "egypt_uhis_ews.db"
conn = sqlite3.connect(DB_PATH)

print("=" * 70)
print("  EXPANDED DATASET VERIFICATION REPORT")
print("=" * 70)

# -------------------------------------------------
# 1. Overall Volume
# -------------------------------------------------
print("\n1. OVERALL VOLUME")
print("-" * 50)

tables = ["facilities", "households", "individuals", "visits", "life_events"]
for table in tables:
    count = pd.read_sql(f"SELECT COUNT(*) AS cnt FROM {table}", conn).iloc[0, 0]
    print(f"{table:<20}: {count:>8,}")

# -------------------------------------------------
# 2. Geographic Distribution
# -------------------------------------------------
print("\n2. GEOGRAPHIC DISTRIBUTION")
print("-" * 50)

print("\nFacilities per Governorate:")
print(pd.read_sql("""
    SELECT governorate, COUNT(*) AS facilities
    FROM facilities
    GROUP BY governorate
    ORDER BY facilities DESC
""", conn).to_string(index=False))

print("\nHouseholds per Governorate:")
print(pd.read_sql("""
    SELECT f.governorate, COUNT(h.household_id) AS households
    FROM households h
    JOIN facilities f ON h.facility_id = f.facility_id
    GROUP BY f.governorate
    ORDER BY households DESC
""", conn).to_string(index=False))

print("\nVisits per Governorate:")
print(pd.read_sql("""
    SELECT f.governorate, COUNT(v.visit_id) AS visits
    FROM visits v
    JOIN facilities f ON v.facility_id = f.facility_id
    GROUP BY f.governorate
    ORDER BY visits DESC
""", conn).to_string(index=False))

# -------------------------------------------------
# 3. Household Risk Profile
# -------------------------------------------------
print("\n3. HOUSEHOLD RISK PROFILE")
print("-" * 50)

print(pd.read_sql("""
    SELECT 
        income_level,
        COUNT(*) AS households,
        ROUND(AVG(has_substance_use_flag)*100, 1) AS pct_substance,
        ROUND(AVG(education_mismatch_flag)*100, 1) AS pct_edu_mismatch,
        ROUND(AVG(step_parent_or_orphan_flag)*100, 1) AS pct_step_orphan,
        ROUND(AVG(disability_in_household)*100, 1) AS pct_disability
    FROM households
    GROUP BY income_level
    ORDER BY households DESC
""", conn).to_string(index=False))

# -------------------------------------------------
# 4. Visit-Level Patterns (Core for modeling)
# -------------------------------------------------
print("\n4. VISIT-LEVEL PATTERNS")
print("-" * 50)

print(pd.read_sql("""
    SELECT 
        COUNT(*) AS total_visits,
        SUM(injury_present) AS injury_visits,
        ROUND(SUM(injury_present)*100.0 / COUNT(*), 2) AS pct_injury,
        SUM(malnutrition_flag) AS malnutrition_visits,
        SUM(depression_flag OR anxiety_flag) AS mental_health_visits,
        SUM(reported_abuse_in_notes) AS explicit_abuse_flags,
        SUM(hymen_exam_requested) AS hymen_exam_requests,
        SUM(burn_present) AS burn_visits
    FROM visits
""", conn).to_string(index=False))

print("\nInjury Type Distribution:")
print(pd.read_sql("""
    SELECT injury_type, COUNT(*) AS count
    FROM visits
    WHERE injury_present = 1
    GROUP BY injury_type
    ORDER BY count DESC
""", conn).to_string(index=False))

# -------------------------------------------------
# 5. High-Risk Households Analysis
# -------------------------------------------------
print("\n5. HIGH-RISK HOUSEHOLDS (Top 10 by concerning visits)")
print("-" * 50)

print(pd.read_sql("""
    SELECT 
        v.household_id,
        f.governorate,
        h.income_level,
        COUNT(*) AS total_visits,
        SUM(v.injury_present) AS injury_count,
        SUM(v.malnutrition_flag) AS malnutrition_count,
        SUM(v.depression_flag OR v.anxiety_flag) AS mental_count,
        SUM(v.reported_abuse_in_notes) AS abuse_flags
    FROM visits v
    JOIN households h ON v.household_id = h.household_id
    JOIN facilities f ON v.facility_id = f.facility_id
    GROUP BY v.household_id
    HAVING injury_count >= 3 OR abuse_flags >= 1
    ORDER BY abuse_flags DESC, injury_count DESC
    LIMIT 10
""", conn).to_string(index=False))

# -------------------------------------------------
# 6. Data Quality Checks
# -------------------------------------------------
print("\n6. DATA QUALITY CHECKS")
print("-" * 50)

print(pd.read_sql("""
    SELECT 
        SUM(CASE WHEN provider_notes_short IS NULL OR TRIM(provider_notes_short) = '' THEN 1 ELSE 0 END) AS visits_without_notes,
        SUM(CASE WHEN individual_id IS NULL THEN 1 ELSE 0 END) AS missing_individual_id,
        SUM(CASE WHEN household_id IS NULL THEN 1 ELSE 0 END) AS missing_household_id,
        MIN(visit_date) AS earliest_visit,
        MAX(visit_date) AS latest_visit
    FROM visits
""", conn).to_string(index=False))

# -------------------------------------------------
# 7. Summary
# -------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

summary = pd.read_sql("""
    SELECT 
        (SELECT COUNT(*) FROM facilities) AS facilities,
        (SELECT COUNT(*) FROM households) AS households,
        (SELECT COUNT(*) FROM individuals) AS individuals,
        (SELECT COUNT(*) FROM visits) AS visits,
        (SELECT COUNT(*) FROM life_events) AS life_events,
        (SELECT ROUND(AVG(cnt),1) FROM (
            SELECT household_id, COUNT(*) AS cnt FROM visits GROUP BY household_id
        )) AS avg_visits_per_household
""", conn)

print(summary.T.to_string(header=False))

conn.close()
print("\nVerification completed successfully.")

