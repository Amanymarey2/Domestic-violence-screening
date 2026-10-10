#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import sqlite3
import pandas as pd
import time
import random
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np

# Load client configuration normally
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_PATH = "egypt_uhis_ews.db"
MODEL = "gpt-4o-mini"
BATCH_DELAY = 1.1
MAX_RETRIES = 3

# -------------------------------------------------
# Lists for Dynamic Context Injection (Prevents Semantic Collapse)
# -------------------------------------------------
ROUTINE_REASONS = [
    "دور برد وإنفلونزا وكحة شديدة", "آلام أسفل الظهر بسبب الشغل في الغيط",
    "تجديد علاج الضغط الشهري", "متابعة السكر العادية وصرف الروشتة",
    "صداع ودوخة من الشمس", "مغص كلوي ومشاكل في الأملاح أميبا",
    "نزلة معوية وإسهال بقاله يومين", "تغيير على جرح قديم ناشف",
    "كشف روتيني واستشارة عامة للوحدة", "صرف فيتامينات ومكملات غذائية",
    "متابعة تنظيم الأسرة وصرف الحبوب"
]

UPPER_EGYPT_SLANG = [
    "العيان جالي الوحدة يشكي من", "جاء لتوقيع الكشف الطبي بسبب",
    "الحالة كشف باطنة عشان", "تذكرة استقبال كشف عظام", 
    "مراجعة النهاردة والشكوى كانت", "ال مريض باعتة التمريض تشتكي من",
    "حضر للعيادة يعاني من"
]

# -------------------------------------------------
# Dynamic Prompt Generator
# -------------------------------------------------
def create_prompt(row):
    # Select a random entry variant to scramble opening structures
    opening_style = random.choice(UPPER_EGYPT_SLANG)
    
    if row["reported_abuse_in_notes"] == 1 or row["injury_type"] == "Assault" or row.get("hymen_exam_requested", 0) == 1:
        situation = "high"
        guidance = (
            "The patient presents with notable physical findings (bruises, trauma patterns, multiple lacerations). "
            "Document the objective physical injuries clinically. Mention that the timeline or history provided "
            "by the family/companion feels detached, vague, or inconsistent (e.g., 'التاريخ مش راكب', 'القصة غير واضحة'). "
            "Maintain strict clinical professionalism. Do not use direct emotional words."
        )
    elif row["injury_present"] == 1 and (row["malnutrition_flag"] == 1 or row["depression_flag"] == 1 or row["anxiety_flag"] == 1):
        situation = "moderate"
        guidance = (
            "The patient shows minor injuries claimed as a home accident, combined with weight loss or flat affect/anxiety. "
            "Write in an understated, slightly vague clinical shorthand. Mention the home accident, but also "
            "note the subtle constitutional issues (e.g., 'ملامح قلقة', 'شكلها خاسة', 'تكرار الحوادث المنزلية'). "
            "Do not state any suspicions out loud."
        )
    elif row["injury_present"] == 1:
        situation = "mild"
        # Scramble home injury details so it isn't always "وقع في البيت"
        accident_detail = random.choice(["اتكعبلت في حصيره", "وقع من فوق الحمار", "اتزحلق في المطبخ", "حرق زيت خفيف", "وقع من على السلم"])
        situation = "mild"
        guidance = f"""Simple single injury clearly matching an ordinary accident: '{accident_detail}'. Keep it extremely brief, raw, and typical of a chaotic clinic log."""
        
    elif row["malnutrition_flag"] == 1 or row["depression_flag"] == 1 or row["anxiety_flag"] == 1:
        situation = "soft"
        guidance = (
            "The patient presents for primary care, but notes indicate weight change, fatigue, or low mood. "
            "Document the physical or mood complaint as a side note. Keep it non-alarming."
        )
    else:
        # ---- FIX FOR THE CLONING TRAIN: Inject Random Non-Abuse Medical Contexts ----
        situation = "routine"
        random_complaint = random.choice(ROUTINE_REASONS)
        guidance = f"""This is a routine visit. Instead of writing a blank template, document that the patient came explicitly for this medical reason: '{random_complaint}'. Keep the clinical assessment completely normal and clear."""

    prompt = f"""
You are a busy, exhausted family medicine physician working in a rural primary healthcare unit in Upper Egypt (rural Minya, Sohag, or Assiut). 

Write ONE unique clinical note only (1 to 2 sentences maximum).

Strict Formatting and Linguistic Constraints:
1. START the note immediately using variations like: "{opening_style} ..." to disrupt standard line setups.
2. Mix conversational Egyptian Arabic with standard abbreviated medical shorthand (e.g., 'BP', 'علاج', 'روشتة', 'استشارة', 'صرف').
3. DO NOT use the following banned template phrases: 'كل حاجة تمام', 'لا شكاوى', 'ضغط وسكر طبيعيين', 'مريض جاي روتيني', 'مفيش شكاوى'. If you want to say they are stable, mix it up (e.g., 'العلامات الحيوية مستقرة', 'الفحص العضوي سليم الحين').
4. Write like a real doctor rushing through handwriting on a chart—loose grammar, quick fragmented observations.

Situation Profile: {situation}
Instructions: {guidance}

Context Indicators (Do not copy these terms directly):
- Injury: {"Yes" if row["injury_present"] else "No"} | Type: {row["injury_type"] if pd.notna(row["injury_type"]) else "None"}
- Nutrition Drop: {"Yes" if row["malnutrition_flag"] else "No"}
- Mood Distress: {"Yes" if (row["depression_flag"] or row["anxiety_flag"]) else "No"}

Return ONLY the text of the note. No labels, no quotation marks, no markdown blocks.
"""
    return prompt.strip()

# -------------------------------------------------
# Main Generation
# -------------------------------------------------
def generate_all_remaining_notes():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql("""
        SELECT 
            visit_id,
            injury_present,
            injury_type,
            injury_severity,
            malnutrition_flag,
            depression_flag,
            anxiety_flag,
            reported_abuse_in_notes,
            hymen_exam_requested
        FROM visits
        WHERE provider_notes_short IS NULL 
           OR TRIM(provider_notes_short) = ''
    """, conn)

    total = len(df)
    print(f"Total remaining visits to process: {total}")

    if total == 0:
        print("No remaining visits need notes.")
        conn.close()
        return

    updated = 0
    failed = 0

    for idx, row in tqdm(df.iterrows(), total=total, desc="Generating notes"):
        prompt = create_prompt(row)
        success = False

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that writes realistic Egyptian clinical notes."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=120
                )

                note = response.choices[0].message.content.strip()
                note = note.strip('"').strip("'").strip()

                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE visits
                    SET provider_notes_short = ?
                    WHERE visit_id = ?
                """, (note, row["visit_id"]))
                conn.commit()

                updated += 1
                success = True
                break

            except Exception as e:
                print(f"\nError on {row['visit_id']} (attempt {attempt+1}): {e}")
                time.sleep(2 + attempt)

        if not success:
            failed += 1

        time.sleep(BATCH_DELAY)

    conn.close()

    print("\n" + "="*50)
    print("GENERATION COMPLETED")
    print("="*50)
    print(f"Successfully updated : {updated}")
    print(f"Failed               : {failed}")
    print(f"Total processed      : {updated + failed}")


if __name__ == "__main__":
    generate_all_remaining_notes()


