[200~-- Enable foreign keys execution context in SQLite
PRAGMA foreign_keys = ON;

-- 1. FACILITIES TABLE
CREATE TABLE facilities (
    facility_id TEXT PRIMARY KEY,
    facility_name TEXT NOT NULL,
    facility_type TEXT NOT NULL,
    village_code TEXT NOT NULL,
    markaz TEXT NOT NULL,
    governorate TEXT NOT NULL,
    is_uhis_digitized INTEGER DEFAULT 1,
    urbanization TEXT DEFAULT 'Rural'
);

-- 2. HOUSEHOLDS TABLE
CREATE TABLE households (
    household_id TEXT PRIMARY KEY,
    facility_id TEXT NOT NULL,
    head_of_household_id TEXT, -- Updated downstream to prevent insert dependency loops
    registration_date TEXT NOT NULL,
    household_size INTEGER NOT NULL,
    housing_type TEXT NOT NULL,
    crowding_index REAL NOT NULL,
    income_level TEXT NOT NULL,
    income_source TEXT NOT NULL,
    has_substance_use_flag INTEGER DEFAULT 0,
    education_mismatch_flag INTEGER DEFAULT 0,
    step_parent_or_orphan_flag INTEGER DEFAULT 0,
    disability_in_household INTEGER DEFAULT 0,
    geographic_isolation_score REAL NOT NULL,
    last_updated TEXT NOT NULL,
    FOREIGN KEY (facility_id) REFERENCES facilities(facility_id) ON DELETE CASCADE
);

-- 3. INDIVIDUALS TABLE
CREATE TABLE individuals (
    individual_id TEXT PRIMARY KEY,
    household_id TEXT NOT NULL,
    national_id_hash TEXT NOT NULL,
    relationship_to_head TEXT NOT NULL,
    sex TEXT NOT NULL,
    date_of_birth TEXT NOT NULL,
    age_at_registration INTEGER NOT NULL,
    marital_status TEXT NOT NULL,
    education_level TEXT NOT NULL,
    employment_status TEXT NOT NULL,
    is_head_of_household INTEGER DEFAULT 0,
    chronic_conditions TEXT,
    allergies TEXT,
    vaccination_status TEXT NOT NULL,
    registration_date TEXT NOT NULL,
    is_active INTEGER DEFAULT 1,
    FOREIGN KEY (household_id) REFERENCES households(household_id) ON DELETE CASCADE
);

-- 4. VISITS (THE CLINICAL EARLY WARNING ENGINE)
CREATE TABLE visits (
    visit_id TEXT PRIMARY KEY,
    individual_id TEXT NOT NULL,
    household_id TEXT NOT NULL,
    facility_id TEXT NOT NULL,
    visit_date TEXT NOT NULL,
    visit_type TEXT NOT NULL,
    chief_complaint TEXT NOT NULL,
    injury_present INTEGER DEFAULT 0,
    injury_type TEXT,
    injury_severity TEXT,
    injury_location TEXT,
    days_since_injury INTEGER,
    history_consistency_flag TEXT,
    burn_present INTEGER DEFAULT 0,
    burn_degree TEXT,
    weight_kg REAL,
    bmi REAL,
    weight_change_since_last REAL,
    malnutrition_flag INTEGER DEFAULT 0,
    alopecia_flag INTEGER DEFAULT 0,
    hypopigmentation_flag INTEGER DEFAULT 0,
    neglected_chronic_flag INTEGER DEFAULT 0,
    mental_health_score INTEGER,
    anxiety_flag INTEGER DEFAULT 0,
    depression_flag INTEGER DEFAULT 0,
    sleep_disturbance_flag INTEGER DEFAULT 0,
    hymen_exam_requested INTEGER DEFAULT 0,
    reported_abuse_in_notes INTEGER DEFAULT 0,
    neurologic_or_psych_disorder INTEGER DEFAULT 0,
    referral_token_issued INTEGER DEFAULT 0,
    referral_destination TEXT,
    diagnosis_codes TEXT,
    provider_notes_short TEXT, -- Loaded from your cleaned LLM script
    created_at TEXT NOT NULL,
    FOREIGN KEY (individual_id) REFERENCES individuals(individual_id) ON DELETE CASCADE,
    FOREIGN KEY (household_id) REFERENCES households(household_id),
    FOREIGN KEY (facility_id) REFERENCES facilities(facility_id)
);

-- 5. LIFE EVENTS TABLE
CREATE TABLE life_events (
    event_id TEXT PRIMARY KEY,
    individual_id TEXT NOT NULL,
    household_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_date TEXT NOT NULL,
    related_individual_id TEXT,
    FOREIGN KEY (individual_id) REFERENCES individuals(individual_id) ON DELETE CASCADE,
    FOREIGN KEY (household_id) REFERENCES households(household_id)
);

-- -------------------------------------------------
-- PERFORMANCE & SPEED INDEX OPTIMIZATIONS
-- -------------------------------------------------
-- Critical Index for the 180-Day Rolling Windows computation we built earlier!
CREATE INDEX idx_visits_rolling_timeline 
ON visits (individual_id, visit_date, injury_present);

-- Critical Index for Joining patient history timelines across households
CREATE INDEX idx_individuals_household_lookup 
ON individuals (household_id);

