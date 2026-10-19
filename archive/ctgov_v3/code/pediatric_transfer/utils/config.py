"""Global configuration and endpoints."""
from __future__ import annotations

SEEDS = [101, 202, 303, 404, 505]
GLOBAL_SEED = 42
BOOTSTRAP_N = 2000
MIN_ADULT_N = 10  # min adult structure-outcome rows for endpoint eligibility
MIN_PED_N = 5
MIN_AT_RISK = 5
CHEMBL_TARGET_CAP = 256
CHEMBL_MIN_COVERAGE = 0.05  # fraction of adult compounds with any activity on target
FAERS_PAGE_LIMIT = 100
FAERS_MAX_PAGES_PER_DOMAIN = 250  # >=25000 reports/domain with openFDA key
CTGOV_PAGE_SIZE = 100
SMOKE_ADULT_TRIALS = 20
SMOKE_PED_TRIALS = 20
SMOKE_FAERS = 100

OUTCOMES = [
    "ANY_SERIOUS",
    "ANY_OTHER_AE",
    "MORTALITY",
    "CARDIAC",
    "HEPATIC",
    "RENAL",
    "NEUROLOGIC",
    "RESPIRATORY",
    "HEMATOLOGIC",
    "GASTROINTESTINAL",
    "DERMATOLOGIC",
    "IMMUNE_INFECTION",
    "ENDOCRINE_METABOLIC",
    "PSYCHIATRIC",
    "CONGENITAL_DEVELOPMENTAL",
]

# MedDRA SOC preferred names → outcome
SOC_TO_OUTCOME = {
    "cardiac disorders": "CARDIAC",
    "hepatobiliary disorders": "HEPATIC",
    "renal and urinary disorders": "RENAL",
    "nervous system disorders": "NEUROLOGIC",
    "respiratory, thoracic and mediastinal disorders": "RESPIRATORY",
    "blood and lymphatic system disorders": "HEMATOLOGIC",
    "gastrointestinal disorders": "GASTROINTESTINAL",
    "skin and subcutaneous tissue disorders": "DERMATOLOGIC",
    "infections and infestations": "IMMUNE_INFECTION",
    "immune system disorders": "IMMUNE_INFECTION",
    "endocrine disorders": "ENDOCRINE_METABOLIC",
    "metabolism and nutrition disorders": "ENDOCRINE_METABOLIC",
    "psychiatric disorders": "PSYCHIATRIC",
    "congenital, familial and genetic disorders": "CONGENITAL_DEVELOPMENTAL",
    "pregnancy, puerperium and perinatal conditions": "CONGENITAL_DEVELOPMENTAL",
}

# Frozen keyword map (term → outcome) when SOC unavailable
KEYWORD_MAP = {
    "death": "MORTALITY",
    "died": "MORTALITY",
    "fatal": "MORTALITY",
    "mortality": "MORTALITY",
    "cardiac": "CARDIAC",
    "myocardial": "CARDIAC",
    "arrhythmia": "CARDIAC",
    "tachycardia": "CARDIAC",
    "bradycardia": "CARDIAC",
    "heart failure": "CARDIAC",
    "qt": "CARDIAC",
    "hepatic": "HEPATIC",
    "liver": "HEPATIC",
    "hepatitis": "HEPATIC",
    "transaminase": "HEPATIC",
    "bilirubin": "HEPATIC",
    "renal": "RENAL",
    "kidney": "RENAL",
    "creatinine": "RENAL",
    "nephro": "RENAL",
    "seizure": "NEUROLOGIC",
    "neuropath": "NEUROLOGIC",
    "headache": "NEUROLOGIC",
    "dizziness": "NEUROLOGIC",
    "stroke": "NEUROLOGIC",
    "convulsion": "NEUROLOGIC",
    "pneumonia": "RESPIRATORY",
    "dyspnea": "RESPIRATORY",
    "respiratory": "RESPIRATORY",
    "cough": "RESPIRATORY",
    "asthma": "RESPIRATORY",
    "anemia": "HEMATOLOGIC",
    "neutropenia": "HEMATOLOGIC",
    "thrombocytopenia": "HEMATOLOGIC",
    "leukopenia": "HEMATOLOGIC",
    "pancytopenia": "HEMATOLOGIC",
    "nausea": "GASTROINTESTINAL",
    "vomiting": "GASTROINTESTINAL",
    "diarrhea": "GASTROINTESTINAL",
    "constipation": "GASTROINTESTINAL",
    "abdominal": "GASTROINTESTINAL",
    "rash": "DERMATOLOGIC",
    "pruritus": "DERMATOLOGIC",
    "dermatitis": "DERMATOLOGIC",
    "urticaria": "DERMATOLOGIC",
    "infection": "IMMUNE_INFECTION",
    "sepsis": "IMMUNE_INFECTION",
    "hypersensitivity": "IMMUNE_INFECTION",
    "anaphylaxis": "IMMUNE_INFECTION",
    "hypothyroid": "ENDOCRINE_METABOLIC",
    "hyperglycemia": "ENDOCRINE_METABOLIC",
    "hypoglycemia": "ENDOCRINE_METABOLIC",
    "diabetes": "ENDOCRINE_METABOLIC",
    "depression": "PSYCHIATRIC",
    "anxiety": "PSYCHIATRIC",
    "suicid": "PSYCHIATRIC",
    "insomnia": "PSYCHIATRIC",
    "congenital": "CONGENITAL_DEVELOPMENTAL",
    "malformation": "CONGENITAL_DEVELOPMENTAL",
    "developmental": "CONGENITAL_DEVELOPMENTAL",
}

CONTROL_KEYWORDS = {
    "placebo", "untreated", "no treatment", "sham", "control", "vehicle",
    "saline", "observation", "best supportive care", "supportive care only",
}
NONDRUG_KEYWORDS = {
    "behavioral", "device", "dietary", "diet", "procedure", "surgery",
    "exercise", "counseling", "psychotherapy", "radiation", "radiotherapy",
    "physiotherapy", "acupuncture", "massage",
}

CTGOV_BASE = "https://clinicaltrials.gov/api/v2/studies"
OPENFDA_BASE = "https://api.fda.gov/drug/event.json"
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
