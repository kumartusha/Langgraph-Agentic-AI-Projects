import pandas as pd
import random
from datetime import datetime
import os

# Define file paths (assuming you put the demo CSVs in a 'data' folder)
DATA_DIR = "./data"
OUTPUT_FILE = f"{DATA_DIR}/NOTEEVENTS.csv"

def generate_synthetic_note(subject_id, hadm_id, dob, gender, admission_time, discharge_time, diagnosis, prescriptions):
    """Generates a synthetic discharge summary based on structured data."""
    
    medications_list = "\n".join([f"- {med}" for med in prescriptions]) if prescriptions else "None recorded."
    
    # Template for a standard Discharge Summary
    note_text = f"""Admission Date:  [{str(admission_time).split(' ')[0]}]       Discharge Date:  [{str(discharge_time).split(' ')[0]}]
Date of Birth:   [{str(dob).split(' ')[0]}]      Sex:  {gender}
Service:  MEDICINE

Chief Complaint:
Admitted for management of {diagnosis}.

History of Present Illness:
The patient is a {gender} presenting with symptoms consistent with {diagnosis}. 
Patient was admitted for further evaluation and management. Initial workup was notable for elevated markers and clinical signs necessitating admission.

Past Medical History:
Hypertension, Hyperlipidemia.

Medications on Admission:
- Aspirin 81mg
- Atorvastatin 40mg

Discharge Diagnosis:
Primary: {diagnosis}

Discharge Medications:
{medications_list}

Discharge Instructions:
1. Take medications as prescribed above.
2. Follow up with PCP in 1-2 weeks.
3. Return to the Emergency Department if symptoms worsen or you experience chest pain, severe shortness of breath, or new concerning symptoms.
"""
    return note_text

def main():
    print("Loading MIMIC-III Demo Data...")
    try:
        patients_df = pd.read_csv(f"{DATA_DIR}/PATIENTS.csv")
        patients_df.columns = patients_df.columns.str.upper()
        
        admissions_df = pd.read_csv(f"{DATA_DIR}/ADMISSIONS.csv")
        admissions_df.columns = admissions_df.columns.str.upper()
        
        prescriptions_df = pd.read_csv(f"{DATA_DIR}/PRESCRIPTIONS.csv")
        prescriptions_df.columns = prescriptions_df.columns.str.upper()
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have downloaded PATIENTS.csv, ADMISSIONS.csv, and PRESCRIPTIONS.csv from PhysioNet into the 'data' folder.")
        return

    notes_data = []
    row_id_counter = 1

    print("Generating synthetic NOTEEVENTS...")
    
    # Iterate through each admission to generate a discharge summary
    for index, admission in admissions_df.iterrows():
        subject_id = admission['SUBJECT_ID']
        hadm_id = admission['HADM_ID']
        
        # Get Patient demographic data
        patient_info = patients_df[patients_df['SUBJECT_ID'] == subject_id].iloc[0]
        dob = str(patient_info['DOB'])
        gender = patient_info['GENDER']
        
        # Get Prescriptions for this admission
        adm_prescriptions = prescriptions_df[prescriptions_df['HADM_ID'] == hadm_id]
        drugs = adm_prescriptions['DRUG'].dropna().unique().tolist()
        
        # Get Diagnosis from admissions (rough primary diagnosis)
        diagnosis = str(admission['DIAGNOSIS']).capitalize() if pd.notnull(admission['DIAGNOSIS']) else "Medical observation"

        # Generate the synthetic note
        synthetic_text = generate_synthetic_note(
            subject_id=subject_id,
            hadm_id=hadm_id,
            dob=dob,
            gender=gender,
            admission_time=admission['ADMITTIME'],
            discharge_time=admission['DISCHTIME'],
            diagnosis=diagnosis,
            prescriptions=drugs
        )

        # Append to our notes list matching the NOTEEVENTS schema
        notes_data.append({
            'ROW_ID': row_id_counter,
            'SUBJECT_ID': subject_id,
            'HADM_ID': hadm_id,
            'CHARTDATE': str(admission['DISCHTIME']).split(' ')[0],
            'CHARTTIME': admission['DISCHTIME'],
            'STORETIME': admission['DISCHTIME'],
            'CATEGORY': 'Discharge summary',
            'DESCRIPTION': 'Report',
            'CGID': random.randint(10000, 99999), # Fake caregiver ID
            'ISERROR': 0,
            'TEXT': synthetic_text
        })
        row_id_counter += 1

    # Create DataFrame and save
    noteevents_df = pd.DataFrame(notes_data)
    noteevents_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSuccess! Generated {len(noteevents_df)} synthetic discharge summaries.")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
