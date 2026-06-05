import pandas as pd
from src.graph import build_graph

def test_pipeline():
    print("Loading test data...")
    try:
        notes_df = pd.read_csv("data/NOTEEVENTS.csv")
        prescriptions_df = pd.read_csv("data/PRESCRIPTIONS.csv")
    except FileNotFoundError:
        print("Error: Could not find CSV files in data/ directory. Did you generate the synthetic data?")
        return

    if len(notes_df) == 0:
        print("No notes available.")
        return
        
    # Grab the first patient record to test
    test_note = notes_df.iloc[0]
    subject_id = test_note['SUBJECT_ID']
    hadm_id = test_note['HADM_ID']
    raw_text = test_note['TEXT']
    
    # Get ground truth meds from prescriptions table
    patient_meds = prescriptions_df[prescriptions_df['HADM_ID'] == hadm_id]['DRUG'].tolist()
    
    print(f"\n==========================================")
    print(f" Testing Pipeline for Subject {subject_id}")
    print(f"==========================================\n")
    print(f"Ground Truth Meds loaded: {patient_meds}\n")
    
    app = build_graph()
    
    inputs = {
        "raw_clinical_note": raw_text,
        "ground_truth_meds": patient_meds
    }
    
    print("Starting LangGraph workflow execution...")
    # Invoke runs the whole graph and returns the final state
    final_state = app.invoke(inputs)
    
    print("\n==========================================")
    print(" FINAL DRAFT SUMMARY")
    print("==========================================")
    print(final_state.get("draft_summary"))
    
    print("\n==========================================")
    print(" SAFETY & RECONCILIATION FEEDBACK")
    print("==========================================")
    print(f"Safety Approved: {final_state.get('safety_approved')}")
    print(f"Safety Feedback: {final_state.get('safety_feedback')}")
    print("\nReconciliation Flags:")
    for flag in final_state.get("reconciliation_flags", []):
        print(f" - {flag}")

if __name__ == "__main__":
    test_pipeline()




# How to run.
# pip install -r requirements.txt
# python3 -m src.main
