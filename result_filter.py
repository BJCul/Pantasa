import pandas as pd
import os

def filter_and_save_csv(input_file, output_file, num_tp, num_tn, num_fp, num_fn1, num_fn2):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Ensure the required columns exist
    if not all(col in df.columns for col in ["Detection Result", "Correction Result", "Original Sentence"]):
        raise ValueError("One or more required columns ('Detection Result', 'Correction Result', 'Original Sentence') are missing in the CSV file.")
    
    # Load existing data to check for duplicates before filtering
    existing_sentences = set()
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        if "Original Sentence" in existing_df.columns:
            existing_sentences = set(existing_df["Original Sentence"].tolist())
    
    # Remove existing sentences before filtering
    df = df[~df["Original Sentence"].isin(existing_sentences)]
    
    # Filter and select specified number of rows for each type
    tp_df = df[(df["Correction Result"] == "TP") & (df["Detection Result"] == "TP")].head(num_tp)
    tn_df = df[(df["Correction Result"] == "TN") & (df["Detection Result"] == "TN")].head(num_tn)
    fp_df = df[(df["Correction Result"] == "FP") & (df["Detection Result"] == "FP")].head(num_fp)
    fn1_df = df[(df["Correction Result"] == "FN") & (df["Detection Result"] == "FN")].head(num_fn1)
    fn2_df = df[(df["Correction Result"] == "FN") & (df["Detection Result"] == "TP")].head(num_fn2)
    
    # Concatenate all filtered dataframes
    final_df = pd.concat([tp_df, tn_df, fp_df, fn1_df, fn2_df])
    
    # Save to CSV
    if not final_df.empty:
        if os.path.exists(output_file):
            final_df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            final_df.to_csv(output_file, index=False)
    
    print(f"Saved {len(final_df)} new rows to {output_file}")

# Example usage
input_csv = "data/processed/raw_result/correction_evaluation_result33.csv"
output_csv = "data/processed/raw_result/raw_data.csv"
num_tp = 24 
num_tn = 0
num_fp = 0  
num_fn1 = 0  # FN for both detection and correction
num_fn2 = 0 # TP for detection, FN for correction

filter_and_save_csv(input_csv, output_csv, num_tp, num_tn, num_fp, num_fn1, num_fn2)