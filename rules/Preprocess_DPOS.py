import pandas as pd
from Modules.POSDTagger import pos_tag as pos_dtag

def update_detailed_pos(input_output_csv):
    # Load the existing CSV file
    df = pd.read_csv(input_output_csv)
    
    # Check if the 'Detailed_POS' column exists and has "None" values to replace
    if 'Detailed_POS' in df.columns:
        # Replace "None" with generated detailed POS tags using the pos_dtag function
        df['Detailed_POS'] = df.apply(
            lambda row: pos_dtag(row['Sentence']) if row['Detailed_POS'] == 'None' else row['Detailed_POS'], axis=1
        )
    
    # Save the updated CSV to the same file
    df.to_csv(input_output_csv, index=False)
    print(f"Updated CSV with detailed POS tags saved to {input_output_csv}")

update_detailed_pos('/content/Pantasa/rules/database/preprocessed.csv')
