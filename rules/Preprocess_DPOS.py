import pandas as pd
from Modules.POSDTagger import pos_tag as pos_dtag

def regenerate_detailed_pos(input_output_csv):
    # Load the existing CSV file
    df = pd.read_csv(input_output_csv)
    
    # Apply detailed POS tagging to every sentence in the "Sentence" column
    if 'Sentence' in df.columns:
        df['Detailed_POS'] = df['Sentence'].apply(lambda sentence: pos_dtag(sentence) if pd.notnull(sentence) else "")
    
    # Save the updated CSV to the same file
    df.to_csv(input_output_csv, index=False)
    print(f"Regenerated CSV with detailed POS tags saved to {input_output_csv}")


regenerate_detailed_pos('/content/Pantasa/rules/database/preprocessed.csv')
