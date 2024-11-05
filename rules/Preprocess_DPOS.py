import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Import tqdm for progress tracking
from Modules.POSDTagger import pos_tag as pos_dtag

def tag_sentence(sentence):
    """Function to tag a single sentence using the POS tagger."""
    return pos_dtag(sentence) if pd.notnull(sentence) else ""

def regenerate_detailed_pos(input_output_csv, max_workers=7, start_row=0):
    # Load the existing CSV file
    df = pd.read_csv(input_output_csv)
    
    # Slice the DataFrame starting from the specified row
    df_to_process = df[start_row:]
    
    # Apply detailed POS tagging in parallel using ThreadPoolExecutor
    if 'Sentence' in df_to_process.columns:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use the executor to map the function to each sentence in parallel
            # Wrap the dataframe's sentences with tqdm to track the progress
            detailed_pos_results = list(tqdm(executor.map(tag_sentence, df_to_process['Sentence']), total=len(df_to_process), desc="Processing Sentences"))
        
        # Update the DataFrame with the results for the processed rows
        df_to_process['Detailed_POS'] = detailed_pos_results
        
        # Replace only the processed part in the original DataFrame
        df.update(df_to_process)
    
    # Save the updated CSV to the same file
    df.to_csv(input_output_csv, index=False)
    print(f"Regenerated CSV with detailed POS tags starting from row {start_row} saved to {input_output_csv}")



def run_preprocessing():
    # Define your file paths here
    preprocess_csv = "rules/database/preprocessed.csv"     # File to save the preprocessed output

    # Start the preprocessing
    regenerate_detailed_pos(preprocess_csv, max_workers=8, start_row=0)

# Automatically run when the script is executed
if __name__ == "__main__":
    run_preprocessing()