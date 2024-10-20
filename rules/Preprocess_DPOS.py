import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Import tqdm for progress tracking
from Modules.POSDTagger import pos_tag as pos_dtag

def tag_sentence(sentence):
    """Function to tag a single sentence using the POS tagger."""
    return pos_dtag(sentence) if pd.notnull(sentence) else ""

def regenerate_detailed_pos(input_output_csv, max_workers=7):
    # Load the existing CSV file
    df = pd.read_csv(input_output_csv)
    
    # Apply detailed POS tagging in parallel using ThreadPoolExecutor
    if 'Sentence' in df.columns:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use the executor to map the function to each sentence in parallel
            # Wrap the dataframe's sentences with tqdm to track the progress
            detailed_pos_results = list(tqdm(executor.map(tag_sentence, df['Sentence']), total=len(df), desc="Processing Sentences"))
        
        # Update the DataFrame with the results
        df['Detailed_POS'] = detailed_pos_results
    
    # Save the updated CSV to the same file
    df.to_csv(input_output_csv, index=False)
    print(f"Regenerated CSV with detailed POS tags saved to {input_output_csv}")

# Example usage
regenerate_detailed_pos('rules/database/preprocessed.csv')
