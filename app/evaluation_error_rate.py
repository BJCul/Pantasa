import pandas as pd
import os
import logging
from pantasa_checker import pantasa_checker  # Adjust import if needed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pantasa_on_dataset(input_csv, output_csv,
                           jar_path, model_path, rule_path, directory_path, pos_path,
                           save_interval=1):
    # Read the input CSV
    df = pd.read_csv(input_csv)

    # Check required columns
    if 'Original Sentence' not in df.columns or 'Gold Sentence' not in df.columns:
        raise ValueError("Input CSV must contain 'Original Sentence' and 'Gold Sentence' columns.")

    # Check if output file exists to determine if resuming
    if os.path.exists(output_csv):
        # Load existing results and find how many are processed
        results_df = pd.read_csv(output_csv)
        processed_indices = set(results_df.index)
        logger.info(f"Resuming from existing results. {len(processed_indices)} sentences already processed.")
    else:
        # Initialize empty DataFrame with required columns
        results_df = pd.DataFrame(columns=[
            'Original Sentence',
            'Gold Sentence',
            'Corrected Sentence',
            'Incorrect Words',
            'Suggestions'
        ])
        processed_indices = set()

    total_predictions = len(df)

    for idx, row in df.iterrows():
        if idx in processed_indices:
            # Already processed this index (if resuming)
            continue

        original = row['Original Sentence']
        gold = row['Gold Sentence']

        try:
            # Run pantasa_checker on the original sentence
            corrected_sentence, spell_suggestions, incorrect_words = pantasa_checker(
                original, jar_path, model_path, rule_path, directory_path, pos_path
            )

            # Convert suggestions/words to strings if needed
            incorrect_words_str = str(incorrect_words) if incorrect_words is not None else "[]"
            suggestions_str = str(spell_suggestions) if spell_suggestions is not None else ""
        except Exception as e:
            logger.error(f"Error processing sentence at index {idx}: {e}")
            # In case of error, record None/empty fields to indicate failure
            corrected_sentence = None
            incorrect_words_str = "[]"
            suggestions_str = ""

        # Add the result to the results DataFrame
        new_row = pd.DataFrame([{
            'Original Sentence': original,
            'Gold Sentence': gold,
            'Corrected Sentence': corrected_sentence,
            'Incorrect Words': incorrect_words_str,
            'Suggestions': suggestions_str
        }], index=[idx])

        results_df = pd.concat([results_df, new_row]).sort_index()

        # Save progress after each sentence or at intervals
        if (idx + 1) % save_interval == 0 or (idx + 1) == total_predictions:
            results_df.to_csv(output_csv, index=False)
            logger.info(f"Saved progress after processing sentence {idx + 1}/{total_predictions}")

        print(f"Processed sentence {idx + 1}/{total_predictions}")
    
    logger.info(f"All sentences processed. Final results saved to '{output_csv}'.")

if __name__ == "__main__":

    jar_path = 'rules/Libraries/FSPOST/stanford-postagger.jar'
    model_path = 'rules/Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    rule_path = 'data/processed/detailed_hngram.csv'
    directory_path = 'data/raw/dictionary.csv'
    pos_path = 'data/processed/pos_dic'

    # Input CSV with Original and Gold sentences
    input_csv = 'data/processed/first_two_columns_cleaned_final.csv'
    # Output CSV file
    output_csv = 'data/processed/erroneous_output_result.csv'

    # Run the script with a save interval of 1 (save after every sentence)
    run_pantasa_on_dataset(input_csv, output_csv,
                           jar_path, model_path, rule_path, directory_path, pos_path,
                           save_interval=1)