import pandas as pd
import os
import logging
from .pantasa_checker import pantasa_checker  # Adjust this import if needed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_erroneous_output_result(
    input_csv, output_csv,
    jar_path, model_path, rule_path, directory_path, pos_path,
    save_interval=1
):
    # Read the input CSV containing original and gold sentences
    df = pd.read_csv(input_csv)

    # Check if output file exists to determine if we are resuming
    if os.path.exists(output_csv):
        # Load existing results
        results_df = pd.read_csv(output_csv)
        processed_indices = set(results_df.index.tolist())
        logger.info(f"Resuming from existing results. {len(processed_indices)} sentences already processed.")
    else:
        # Initialize an empty DataFrame
        results_df = pd.DataFrame(columns=[
            'Original Sentence',
            'Corrected Sentence',
            'Incorrect Words',
            'Spell Suggestions'
        ])
        processed_indices = set()

    total_sentences = len(df)
    for idx, row in df.iterrows():
        if idx in processed_indices:
            continue  # Skip already processed sentences if resuming

        original_sentence = row['Original Sentence']

        try:
            # Run the pantasa_checker on the original sentence
            corrected_sentence, spell_suggestions, incorrect_words = pantasa_checker(
                original_sentence, jar_path, model_path, rule_path, directory_path, pos_path
            )
        except Exception as e:
            logger.error(f"Error processing sentence at index {idx}: {e}")
            # If an error occurs, we still record the original sentence but with None for corrections
            corrected_sentence = None
            spell_suggestions = None
            incorrect_words = None

        # Append the result as a new row
        new_row = pd.DataFrame([{
            'Original Sentence': original_sentence,
            'Corrected Sentence': corrected_sentence,
            'Incorrect Words': incorrect_words,
            'Spell Suggestions': spell_suggestions
        }])

        results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Save progress after each sentence or after every save_interval sentences
        if (idx + 1) % save_interval == 0 or (idx + 1) == total_sentences:
            results_df.to_csv(output_csv, index=False)
            logger.info(f"Saved progress after processing sentence {idx + 1}/{total_sentences}")

        print(f"Processed sentence {idx + 1}/{total_sentences}")

    logger.info(f"All sentences processed. Final results saved to '{output_csv}'")

if __name__ == "__main__":
    # Update these paths as needed
    input_csv = 'data/processed/erroneous_gold.csv'  # The CSV you provided
    output_csv = 'data/processed/correction_erroneous_output_result.csv'

    # Paths required for pantasa_checker
    jar_path = 'rules/Libraries/FSPOST/stanford-postagger.jar'
    model_path = 'rules/Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    rule_path = 'data/processed/detailed_hngram.csv'
    directory_path = 'data/raw/dictionary.csv'
    pos_path = 'data/processed/pos_dic'

    # Generate the erroneous_output_result.csv
    # Set save_interval to 1 to save after every sentence
    generate_erroneous_output_result(
        input_csv,
        output_csv,
        jar_path,
        model_path,
        rule_path,
        directory_path,
        pos_path,
        save_interval=1
    )
