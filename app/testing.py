import pandas as pd
import os
import logging
from .pantasa_checker import pantasa_checker  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transform_text_to_csv(input_txt_path, output_csv_path):
    """
    Reads sentences from a text file and writes them into a CSV file with a 'sentence' column.

    Args:
    - input_txt_path: Path to the input text file containing sentences.
    - output_csv_path: Path where the output CSV file will be saved.
    """
    # Read the text file
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove any leading/trailing whitespace and empty lines
    sentences = [line.strip() for line in lines if line.strip()]
    
    # Create a DataFrame
    df = pd.DataFrame({'sentence': sentences})
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    
    print(f"Successfully converted '{input_txt_path}' to '{output_csv_path}'")


def test_pantasa(csv_path, jar_path, model_path, rule_path, directory_path, pos_path, output_csv_path):
    """
    Reads sentences from a CSV file, processes each sentence using pantasa_checker,
    and collects the results. If an error occurs, it logs the error and continues processing.
    The results are saved incrementally to avoid data loss in case of unexpected termination.

    Args:
    - csv_path: Path to the input CSV file containing sentences.
    - jar_path, model_path, rule_path, directory_path, pos_path: Paths required for pantasa_checker.
    - output_csv_path: Path where the output CSV file will be saved.

    Returns:
    - results_df: A DataFrame containing original sentences, corrected sentences, and any other outputs.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    if 'sentence' not in df.columns:
        raise ValueError("CSV file must contain a 'sentence' column.")
    
    # Check if output file exists to determine if we are resuming
    if os.path.exists(output_csv_path):
        # Load existing results
        results_df = pd.read_csv(output_csv_path)
        processed_indices = set(results_df.index.tolist())
        logger.info(f"Resuming from existing results. {len(processed_indices)} sentences already processed.")
    else:
        # Initialize an empty DataFrame
        results_df = pd.DataFrame(columns=[
            'Original Sentence', 'Corrected Sentence', 'Incorrect Words', 'Spell Suggestions'
        ])
        processed_indices = set()

    # Process each sentence
    total_sentences = len(df)
    for idx, row in df.iterrows():
        if idx in processed_indices:
            continue  # Skip already processed sentences

        sentence = row['sentence']

        try:
            # Call pantasa_checker
            corrected_sentence, spell_suggestions, incorrect_words = pantasa_checker(
                sentence, jar_path, model_path, rule_path, directory_path, pos_path
            )
        except Exception as e:
            logger.error(f"Error processing sentence at index {idx}: {e}")
            corrected_sentence = None
            spell_suggestions = None
            incorrect_words = None

        # Create a new row as a DataFrame
        new_row = pd.DataFrame([{
            'Original Sentence': sentence,
            'Corrected Sentence': corrected_sentence,
            'Incorrect Words': incorrect_words,
            'Suggestions': spell_suggestions
        }])

        # Concatenate the new row with the results DataFrame
        results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Save progress after each sentence
        results_df.to_csv(output_csv_path, index=False)

        # Optionally, print progress
        print(f"Processed sentence {idx + 1}/{total_sentences}")

    return results_df


if __name__ == "__main__":
    # Paths to your text file and output CSV for erroneous data
    input_txt_path = 'data/raw/piptylines.txt'  # Replace with your error free sentences text file path
    output_csv_path = 'data/raw/error_free_dataset.csv'    # Desired output CSV file path for error free data

    # First, transform your text file into CSV
    transform_text_to_csv(input_txt_path, output_csv_path)

    # Now, run the test using the generated CSV file
    csv_path = output_csv_path  # Use the CSV file just created
    jar_path = 'rules/Libraries/FSPOST/stanford-postagger.jar'
    model_path = 'rules/Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    rule_path = 'data/processed/detailed_hngram.csv'
    directory_path = 'data/raw/dictionary.csv'
    pos_path = 'data/processed/pos_dic'

    # Output CSV file to save results incrementally
    results_output_csv = 'data/processed/error_free_output_result.csv'

    # Run the test
    results_df = test_pantasa(
        csv_path,
        jar_path,
        model_path,
        rule_path,
        directory_path,
        pos_path,
        results_output_csv
    )

    # Display the results
    print(results_df)
