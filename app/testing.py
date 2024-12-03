import pandas as pd

from .pantasa_checker import pantasa_checker

def test_pantasa(csv_path, jar_path, model_path, rule_path, directory_path, pos_path):
    """
    Reads sentences from a CSV file, processes each sentence using pantasa_checker,
    and collects the results.
    
    Args:
    - csv_path: Path to the input CSV file containing sentences.
    - jar_path, model_path, rule_path, directory_path, pos_path: Paths required for pantasa_checker.
    
    Returns:
    - results_df: A DataFrame containing original sentences, corrected sentences, and any other outputs.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    if 'sentence' not in df.columns:
        raise ValueError("CSV file must contain a 'sentence' column.")
    
    # Lists to store results
    original_sentences = []
    corrected_sentences = []
    spell_suggestions_list = []
    incorrect_words_list = []
    
    # Process each sentence
    for idx, row in df.iterrows():
        sentence = row['sentence']
        original_sentences.append(sentence)
        
        # Call pantasa_checker
        corrected_sentence, spell_suggestions, incorrect_words = pantasa_checker(
            sentence, jar_path, model_path, rule_path, directory_path, pos_path
        )
        
        corrected_sentences.append(corrected_sentence)
        spell_suggestions_list.append(spell_suggestions)
        incorrect_words_list.append(incorrect_words)
        
        # Optionally, print progress
        print(f"Processed sentence {idx + 1}/{len(df)}")
    
    # Create a results DataFrame
    results_df = pd.DataFrame({
        'Original Sentence': original_sentences,
        'Corrected Sentence': corrected_sentences,
        'Incorrect Words': incorrect_words_list,
        'Spell Suggestions': spell_suggestions_list
    })
    
    return results_df

# Example usage:
csv_path = 'data/raw/dataset_test.csv'  # Replace with your CSV file path
jar_path = 'rules/Libraries/FSPOST/stanford-postagger.jar'
model_path = 'rules/Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
rule_path = 'data/processed/detailed_hngram.csv'
directory_path = 'data/raw/dictionary.csv'
pos_path = 'data/processed/pos_dic'

# Run the test
results_df = test_pantasa(csv_path, jar_path, model_path, rule_path, directory_path, pos_path)

# Optionally, save results to a new CSV file
results_df.to_csv('data/processed/output_result.csv', index=False)

# Display the results
print(results_df)
