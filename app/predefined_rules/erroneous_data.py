import pandas as pd
import sys
from rd_rule import rd_interchange
from hyphen_rule import correct_hyphenation


def introduce_errors_using_rules(sentence):
    """
    Introduce one grammatical error and one typographical error in a given sentence
    using rules defined in the rule files.
    """
    # Step 1: Reverse rd_interchange corrections (e.g., din to rin, daw to raw)
    def reverse_rd_rule(text):
        corrected = rd_interchange(text)    
        # Reverse correction logic: Introduce deliberate errors (e.g., rin -> din, raw -> daw)
        erroneous = corrected.replace('rin', 'din').replace('raw', 'daw')  # Example of reversal
        return erroneous

    # Step 2: Reverse correct_hyphenation (e.g., remove hyphens or insert unnecessary ones)
    def reverse_hyphenation_rule(text):
        corrected = correct_hyphenation(text)
        # Reverse logic: Remove correct hyphens or add incorrect ones
        erroneous = corrected.replace('-', '')  # Remove hyphens for demonstration
        return erroneous

    # Apply one grammatical error
    sentence_with_grammar_error = reverse_rd_rule(sentence)

    # Apply one typographical error
    sentence_with_typo_error = reverse_hyphenation_rule(sentence_with_grammar_error)

    return sentence_with_typo_error

def process_csv(input_output_csv):
    """
    Process a CSV file, injecting errors into sentences where the second column is blank.
    The input is from the first column, and the output is saved in the second column.
    """
    df = pd.read_csv(input_output_csv)

    # Ensure the DataFrame has at least two columns
    if df.shape[1] < 2:
        raise ValueError("The input CSV must have at least two columns.")

    # Process rows where the second column is NaN or empty
    for index, row in df.iterrows():
        if pd.isna(row[1]) or row[1] == '':
            original_sentence = row[0]
            erroneous_sentence = introduce_errors_using_rules(original_sentence)
            df.at[index, df.columns[1]] = erroneous_sentence

    # Save the updated DataFrame back to the same CSV file
    df.to_csv(input_output_csv, index=False)
    print(f"Processed CSV saved to {input_output_csv}")

# Example usage
input_output_csv = 'app/predefined_rules/erraneous_sentences.csv'
process_csv(input_output_csv)
