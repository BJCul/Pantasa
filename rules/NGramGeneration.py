import csv
import os
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging

# Set of punctuation marks that can act as sentence terminators
TERMINATOR = set('.!?,')

# Thread lock for safe ID updates
id_lock = Lock()
start_id = 0

def escape_unwrapped_quotes(sentence):
    """Escape double quotes if they are not at the start and end of the sentence."""
    if '"' in sentence and not (sentence.startswith('"') and sentence.endswith('"')):
        sentence = sentence.replace('"', '""')
    return sentence

def wrap_sentence_with_commas(sentence):
    """Wrap sentence with double quotes if it contains a comma and isn't already wrapped."""
    if ',' in sentence and not (sentence.startswith('"') and sentence.endswith('"')):
        sentence = f'"{sentence}"'
    return sentence

def redo_escape_and_wrap(sentence):
    """Reapply double quotes and wrapping rules if conditions are met, avoiding redundant escapes."""
    sentence = escape_unwrapped_quotes(sentence)
    return wrap_sentence_with_commas(sentence)

def undo_escape_and_wrap(sentence):
    """Revert double quotes and wrapping for processing."""
    if sentence.startswith('"') and sentence.endswith('"'):
        sentence = sentence[1:-1]
    return sentence.replace('""', '"')

def get_latest_id(output_file):
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as csv_file:
                reader = csv.DictReader(csv_file)
                ids = [int(row['N-Gram_ID']) for row in reader if row['N-Gram_ID'].isdigit()]
            return max(ids, default=0) + 1 if ids else 0
        except Exception as e:
            print(f"Error reading {output_file} for ID: {e}")
    return 0


def split_sentence_preserving_abbreviations(sentence):
    """
    Split a sentence into tokens while preserving abbreviations, hyphenated words,
    and treating punctuation separately.
    """
    tokens = []
    buffer = ""
    for char in sentence:
        if char.isalnum() or (char == '-' and buffer and buffer[-1].isalnum()):
            # Accumulate alphanumeric characters and hyphens as part of a word
            buffer += char
        elif char == '.':
            # Check if it might be part of an abbreviation
            if buffer and buffer[0].isupper() and len(buffer) <= 3:  # Likely an abbreviation
                buffer += char
            else:
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
                tokens.append(char)  # Add period as a separate token
        elif char in TERMINATOR or char in {'"', "'", ':', ';'}:
            # Handle sentence terminators or other punctuation
            if buffer:
                tokens.append(buffer)
                buffer = ""
            tokens.append(char)  # Add punctuation as a separate token
        elif char.isspace():
            if buffer:
                tokens.append(buffer)
                buffer = ""
        else:
            # Handle unexpected or special characters
            if buffer:
                tokens.append(buffer)
                buffer = ""
            tokens.append(char)  # Add special characters as separate tokens

    # Add any remaining buffer as the last token
    if buffer:
        tokens.append(buffer)

    return tokens


def generate_ngrams(sequence, rough_pos_sequence, detailed_pos_sequence, lemma_sequence, ngram_range=(2, 7)):
    """
    Generate n-grams from the processed token sequences.
    """
    tokens = split_sentence_preserving_abbreviations(sequence)
    rough_pos_tags = rough_pos_sequence.split()
    detailed_pos_tags = detailed_pos_sequence.split()
    lemmas = split_sentence_preserving_abbreviations(lemma_sequence)

    # Check for length mismatches
    if len(tokens) != len(lemmas):
        raise ValueError("Token and Lemma sequences must have the same length.")
    if len(rough_pos_tags) != len(detailed_pos_tags):
        raise ValueError("Rough POS and Detailed POS sequences must have the same length.")

    ngram_sequences = defaultdict(list)
    for n in range(ngram_range[0], ngram_range[1] + 1):
        token_ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        lemma_ngrams = [tuple(lemmas[i:i + n]) for i in range(len(lemmas) - n + 1)]
        rough_pos_ngrams = [tuple(rough_pos_tags[i:i + n]) for i in range(len(rough_pos_tags) - n + 1)]
        detailed_pos_ngrams = [tuple(detailed_pos_tags[i:i + n]) for i in range(len(detailed_pos_tags) - n + 1)]

        with id_lock:
            global start_id
            for t_ngram, l_ngram, rp_ngram, dp_ngram in zip(token_ngrams, lemma_ngrams, rough_pos_ngrams, detailed_pos_ngrams):
                ngram_sequences[n].append((f"{start_id:06d}", n, ' '.join(rp_ngram), ' '.join(dp_ngram), ' '.join(t_ngram), ' '.join(l_ngram)))
                start_id += 1

    return ngram_sequences


def process_row(row):
    """
    Process a single row and generate n-grams from it.
    """
    try:
        sentence = undo_escape_and_wrap(row['Sentence'])
        rough_pos = row['Rough_POS']
        detailed_pos = row['Detailed_POS']
        lemmatized = undo_escape_and_wrap(row['Lemmatized_Sentence'])

        ngram_data = generate_ngrams(sentence, rough_pos, detailed_pos, lemmatized)

        results = []
        for ngram_size, ngrams_list in ngram_data.items():
            for ngram_tuple in ngrams_list:
                ngram_id, ngram_size, rough_pos_str, detailed_pos_str, ngram_str, lemma_str = ngram_tuple
                ngram_str = redo_escape_and_wrap(ngram_str)
                lemma_str = redo_escape_and_wrap(lemma_str)
                
                results.append({
                    'N-Gram_ID': ngram_id,
                    'N-Gram_Size': ngram_size,
                    'RoughPOS_N-Gram': rough_pos_str,
                    'DetailedPOS_N-Gram': detailed_pos_str,
                    'N-Gram': ngram_str,
                    'Lemma_N-Gram': lemma_str
                })
        return results
    except ValueError as e:
        # Log the error and skip the problematic row
        print(f"Skipping row due to error: {e}")
        return []

def process_csv(input_file, output_file, start_row=0):
    """
    Process the input CSV file row by row, generate n-grams, and write to output.
    """
    global start_id
    start_id = get_latest_id(output_file)

    results = []
    max_workers = os.cpu_count()
    with open(input_file, 'r', encoding='utf-8') as csv_file:
        reader = list(csv.DictReader(csv_file))
        rows_to_process = reader[start_row:]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_row, row): row for row in rows_to_process}
            with tqdm(total=len(futures), desc="Processing Rows") as pbar:
                for future in as_completed(futures):
                    results.extend(future.result())
                    pbar.update(1)

    with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
        fieldnames = ['N-Gram_ID', 'N-Gram_Size', 'RoughPOS_N-Gram', 'DetailedPOS_N-Gram', 'N-Gram', 'Lemma_N-Gram']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logging.info(f"Processed data saved to {output_file}")


# Example usage
input_csv = 'rules/database/preprocessed.csv'
output_csv = 'rules/database/ngram.csv'
start_row = 0  # Start processing from the start row
process_csv(input_csv, output_csv, start_row)
