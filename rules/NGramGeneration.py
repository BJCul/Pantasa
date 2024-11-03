import csv
import os
from collections import defaultdict
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Define the regular expression for tokenizing word sequences including punctuation
regex = r"[^.!?,;:—\s][^.!?,;:—]*[.!?,;:—]?['\"]?(?=\s|$)"

# Global variables for unique ID and thread lock
start_id = 0
id_lock = Lock()

def get_latest_id(output_file):
    """Retrieve the latest N-Gram_ID from the output file if it exists, to continue ID generation."""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as csv_file:
                reader = csv.DictReader(csv_file)
                ids = [int(row['N-Gram_ID']) for row in reader if row['N-Gram_ID'].isdigit()]
            return max(ids, default=0) + 1 if ids else 0
        except Exception as e:
            print(f"Error reading {output_file} for ID: {e}")
    return 0

def redo_escape_and_wrap(sentence):
    """Escape double quotes for CSV compatibility and wrap with quotes if necessary."""
    sentence = sentence.replace('"', '""')
    if ',' in sentence or (sentence.startswith('""') and sentence.endswith('""')):
        sentence = f'"{sentence}"'
    return sentence

def undo_escape_and_wrap(sentence):
    """Revert double quotes and wrapping for processing."""
    if sentence.startswith('"') and sentence.endswith('"'):
        sentence = sentence[1:-1]
    return sentence.replace('""', '"')

def separate_punctuation(sequence):
    """Separate punctuation attached at the beginning or end of words."""
    tokens = []
    for word in sequence.split():
        if re.match(r'^[^\w\s]', word):  # Leading punctuation
            tokens.append(word[0])
            tokens.append(word[1:])
        elif re.match(r'[^\w\s]$', word):  # Trailing punctuation
            tokens.append(word[:-1])
            tokens.append(word[-1])
        else:
            tokens.append(word)
    return tokens

def custom_ngrams(sequence, n):
    """Generate n-grams from a sequence."""
    return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]

def generate_ngrams(word_sequence, rough_pos_sequence, detailed_pos_sequence, lemma_sequence, ngram_range=(1, 7), add_newline=False):
    ngram_sequences = defaultdict(list)
    
    words = separate_punctuation(word_sequence)
    rough_pos_tags = rough_pos_sequence.split()
    detailed_pos_tags = detailed_pos_sequence.split()
    lemmas = separate_punctuation(lemma_sequence)
    
    # Check length matching
    if len(words) != len(lemmas):
        raise ValueError("Words and Lemmas sequence lengths do not match")
    if len(rough_pos_tags) != len(detailed_pos_tags):
        raise ValueError("Rough POS and Detailed POS sequence lengths do not match")
    
    for n in range(ngram_range[0], ngram_range[1] + 1):
        word_n_grams = custom_ngrams(words, n)
        rough_pos_n_grams = custom_ngrams(rough_pos_tags, n)
        detailed_pos_n_grams = custom_ngrams(detailed_pos_tags, n)
        lemma_n_grams = custom_ngrams(lemmas, n)
        
        with id_lock:  # Ensure thread-safe access to the shared ID counter
            global start_id
            
            for word_gram, rough_pos_gram, detailed_pos_gram, lemma_gram in zip(word_n_grams, rough_pos_n_grams, detailed_pos_n_grams, lemma_n_grams):
                unique_detailed_tags = set(tag for detailed_tag in detailed_pos_gram for tag in detailed_tag.split('_'))
                
                if len(unique_detailed_tags) >= 4:
                    ngram_str = ' '.join(word_gram)
                    lemma_str = ' '.join(lemma_gram)
                    rough_pos_str = ' '.join(rough_pos_gram)
                    detailed_pos_str = ' '.join(detailed_pos_gram)
                    
                    if add_newline:
                        ngram_str += '\n'
                        lemma_str += '\n'
                        rough_pos_str += '\n'
                        detailed_pos_str += '\n'
                    
                    ngram_id = f"{start_id:06d}"
                    ngram_sequences[n].append((ngram_id, n, rough_pos_str, detailed_pos_str, ngram_str, lemma_str))
                    start_id += 1  # Increment the global ID
    
    return ngram_sequences

def process_row(row):
    """Process a single row and generate n-grams from it."""
    sentence = undo_escape_and_wrap(row['Sentence'])
    rough_pos = row['Rough_POS']
    detailed_pos = row['Detailed_POS']
    lemmatized = undo_escape_and_wrap(row['Lemmatized_Sentence'])
    
    ngram_data = generate_ngrams(sentence, rough_pos, detailed_pos, lemmatized)
    
    results = []
    for ngram_size, ngrams_list in ngram_data.items():
        for ngram_tuple in ngrams_list:
            ngram_id, ngram_size, rough_pos_str, detailed_pos_str, ngram_str, lemma_str = ngram_tuple
            
            # Reapply escape and wrap before saving
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

def process_csv(input_file, output_file, start_row=0):
    global start_id
    start_id = get_latest_id(output_file)  # Initialize with the last ID in output

    results = []
    max_workers = os.cpu_count()

    with open(input_file, 'r', encoding='utf-8') as csv_file:
        reader = list(csv.DictReader(csv_file))

        # Skip to the specified start_row
        rows_to_process = reader[start_row:]
        
        # Progress bar for processing rows with parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_row, row): row for row in rows_to_process}
            with tqdm(total=len(futures), desc="Processing Rows") as pbar:
                for future in as_completed(futures):
                    try:
                        row_results = future.result()
                        results.extend(row_results)
                    except ValueError as e:
                        print(f"Skipping row due to error: {e}")
                    pbar.update(1)

    # Write results to output CSV
    with open(output_file, 'a', newline='', encoding='utf-8') as out_file:
        fieldnames = ['N-Gram_ID', 'N-Gram_Size', 'RoughPOS_N-Gram', 'DetailedPOS_N-Gram', 'N-Gram', 'Lemma_N-Gram']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        if os.stat(output_file).st_size == 0:
            writer.writeheader()
        writer.writerows(results)

    print(f"Processed data saved to {output_file}")

# Example usage
input_csv = 'rules/database/preprocessed.csv'
output_csv = 'rules/database/ngram.csv'
start_row = 14684  # Start processing from the start row
process_csv(input_csv, output_csv, start_row)
