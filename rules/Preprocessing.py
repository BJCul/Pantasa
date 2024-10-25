import csv
import os
from collections import defaultdict
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the regular expression for tokenizing word sequences including punctuation
regex = r"[^.!?,;:—\s][^.!?,;:—]*[.!?,;:—]?['\"]?(?=\s|$)"

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

def undo_escape_and_wrap(sentence):
    """Revert double quotes and wrapping for processing."""
    if sentence.startswith('"') and sentence.endswith('"'):
        sentence = sentence[1:-1]
    return sentence.replace('""', '"')

def redo_escape_and_wrap(sentence):
    """Reapply double quotes and wrapping rules if conditions are met, avoiding redundant escapes."""
    # First, escape unwrapped quotes
    sentence = escape_unwrapped_quotes(sentence)
    # Then, wrap the sentence if it contains a comma
    return wrap_sentence_with_commas(sentence)

def separate_punctuation(sequence):
    """Separate punctuation attached at the beginning or end of words."""
    tokens = []
    for word in sequence.split():
        if re.match(r'^[^\w\s]', word):  # Leading punctuation
            tokens.append(word[0])  # Add punctuation as a separate token
            tokens.append(word[1:])  # Add the remaining word
        elif re.match(r'[^\w\s]$', word):  # Trailing punctuation
            tokens.append(word[:-1])  # Add word without punctuation
            tokens.append(word[-1])  # Add punctuation as a separate token
        else:
            tokens.append(word)
    return tokens

def custom_ngrams(sequence, n):
    """Generate n-grams from a sequence."""
    return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]

def generate_ngrams(word_sequence, rough_pos_sequence, detailed_pos_sequence, lemma_sequence, ngram_range=(1, 7), add_newline=False, start_id=0):
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
    
    current_id = start_id
    
    for n in range(ngram_range[0], ngram_range[1] + 1):
        word_n_grams = custom_ngrams(words, n)
        rough_pos_n_grams = custom_ngrams(rough_pos_tags, n)
        detailed_pos_n_grams = custom_ngrams(detailed_pos_tags, n)
        lemma_n_grams = custom_ngrams(lemmas, n)
        
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
                
                ngram_id = f"{current_id:06d}"
                ngram_sequences[n].append((ngram_id, n, rough_pos_str, detailed_pos_str, ngram_str, lemma_str))
                current_id += 1
    
    return ngram_sequences, current_id

def process_row(row, start_id):
    """Process a single row and generate n-grams from it."""
    sentence = undo_escape_and_wrap(row['Sentence'])
    rough_pos = row['Rough_POS']
    detailed_pos = row['Detailed_POS']
    lemmatized = undo_escape_and_wrap(row['Lemmatized_Sentence'])
    
    ngram_data, updated_start_id = generate_ngrams(sentence, rough_pos, detailed_pos, lemmatized, start_id=start_id)
    
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
    return results, updated_start_id

def process_csv(input_file, output_file):
    results = []
    start_id = 0
    max_workers = os.cpu_count()

    with open(input_file, 'r', encoding='utf-8') as csv_file:
        reader = list(csv.DictReader(csv_file))

        # Progress bar for processing rows with parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_row, row, start_id): row for row in reader}
            with tqdm(total=len(futures), desc="Processing Rows") as pbar:
                for future in as_completed(futures):
                    try:
                        row_results, start_id = future.result()
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
process_csv(input_csv, output_csv)
