import csv
from collections import defaultdict
import re
import pandas as pd

# Define punctuation POS tags
punctuation_pos_tags = {
    ".": "PMP",
    "!": "PME",
    "?": "PMQ",
    ",": "PMC",
    ";": "PMSC",
    ":": "PMSC",
    "@": "PMS", "/": "PMS", "+": "PMS", "*": "PMS", 
    "(": "PMS", ")": "PMS", "\"": "PMS", "'": "PMS", 
    "~": "PMS", "&": "PMS", "%": "PMS", "$": "PMS", 
    "#": "PMS", "=": "PMS", "-": "PMS"
}

# Function to tokenize with punctuation handling
def tokenize_with_punctuation(sequence):
    """Tokenize a sequence, treating punctuation marks as separate tokens."""
    regex = r"\w+|[.!?,;:—]"
    return re.findall(regex, sequence)

# Function to align POS tags with tokens
def align_pos_tags_with_tokens(tokens, pos_tags):
    """Align the POS tags with each token including punctuations."""
    aligned_pos_tags = []
    pos_index = 0

    for token in tokens:
        if token in punctuation_pos_tags:
            aligned_pos_tags.append(punctuation_pos_tags[token])
        else:
            if pos_index < len(pos_tags):
                aligned_pos_tags.append(pos_tags[pos_index])
                pos_index += 1
            else:
                aligned_pos_tags.append("UNK")  # Use UNK for any unexpected length issues

    return aligned_pos_tags

# Function to generate n-grams
def custom_ngrams(sequence, n):
    """Generate n-grams from a sequence."""
    return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]

# Main n-gram generation function
def generate_ngrams_full_range(word_sequence, rough_pos_sequence, detailed_pos_sequence, lemma_sequence, ngram_range=(2, 7), add_newline=False, start_id=0):
    ngram_sequences = defaultdict(list)
    
    # Tokenize the word sequence to include punctuation as separate tokens
    words = tokenize_with_punctuation(word_sequence)
    rough_pos_tags = align_pos_tags_with_tokens(words, rough_pos_sequence.split())
    detailed_pos_tags = align_pos_tags_with_tokens(words, detailed_pos_sequence.split())
    lemmas = tokenize_with_punctuation(lemma_sequence)
    
    # Ensure the lengths match
    if len(rough_pos_tags) != len(words) or len(detailed_pos_tags) != len(words) or len(lemmas) != len(words):
        raise ValueError("Sequences lengths do not match")

    current_id = start_id
    
    for n in range(ngram_range[0], ngram_range[1] + 1):
        word_n_grams = custom_ngrams(words, n)
        rough_pos_n_grams = custom_ngrams(rough_pos_tags, n)
        detailed_pos_n_grams = custom_ngrams(detailed_pos_tags, n)
        lemma_n_grams = custom_ngrams(lemmas, n)
        
        for word_gram, rough_pos_gram, detailed_pos_gram, lemma_gram in zip(word_n_grams, rough_pos_n_grams, detailed_pos_n_grams, lemma_n_grams):
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

# Process the entire CSV file
def process_csv(input_file, output_file):
    results = []
    start_id = 0

    with open(input_file, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        
        for row in reader:
            try:
                # Extract relevant columns
                sentence = row['Sentence']
                rough_pos = row['Rough_POS']
                detailed_pos = row['Detailed_PO']
                lemmatized = row['Lemmatized']

                # Generate n-grams with full range
                ngram_data, start_id = generate_ngrams_full_range(sentence, rough_pos, detailed_pos, lemmatized, ngram_range=(2, 7), start_id=start_id)
                
                for ngram_size, ngrams_list in ngram_data.items():
                    for ngram_tuple in ngrams_list:
                        ngram_id, ngram_size, rough_pos_str, detailed_pos_str, ngram_str, lemma_str = ngram_tuple
                        results.append({
                            'N-Gram_ID': ngram_id,
                            'N-Gram_Size': ngram_size,
                            'RoughPOS_N-Gram': rough_pos_str,
                            'DetailedPOS_N-Gram': detailed_pos_str,
                            'N-Gram': ngram_str,
                            'Lemma_N-Gram': lemma_str
                        })
            except ValueError as e:
                print(f"Skipping line due to error: {e}")
            except KeyError as e:
                print(f"Skipping line due to missing column: {e}")
    
    # Write results to output CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
        fieldnames = ['N-Gram_ID', 'N-Gram_Size', 'RoughPOS_N-Gram', 'DetailedPOS_N-Gram', 'N-Gram', 'Lemma_N-Gram']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

# Example usage
input_csv = 'rules/database/preprocessed.csv'  # Replace with your input CSV path
output_csv = 'rules/database/ngram.csv'  # Replace with your desired output CSV path
process_csv(input_csv, output_csv)
