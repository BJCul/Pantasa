import csv
from collections import defaultdict
import re

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

def build_ngram_frequency_and_export(file_path, output_path):
    # Dictionaries to store word frequencies and their associated POS and Lemma
    word_data = defaultdict(lambda: {'POS': None, 'Lemma': None, 'Frequency': 0})
    current_id = 0  # Initialize unique ID counter
    
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            # Undo escape and wrapping for each field to ensure correct processing
            ngram = undo_escape_and_wrap(row['N-Gram'])
            pos = undo_escape_and_wrap(row['DetailedPOS_N-Gram'])
            lemma = undo_escape_and_wrap(row['Lemma_N-Gram'])
            
            # Split the n-gram into individual words
            words = ngram.split()
            pos_tags = pos.split()
            lemmas = lemma.split()

            # Update the frequency dictionary for each word in the n-gram
            for word, word_pos, word_lemma in zip(words, pos_tags, lemmas):
                word_data[word]['POS'] = word_pos  # Store POS tag
                word_data[word]['Lemma'] = word_lemma  # Store Lemma
                word_data[word]['Frequency'] += 1  # Increment word frequency

    # Write the result to the output CSV file
    with open(output_path, 'w', newline='', encoding='utf-8') as out_file:
        fieldnames = ['ID', 'Word', 'POS', 'Lemma', 'Frequency']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for word, data in word_data.items():
            writer.writerow({
                'ID': current_id,
                'Word': redo_escape_and_wrap(word),
                'POS': redo_escape_and_wrap(data['POS']),
                'Lemma': redo_escape_and_wrap(data['Lemma']),
                'Frequency': data['Frequency']
            })
            current_id += 1  # Increment ID for each new word entry

    print(f"Word frequency data saved to {output_path}")

# File paths for input and output
input_path = 'rules/database/ngram.csv'
output_path = 'rules/database/word_frequency.csv'

# Generate the CSV with n-gram frequencies
build_ngram_frequency_and_export(input_path, output_path)

print(f"Frequency summary CSV saved to {output_path}")
