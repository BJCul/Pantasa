import csv
from collections import defaultdict
import re
from tqdm import tqdm

# Define the regular expression for tokenizing word sequences including punctuation
regex = r"[^.!?,;:—\s][^.!?,;:—]*[.!?,;:—]?['\"]?(?=\s|$)"

def escape_unwrapped_quotes(sentence):
    """Escape double quotes if they are not at the start and end of the sentence."""
    if '"' in sentence and not (sentence.startswith('"') and sentence.endswith('"')):
        sentence = sentence.replace('"', '""')
    elif '"' in sentence and sentence.startswith('"') and sentence.endswith('"'):
        sentence = sentence.replace('"', '"""')
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
    sentence = sentence.replace('""', '"').replace('"""', '"')
    return sentence

def redo_escape_and_wrap(sentence):
    """Reapply double quotes and wrapping rules if conditions are met."""
    sentence = escape_unwrapped_quotes(sentence)
    sentence = wrap_sentence_with_commas(sentence)
    return sentence

def custom_ngrams(sequence, n):
    """Generate n-grams from a sequence."""
    return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]

def generate_ngrams(word_sequence, rough_pos_sequence, detailed_pos_sequence, lemma_sequence, ngram_range=(1, 7), add_newline=False, start_id=0):
    ngram_sequences = defaultdict(list)
    
    words = word_sequence.split()
    rough_pos_tags = rough_pos_sequence.split()
    detailed_pos_tags = detailed_pos_sequence.split()
    lemmas = lemma_sequence.split()
    
    if len(words) != len(rough_pos_tags) or len(rough_pos_tags) != len(detailed_pos_tags) or len(words) != len(lemmas):
        raise ValueError("Sequence lengths do not match")

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

def process_csv(input_file, output_file):
    results = []
    start_id = 0

    with open(input_file, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        
        for row in tqdm(reader, desc="Processing Rows"):
            # Undo escape and wrap for processing
            sentence = undo_escape_and_wrap(row['Sentence'])
            rough_pos = row['Rough_POS']
            detailed_pos = row['Detailed_POS']
            lemmatized = undo_escape_and_wrap(row['Lemmatized_Sentence'])
            
            try:
                # Generate n-grams
                ngram_data, start_id = generate_ngrams(sentence, rough_pos, detailed_pos, lemmatized, start_id=start_id)
                
                for ngram_size, ngrams_list in ngram_data.items():
                    for ngram_tuple in ngrams_list:
                        ngram_id, ngram_size, rough_pos_str, detailed_pos_str, ngram_str, lemma_str = ngram_tuple
                        
                        # Redo escape and wrap before saving
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
            except ValueError as e:
                print(f"Skipping line due to error: {e}")
    
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
