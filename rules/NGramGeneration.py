import csv
from collections import defaultdict
import re

# Define the regular expression for tokenizing word sequences including punctuation
regex = r"[^.!?,;:—\s][^.!?,;:—]*[.!?,;:—]?['\"]?(?=\s|$)"

def custom_ngrams(sequence, n):
    """Generate n-grams from a sequence."""
    return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]


def generate_ngrams(word_sequence, rough_pos_sequence, detailed_pos_sequence, lemma_sequence, ngram_range=(2, 7), add_newline=False, start_id=0):
    ngram_sequences = defaultdict(list)
    
    # Use the regex to find words including punctuation
    words = word_sequence.split()
    rough_pos_tags = rough_pos_sequence.split()
    detailed_pos_tags = detailed_pos_sequence.split()
    lemmas = lemma_sequence.split()
    
    # Debugging: print lengths of sequences
    print(f"Words length: {len(words)}")
    print(f"Rough POS length: {len(rough_pos_tags)}")
    print(f"Detailed POS length: {len(detailed_pos_tags)}")
    print(f"Lemmas length: {len(lemmas)} \n")
    
    # Ensure the lengths match
    if len(rough_pos_tags) != len(detailed_pos_tags) or len(words) != len(lemmas):
        raise ValueError("Sequences lengths do not match")

    current_id = start_id
    
    for n in range(ngram_range[0], ngram_range[1] + 1):
        word_n_grams = custom_ngrams(words, n)
        rough_pos_n_grams = custom_ngrams(rough_pos_tags, n)
        detailed_pos_n_grams = custom_ngrams(detailed_pos_tags, n)
        lemma_n_grams = custom_ngrams(lemmas, n)
        
        for word_gram, rough_pos_gram, detailed_pos_gram, lemma_gram in zip(word_n_grams, rough_pos_n_grams, detailed_pos_n_grams, lemma_n_grams):
            # Split combined detailed POS tags and add to unique detailed tags set
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
        
        for row in reader:
            sentence = row['Sentences']
            rough_pos = row['General_POS_Tagged']
            detailed_pos = row['Detailed_POS_Tagged']
            lemmatized = row['Lemmatized']
            
            try:
                ngram_data, start_id = generate_ngrams(sentence, rough_pos, detailed_pos, lemmatized, start_id=start_id)
                
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
    
    # Write results to output CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
        fieldnames = ['N-Gram_ID', 'N-Gram_Size', 'RoughPOS_N-Gram', 'DetailedPOS_N-Gram', 'N-Gram', 'Lemma_N-Gram']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

# Example usage
input_csv = f'database/cleaned_preprocessed.csv'
output_csv = f'database/ngrams.csv'
process_csv(input_csv, output_csv)
