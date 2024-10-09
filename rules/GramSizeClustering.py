import csv
from collections import defaultdict

def cluster_ngrams_by_size(input_file):
    clustered_ngrams = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        
        for row in reader:
            ngram_size = int(row['N-Gram_Size'])
            ngram_id = row['N-Gram_ID']
            rough_pos_ngram = row['RoughPOS_N-Gram']
            detailed_pos_ngram = row['DetailedPOS_N-Gram']
            ngram = row['N-Gram']
            lemma_ngram = row['Lemma_N-Gram']
            
            clustered_ngrams[ngram_size].append({
                'N-Gram_ID': ngram_id,
                'RoughPOS_N-Gram': rough_pos_ngram,
                'DetailedPOS_N-Gram': detailed_pos_ngram,
                'N-Gram': ngram,
                'Lemma_N-Gram': lemma_ngram
            })
    
    # Write results to separate CSV files for each n-gram size
    for ngram_size, ngrams_list in clustered_ngrams.items():
        output_file = f'database/GramSize/{ngram_size}grams.csv'
        with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
            fieldnames = ['N-Gram_ID', 'RoughPOS_N-Gram', 'DetailedPOS_N-Gram', 'N-Gram', 'Lemma_N-Gram']
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ngrams_list)

# Example usage
input_csv = 'database/ngrams.csv'
cluster_ngrams_by_size(input_csv)