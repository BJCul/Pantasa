import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os

def process_ngram_row(row):
    """Process a single row to extract and return n-gram details based on size."""
    try:
        ngram_size = int(row['N-Gram_Size'])
        ngram_id = row['N-Gram_ID']
        rough_pos_ngram = row['RoughPOS_N-Gram']
        detailed_pos_ngram = row['DetailedPOS_N-Gram']
        ngram = row['N-Gram']
        lemma_ngram = row['Lemma_N-Gram']
        
        return ngram_size, {
            'N-Gram_ID': ngram_id,
            'RoughPOS_N-Gram': rough_pos_ngram,
            'DetailedPOS_N-Gram': detailed_pos_ngram,
            'N-Gram': ngram,
            'Lemma_N-Gram': lemma_ngram
        }
    except KeyError as e:
        print(f"Skipping line due to missing column: {e}")
        return None

def cluster_ngrams_by_size(input_file, output_folder='database/GramSize', max_workers=8):
    clustered_ngrams = defaultdict(list)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the CSV file and process each row with ThreadPoolExecutor
    with open(input_file, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(rows), desc="Clustering N-Grams") as pbar:
            futures = [executor.submit(process_ngram_row, row) for row in rows]

            for future in futures:
                result = future.result()
                if result:
                    ngram_size, ngram_details = result
                    clustered_ngrams[ngram_size].append(ngram_details)
                pbar.update(1)

    # Write results to separate CSV files for each n-gram size
    for ngram_size, ngrams_list in clustered_ngrams.items():
        output_file = os.path.join(output_folder, f'{ngram_size}grams.csv')
        with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
            fieldnames = ['N-Gram_ID', 'RoughPOS_N-Gram', 'DetailedPOS_N-Gram', 'N-Gram', 'Lemma_N-Gram']
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ngrams_list)

    print(f"N-grams clustered and saved to {output_folder}")

# Example usage
input_csv = 'database/ngrams.csv'  # Replace with your input CSV path
cluster_ngrams_by_size(input_csv)
