import os
import pandas as pd
import csv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def get_latest_id(output_file):
    """Get the latest ID from the output file to continue ID generation."""
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            if not df.empty:
                # Get the maximum N-Gram_ID from the file
                latest_id = df['N-Gram_ID'].max()
                return latest_id + 1  # Start with the next ID
        except Exception as e:
            print(f"Error reading {output_file}: {e}")
    return 0  # Start from 0 if file does not exist or is empty

def process_row(row, start_id):
    """Process a single row and return the n-gram details and updated ID."""
    # Example processing logic, replace with actual row processing
    n_grams = [
        {'N-Gram_ID': start_id + i, 
         'N-Gram_Size': len(row['N-Gram'].split()), 
         'RoughPOS_N-Gram': row['RoughPOS'], 
         'DetailedPOS_N-Gram': row['DetailedPOS'], 
         'N-Gram': row['N-Gram'], 
         'Lemma_N-Gram': row['Lemma']}
        for i in range(1)  # Assuming each row results in a single n-gram; adjust as needed
    ]
    return n_grams, start_id + len(n_grams)

def process_csv(input_file, output_file, start_row=0):
    results = []
    # Get the starting ID from the output file to avoid resetting
    start_id = get_latest_id(output_file)

    # Dynamically get the maximum number of CPU cores available
    max_workers = os.cpu_count()

    # Load CSV data into a DataFrame
    df = pd.read_csv(input_file)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Wrap the executor with tqdm to show progress
        with tqdm(total=len(df) - start_row, desc="Processing Rows") as pbar:
            futures = []

            # Iterate from the starting row
            for i, row in df.iterrows():
                if i < start_row:
                    continue  # Skip rows before the start_row index

                # Submit tasks to the executor for rows starting from the specified index
                futures.append(executor.submit(process_row, row, start_id))

                # Retrieve the results and update the start_id for each row
                for future in futures:
                    result, start_id = future.result()
                    results.extend(result)
                    pbar.update(1)

    # Write results to output CSV
    with open(output_file, 'a', newline='', encoding='utf-8') as out_file:  # Appending to output file
        fieldnames = ['N-Gram_ID', 'N-Gram_Size', 'RoughPOS_N-Gram', 'DetailedPOS_N-Gram', 'N-Gram', 'Lemma_N-Gram']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        if os.stat(output_file).st_size == 0:  # If the file is empty, write the header
            writer.writeheader()
        writer.writerows(results)

    print(f"Processed data saved to {output_file}")

# Example usage
input_csv = 'rules/database/preprocessed.csv'  # Replace with your input CSV path
output_csv = 'rules/database/ngram.csv'  # Replace with your desired output CSV path
process_csv(input_csv, output_csv, start_row=0)  # Start processing from row 0
