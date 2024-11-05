import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# List of input CSV files
input_files = [
    'rules/database/POS/2grams.csv',
    'rules/database/POS/3grams.csv',
    'rules/database/POS/4grams.csv',
    'rules/database/POS/5grams.csv',
    'rules/database/POS/6grams.csv',
    'rules/database/POS/7grams.csv'
]

# Output file names
rough_ngrams_output_file = 'rules/database/rough_ngrams_merged.csv'
detailed_ngrams_output_file = 'rules/database/detailed_ngrams_merged.csv'

# Function to process a single file and extract rough and detailed n-grams
def process_file(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Separate rough and detailed n-grams
        rough_ngrams = df[['Pattern_ID', 'RoughPOS_N-Gram', 'Frequency', 'ID_Array']]
        detailed_ngrams = df[['Pattern_ID', 'DetailedPOS_N-Gram', 'Frequency', 'ID_Array']]
        
        return rough_ngrams, detailed_ngrams
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Use ThreadPoolExecutor for parallel processing
rough_ngrams_list = []
detailed_ngrams_list = []

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_file, file): file for file in input_files}

    for future in futures:
        try:
            rough_ngrams, detailed_ngrams = future.result()
            if not rough_ngrams.empty:
                rough_ngrams_list.append(rough_ngrams)
            if not detailed_ngrams.empty:
                detailed_ngrams_list.append(detailed_ngrams)
        except Exception as e:
            print(f"Error processing results: {e}")

# Concatenate all rough and detailed n-grams
all_rough_ngrams = pd.concat(rough_ngrams_list, ignore_index=True)
all_detailed_ngrams = pd.concat(detailed_ngrams_list, ignore_index=True)

# Drop rows with empty n-grams if necessary (optional)
all_rough_ngrams.dropna(subset=['RoughPOS_N-Gram'], inplace=True)
all_detailed_ngrams.dropna(subset=['DetailedPOS_N-Gram'], inplace=True)

# Save the combined DataFrames to CSV files
all_rough_ngrams.to_csv(rough_ngrams_output_file, index=False)
all_detailed_ngrams.to_csv(detailed_ngrams_output_file, index=False)

print(f"Rough n-grams merged data saved to {rough_ngrams_output_file}")
print(f"Detailed n-grams merged data saved to {detailed_ngrams_output_file}")
