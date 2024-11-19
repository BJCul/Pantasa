import pandas as pd

# Replace these file paths with your actual file locations
file_path_1 = "rules/database/POS/7grams.csv"  # Path to the first file (original n-grams file)
file_path_2 = "rules/database/Generalized/POSTComparison/7grams.csv"  # Path to the second file (generalized n-grams file)

# Load the files
ngrams_df = pd.read_csv(file_path_1)
generalized_ngrams_df = pd.read_csv(file_path_2)

# Identify Pattern_IDs present in the first file
valid_pattern_ids = set(ngrams_df['Pattern_ID'])

# Filter rows in the second file based on the presence of Pattern_ID in the first file
filtered_generalized_ngrams_df = generalized_ngrams_df[generalized_ngrams_df['Pattern_ID'].isin(valid_pattern_ids)]

# Overwrite the second file with the filtered data
filtered_generalized_ngrams_df.to_csv(file_path_2, index=False)

print(f"File updated successfully: {file_path_2}")
