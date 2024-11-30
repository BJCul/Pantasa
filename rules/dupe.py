import pandas as pd

# File paths
file_path = 'rules/database/Generalized/POSTComparison/6grams.csv'
freq_file = 'rules/database/POS/6grams.csv'

# Load the datasets
df_patterns = pd.read_csv(file_path)
df_frequencies = pd.read_csv(freq_file)

# Ensure the 'Pattern_ID' and 'Frequency' columns exist in both datasets
if 'Pattern_ID' not in df_patterns.columns or 'Pattern_ID' not in df_frequencies.columns:
    raise ValueError("'Pattern_ID' column not found in one or both files.")

# Filter rows where 'Frequency' is >= 2 in freq_file
valid_ids = df_frequencies[df_frequencies['Frequency'] >= 2]['Pattern_ID']

# Also remove entries in file_path that don't exist in freq_file
common_ids = set(df_patterns['Pattern_ID']).intersection(set(df_frequencies['Pattern_ID']))

# Combine filtering conditions: valid frequency and existing in freq_file
valid_and_common_ids = valid_ids[valid_ids.isin(common_ids)]

# Filter the patterns and frequency DataFrames to only include valid Pattern_IDs
df_patterns_filtered = df_patterns[df_patterns['Pattern_ID'].isin(valid_and_common_ids)]
df_freq_filtered = df_frequencies[df_frequencies['Pattern_ID'].isin(valid_and_common_ids)]

# Sort the filtered patterns DataFrame by 'Pattern_ID'
df_patterns_sorted = df_patterns_filtered.sort_values(by='Pattern_ID', ascending=True)

# Save the filtered files back to CSV
filtered_patterns_file_path = 'filtered_patterns_5grams.csv'
filtered_freq_file_path = 'filtered_frequencies_5grams.csv'

df_patterns_sorted.to_csv(filtered_patterns_file_path, index=False)
df_freq_filtered.to_csv(filtered_freq_file_path, index=False)

print(f"Filtered patterns saved as {filtered_patterns_file_path}")
print(f"Filtered frequencies saved as {filtered_freq_file_path}")
