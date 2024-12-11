import pandas as pd

# Paths to input and output CSV files
source_csv = 'data/processed/gold.csv'  # Replace with the path to your first CSV file
target_csv = 'data/processed/errordt3_output_result.csv'   # Replace with the path to your second CSV file
output_csv = 'data/processed/errordt3_output_result_x.csv'  # File to save the combined result

# Load both CSVs
source_df = pd.read_csv(source_csv)
target_df = pd.read_csv(target_csv)

# Normalize whitespace in both DataFrames to ensure proper merging
source_df['Original Sentence'] = source_df['Original Sentence'].str.strip()
target_df['Original Sentence'] = target_df['Original Sentence'].str.strip()

# Merge based on 'Original Sentence'
merged_df = target_df.merge(
    source_df[['Original Sentence', 'Gold Sentence']],  # Select columns to merge
    on='Original Sentence', 
    how='left'
)

# Reorder columns to place 'Gold Sentence' at the desired position
columns_order = ['Original Sentence', 'Gold Sentence'] + [
    col for col in merged_df.columns if col not in ['Original Sentence', 'Gold Sentence']
]
merged_df = merged_df[columns_order]

# Save the result
merged_df.to_csv(output_csv, index=False)

print(f"Gold Sentence has been appended and saved to {output_csv}.")
