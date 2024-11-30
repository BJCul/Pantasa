import pandas as pd

# Load the dataset
file_path = 'rules/database/Generalized/POSTComparison/7grams.csv'  # Replace with your actual file name

# Read the CSV file
df = pd.read_csv(file_path)

# Sort by 'Pattern_ID'
df_sorted = df.sort_values(by='Pattern_ID', ascending=True)

# Remove duplicates based on the 'POS_N-Gram' column
df_deduplicated = df_sorted.drop_duplicates(subset='POS_N-Gram', keep='first')

# Save the processed file
output_file_path = 'sorted_deduplicated_7grams.csv'
df_deduplicated.to_csv(output_file_path, index=False)

print(f"Processed file saved as {output_file_path}")