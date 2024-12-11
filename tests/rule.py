import pandas as pd

# Load the CSV into a DataFrame
file_path = 'data/processed/detailed_hngram.csv'
df = pd.read_csv(file_path)

# Function to filter out rows where any token starts with 'PM'
def remove_pm_tokens(hybrid_ngram):
    # Remove underscores and split the Hybrid_N-Gram into individual tokens
    tokens = hybrid_ngram.replace("_", " ").split()
    # Check if any token starts with 'PM'
    return all(not token.startswith("PM") for token in tokens)

# Apply the function to filter the DataFrame
filtered_df = df[df['Hybrid_N-Gram'].apply(remove_pm_tokens)]

# Save the filtered DataFrame to a new CSV
output_path = 'data/processed/detailed_hngram_filtered.csv'
filtered_df.to_csv(output_path, index=False)

print(f"Filtered file saved to {output_path}")
