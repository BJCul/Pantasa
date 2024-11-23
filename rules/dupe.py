import pandas as pd

# Load the dataset
input_file = 'rules/database/Generalized/LexemeComparison/3grams.csv'  # Replace with your actual file name

# Read the CSV file
df = pd.read_csv(input_file)

# Sort by Final_Hybrid_N-Gram and MLM_Scores to prioritize higher scores
df = df.sort_values(by=['Final_Hybrid_N-Gram', 'MLM_Scores'], ascending=[True, False])

# Remove duplicates, keeping only the first occurrence based on Final_Hybrid_N-Gram
cleaned_df = df.drop_duplicates(subset=['Final_Hybrid_N-Gram'], keep='first').reset_index(drop=True)

# Optionally regenerate unique Pattern_IDs
# Extract the n-gram size from existing IDs (assuming it's the first digit)
ngram_size = cleaned_df['POS_N-Gram'].str.count(' ') + 1  # Assumes spaces separate POS tags
base_id = 400001  # Adjust based on your ID format
cleaned_df['Pattern_ID'] = [f"{size}{base_id + idx:05}" for idx, size in enumerate(ngram_size, start=1)]

# Overwrite the original input file with the cleaned data
cleaned_df.to_csv(input_file, index=False)

print(f"Duplicate rows resolved based on Final_Hybrid_N-Gram. The dataset in {input_file} has been updated.")