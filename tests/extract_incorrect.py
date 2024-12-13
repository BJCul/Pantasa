import pandas as pd
import string

# Load the CSV file
file_path = 'data/processed/Balanced_Sample__5K_Rows_.csv'  # Replace with your actual file path C:\Projects\Pantasa\data\processed\
df = pd.read_csv(file_path)

# Define punctuation without the hyphen
punctuation_without_hyphen = string.punctuation.replace('-', '')

# Function to clean the sentence by removing punctuation (except hyphen) and numbers
def clean_sentence(sentence):
    if not isinstance(sentence, str):
        return ""
    # Create a translation table that removes punctuation except for '-'
    translator = str.maketrans('', '', punctuation_without_hyphen + string.digits)
    return sentence.translate(translator)

# Clean all columns in the DataFrame
cleaned_df = df.applymap(lambda x: clean_sentence(x) if isinstance(x, str) else x)

# Filter rows where 'Correct or Incorrect' is 'Correct' and sentence has fewer than 10 words
filtered_df = cleaned_df[cleaned_df['Correct or Incorrect'] == 'Incorrect']

# Create a new DataFrame with required columns
result_df = filtered_df[['Original Sentence', 'Gold Sentence']]

# Save the result to a new CSV file
output_file = 'data/processed/incorrect_balarila_t2.csv'
result_df.to_csv(output_file, index=False)

print(f"Filtered and cleaned data saved to {output_file}.")