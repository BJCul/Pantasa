import pandas as pd
import string

# Load the CSV file
file_path = 'data/processed/balarila_d2_test.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Function to clean the sentence by removing punctuation and numbers
def clean_sentence(sentence):
    if not isinstance(sentence, str):
        return ""
    translator = str.maketrans('', '', string.punctuation + string.digits)
    return sentence.translate(translator)

# Clean all columns in the DataFrame
cleaned_df = df.applymap(lambda x: clean_sentence(x) if isinstance(x, str) else x)

# Filter rows where 'Correct or Incorrect' is 'Incorrect' and sentence has less than 10 words
filtered_df = cleaned_df[cleaned_df['Correct or Incorrect'] == 'Incorrect']
filtered_df = filtered_df[
    filtered_df['Original Sentence'].apply(lambda x: 10 < len(str(x).split()) < 15)
]
# Create a new DataFrame with required columns
result_df = filtered_df[['Original Sentence', 'Gold Sentence']]

# Save the result to a new CSV file
output_file = 'data/processed/incorrect_balarila_d2.csv'
result_df.to_csv(output_file, index=False)

print(f"Filtered and cleaned data saved to {output_file}.")
