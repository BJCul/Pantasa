import pandas as pd
from langdetect import detect

# Load the dictionary file (assuming it's a CSV file)
df = pd.read_csv('filtered_taglish_dictionary.csv', header=None, names=['Word'])

# Function to check if the word is English
def is_english(word):
    try:
        return detect(word) == 'en'
    except:
        return False  # If detection fails, treat as not English

# Apply the function to filter out English words
df['Is_English'] = df['Word'].apply(is_english)
tagalog_only_df = df[df['Is_English'] == False]  # Keep only non-English words

# Drop the 'Is_English' column
tagalog_only_df = tagalog_only_df.drop(columns=['Is_English'])

# Save the filtered dictionary
tagalog_only_df.to_csv('filtered_dictionary.csv', index=False, header=False)

print("English words have been removed, and the filtered dictionary has been saved.")