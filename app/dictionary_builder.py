import datasets
from collections import Counter
import csv
import re
from lingua import Language, LanguageDetectorBuilder

languages = [Language.TAGALOG, Language.ENGLISH]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

# Step 1: Load the NewsPH dataset from Hugging Face with trust_remote_code=True
dataset = datasets.load_dataset("jcblaise/newsph", split='train', trust_remote_code=True)

# Step 2: Define a function to check if a word is Tagalog using the correct ISO code
def is_tagalog_word(word):
    language = detector.detect_language_of(word)
    if language and language.iso_code_639_1.name == 'TL':  # Correct code for Tagalog is 'TL'
        return True
    else:
        return False
    
# Step 3: Define a simple tokenizer to extract words
def tokenize(text):
    text = text.lower()
    # Extract words and filter out those that do not match the Tagalog language detection
    words = re.findall(r'\b\w+\b', text)  # Extract words, ignoring punctuation
    return [word for word in words if is_tagalog_word(word)]  # Filter for Tagalog words

# Step 4: Set target number of lines to process
target_lines = 25000  # You can change this to any target number

# Initialize a counter for word frequencies
word_counter = Counter()

# Initialize counter for processed lines
processed_lines = 0

# Step 5: Process the dataset until the target is met
total_lines = len(dataset)
print(f"Total lines in the dataset: {total_lines}")

# Iterate through the dataset in batches
batch_size = 5000  # Adjust batch size as needed
for start_idx in range(0, total_lines, batch_size):
    if processed_lines >= target_lines:
        print(f"Target of {target_lines} lines reached. Stopping the process.")
        break
    
    end_idx = min(start_idx + batch_size, total_lines)  # End index for the batch
    print(f"Processing lines {start_idx} to {end_idx}...")
    
    # Process each batch
    for i in range(start_idx, end_idx):
        if processed_lines >= target_lines:
            break  # Exit the loop if target lines are processed
        
        example = dataset[i]  # Get the current example
        words = tokenize(example['text'])  # Tokenize the text and filter for Tagalog words
        word_counter.update(words)  # Update the word frequency counter
        
        processed_lines += 1  # Increment the line counter

    # Optionally: Save progress after every batch
    temp_output_file = f'word_list_batch_{start_idx}_{end_idx}.csv'
    with open(temp_output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Sort the words alphabetically and remove duplicates before writing to the CSV
        for word in sorted(word_counter):
            writer.writerow([word])
    print(f"Batch {start_idx}-{end_idx} saved to {temp_output_file}")

# Step 6: Save the final word list to a CSV file, sorted alphabetically and without frequencies
final_output_file = 'final_word_list.csv'
with open(final_output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # Sort the final word_counter alphabetically and remove duplicates before saving
    for word in sorted(word_counter):
        writer.writerow([word])

print(f"Final word list saved to {final_output_file}")
