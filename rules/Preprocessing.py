import re
import os
from Modules.Tokenizer import tokenize
from Modules.POSDTagger import pos_tag as pos_dtag
from Modules.POSRTagger import pos_tag as pos_rtag
from Modules.Lemmatizer import lemmatize_sentence 

import sys
from morphinas_project.lemmatizer_client import initialize_stemmer

# Add the path to morphinas_project
sys.path.append(r'C:\Projects\Pantasa\app')

# Initialize the Morphinas lemmatizer once to reuse across function calls
gateway, lemmatizer = initialize_stemmer()

# Set the JVM options to increase the heap size
os.environ['JVM_OPTS'] = '-Xmx2g'

def load_dataset(file_path):
    """Load dataset from a text file instead of a CSV."""
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Each line is expected to have two parts: an ID and a sentence, separated by a tab
            parts = line.strip().split('\t')
            if len(parts) > 1:
                dataset.append(parts[1])  # The second part is the sentence
    return dataset

def save_text_file(text_data, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        for line in text_data:
            f.write(line + "\n")

def load_tokenized_sentences(tokenized_file):
    """Load already tokenized sentences from the tokenized file."""
    tokenized_sentences = set()
    if os.path.exists(tokenized_file):
        with open(tokenized_file, 'r', encoding='utf-8') as file:
            for line in file:
                tokenized_sentences.add(line.strip())
    return tokenized_sentences

def split_sentences(text):
    """
    Split a long text into smaller sentences based on punctuation and rules.
    """
    # Define a regex to split on sentence-ending punctuation followed by space or an uppercase letter
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Ignore very short fragments or single-word "sentences"
        if len(sentence.split()) > 2:  
            cleaned_sentences.append(sentence)
    return cleaned_sentences

def preprocess_text(input_file, tokenized_file, output_file, target_lines, batch_size=700):
    dataset = load_dataset(input_file)
    tokenized_sentences = load_tokenized_sentences(tokenized_file)
    
    new_tokenized_sentences = []
    general_pos_tagged_sentences = []
    detailed_pos_tagged_sentences = []
    lemmatized_sentences = []
    lines_written = 0  # Track how many lines are written to the output

    with open(tokenized_file, 'a', encoding='utf-8') as token_file:
        for text in dataset:
            # Split text into usable sentences
            raw_sentences = tokenize(text)
            for raw_sentence in raw_sentences:
                # Further split raw sentences into smaller sentences
                sentences = split_sentences(raw_sentence)
                for sentence in sentences:
                    if sentence not in tokenized_sentences:
                        new_tokenized_sentences.append(sentence)
                        tokenized_sentences.add(sentence)
                        token_file.write(sentence + "\n")
        print(f"Sentences tokenized to {tokenized_file}")
    
    with open(output_file, 'a', encoding='utf-8') as output:
        for i in range(0, len(new_tokenized_sentences), batch_size):
            if lines_written >= target_lines:
                print(f"Target of {target_lines} lines reached. Halting processing.")
                break  # Stop processing when target lines are met
            
            batch = new_tokenized_sentences[i:i + batch_size]

            general_pos_tagged_batch = []
            detailed_pos_tagged_batch = []
            lemmatized_batch = []

            for sentence in batch:
                if sentence:
                    general_pos_tagged_batch.append(pos_rtag(sentence))
                    detailed_pos_tagged_batch.append(pos_dtag(sentence))
                    lemmatized_batch.append(lemmatize_sentence(sentence))
                else:
                    general_pos_tagged_batch.append('')
                    detailed_pos_tagged_batch.append('')
                    lemmatized_batch.append('')

            # Append batch results to output file immediately
            for tok_sentence, gen_pos, det_pos, lemma in zip(batch, general_pos_tagged_batch, detailed_pos_tagged_batch, lemmatized_batch):
                if lines_written >= target_lines:
                    break  # Stop processing when target lines are met
                # Add single quotes around sentences ending with a comma
                if ',' in tok_sentence:
                    tok_sentence = f'"{tok_sentence}"'
                output.write(f"{tok_sentence},{gen_pos},{det_pos},{lemma},\n")
                lines_written += 1

            # Clear lists after each batch to avoid memory issues
            general_pos_tagged_sentences.clear()
            detailed_pos_tagged_sentences.clear()
            lemmatized_sentences.clear()

    print(f"Preprocessed data saved to {output_file}")

def run_preprocessing():
    # Define your file paths here
    input_txt = "dataset/ALT-Parallel-Corpus-20191206/data_fil.txt"           # Input file (the .txt file)
    tokenized_txt = "database/tokenized_sentences.txt"  # File to save tokenized sentences
    output_csv = "database/preprocessed.csv"     # File to save the preprocessed output
    target_lines = 30000  # Set the target number of lines to process

    # Start the preprocessing
    preprocess_text(input_txt, tokenized_txt, output_csv, target_lines)

# Automatically run when the script is executed
if __name__ == "__main__":
    run_preprocessing()
