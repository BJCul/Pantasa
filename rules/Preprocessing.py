import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Import tqdm for progress tracking
from Modules.Tokenizer import tokenize
from Modules.POSDTagger import pos_tag as pos_dtag
from Modules.POSRTagger import pos_tag as pos_rtag
from Modules.Lemmatizer import lemmatize_sentence

import sys

# Add the path to morphinas_project
sys.path.append('C:/Users/Carlo Agas/Documents/GitHub/Pantasaa')

from morphinas_project.lemmatizer_client import initialize_stemmer

# Initialize the Morphinas lemmatizer once to reuse across function calls
gateway, lemmatizer = initialize_stemmer()

# Set the JVM options to increase the heap size
os.environ['JVM_OPTS'] = '-Xmx2g'

def load_dataset(file_path):
    """Load dataset from a text file, assuming each line contains a single sentence."""
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentence = line.strip()
            if sentence:
                dataset.append(sentence)
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

def load_processed_sentences(output_file):
    """Load already processed sentences from the output file."""
    processed_sentences = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.split(',')  # Assuming the sentence is the first element in each line
                if parts:
                    processed_sentences.add(parts[0].strip('"'))
    return processed_sentences

def process_batch(batch):
    """Process a batch of sentences for general POS tagging, detailed POS tagging, and lemmatization."""
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

    return general_pos_tagged_batch, detailed_pos_tagged_batch, lemmatized_batch

def preprocess_text(input_file, tokenized_file, output_file, batch_size=700):
    # Dynamically get the maximum number of CPU cores available
    max_workers = os.cpu_count()

    dataset = load_dataset(input_file)
    tokenized_sentences = load_tokenized_sentences(tokenized_file)
    processed_sentences = load_processed_sentences(output_file)

    new_tokenized_sentences = []

    with open(tokenized_file, 'a', encoding='utf-8') as token_file:
        for sentence in dataset:
            if sentence not in tokenized_sentences and sentence not in processed_sentences:
                new_tokenized_sentences.append(sentence)
                tokenized_sentences.add(sentence)
                token_file.write(sentence + "\n")
        print(f"Sentences tokenized to {tokenized_file}")

    with open(output_file, 'a', encoding='utf-8') as output:
        for i in range(0, len(new_tokenized_sentences), batch_size):
            batch = new_tokenized_sentences[i:i + batch_size]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Parallel processing for POS tagging and lemmatization
                results = list(tqdm(executor.map(process_batch, [batch]), total=1, desc="Processing Batch"))
                for general_pos, detailed_pos, lemma in results:
                    # Append batch results to output file immediately
                    for tok_sentence, gen_pos, det_pos, lemma in zip(batch, general_pos, detailed_pos, lemma):
                        # Add single quotes around sentences ending with a comma
                        if ',' in tok_sentence:
                            tok_sentence = f'"{tok_sentence}"'
                        output.write(f"{tok_sentence},{gen_pos},{det_pos},{lemma},\n")

    print(f"Preprocessed data saved to {output_file}")

def run_preprocessing():
    # Define your file paths here
    input_txt = "database/ngram_feed.txt"           # Input file (the .txt file)
    tokenized_txt = "database/tokenized_sentences.txt"  # File to save tokenized sentences
    output_csv = "database/preprocessed.csv"     # File to save the preprocessed output

    # Start the preprocessing
    preprocess_text(input_txt, tokenized_txt, output_csv)

# Automatically run when the script is executed
if _name_ == "_main_":
    run_preprocessing()