import os
from itertools import islice
from Modules.Tokenizer import tokenize
from Modules.POSDTagger import pos_tag as pos_dtag
from Modules.POSRTagger import pos_tag as pos_rtag
from Modules.Lemmatizer import lemmatize_sentence 

# Set the JVM options to increase the heap size
os.environ['JVM_OPTS'] = '-Xmx2g'

def load_dataset_in_batches(file_path, batch_size, skip_lines=2500):
    """Load the dataset in batches of lines from a text file, skipping the first 'skip_lines' lines."""
    with open(file_path, 'r', encoding='utf-8') as file:
        # Skip the first 2500 lines
        for _ in range(skip_lines):
            next(file, None)
        
        # Load the remaining lines in batches
        while True:
            lines = list(islice(file, batch_size))
            if not lines:
                break
            yield [line.strip().split('\t')[1] for line in lines if len(line.strip().split('\t')) > 1]

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
            sentences = tokenize(text)
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


            # POS tagging and lemmatization
            general_pos_tagged_batch = []
            detailed_pos_tagged_batch = []
            lemmatized_batch = []

            for sentence in tokenized_sentences:
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

    input_txt = "dataset.txt"           # Input file (the .txt file)
    tokenized_txt = "tokenized_sentences.txt"  # File to save tokenized sentences
    output_csv = "preprocessed.csv"     # File to save the preprocessed output
    target_lines = 1000  # Set the target number of lines to process


    # Start the preprocessing
    preprocess_text(input_txt, tokenized_txt, output_csv, target_lines)

# Automatically run when the script is executed
if __name__ == "__main__":
    run_preprocessing()
