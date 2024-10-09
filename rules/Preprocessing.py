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

def preprocess_text(input_file, tokenized_file, output_file, batch_size=700):
    with open(tokenized_file, 'w', encoding='utf-8') as token_file, open(output_file, 'a', encoding='utf-8') as output:
        # Process the input file in batches
        for dataset in load_dataset_in_batches(input_file, batch_size):
            tokenized_sentences = []
            general_pos_tagged_sentences = []
            detailed_pos_tagged_sentences = []
            lemmatized_sentences = []

            # Tokenize sentences and write them to the tokenized file
            for text in dataset:
                sentences = tokenize(text)
                token_file.write("\n".join(sentences) + "\n")
                tokenized_sentences.extend(sentences)
            print(f"Processed {len(tokenized_sentences)} sentences and saved to {tokenized_file}")

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

            # Append batch results to the output file
            for tok_sentence, gen_pos, det_pos, lemma in zip(tokenized_sentences, general_pos_tagged_batch, detailed_pos_tagged_batch, lemmatized_batch):
                if ',' in tok_sentence:
                    tok_sentence = f'"{tok_sentence}"'
                output.write(f"{tok_sentence},{gen_pos},{det_pos},{lemma},\n")

            # Clear the lists after each batch to avoid memory issues
            tokenized_sentences.clear()
            general_pos_tagged_batch.clear()
            detailed_pos_tagged_batch.clear()
            lemmatized_batch.clear()

    print(f"Preprocessed data saved to {output_file}")

def run_preprocessing():
    # Define your file paths here
    input_txt = "dataset/ALT-Parallel-Corpus-20191206/data_fil.txt"  # Input file (the .txt file)
    tokenized_txt = "rules\database\tokenized_sentences.txt"                        # File to save tokenized sentences
    output_csv = "rules\database\preprocessed.csv"                                  # File to save the preprocessed output

    # Start the preprocessing
    preprocess_text(input_txt, tokenized_txt, output_csv)

# Automatically run when the script is executed
if __name__ == "__main__":
    run_preprocessing()
