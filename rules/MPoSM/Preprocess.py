import sys
import os
import logging
import csv

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules.Modules.preprocessing.Tokenizer import tokenize
from rules.Modules.preprocessing.POSDTagger import pos_tag as pos_dtag
from rules.Modules.preprocessing.POSRTagger import pos_tag as pos_rtag

def load_dataset(file_path, batch_size=5000, max_lines=None):
    """
    Load the dataset in batches of sentences.
    The function will ignore the identifier (first part) and return only the sentences (second part).
    Yields a batch of sentences from the dataset.
    """
    logging.info(f"Loading dataset from {file_path} in batches of {batch_size}")
    with open(file_path, 'r', encoding='utf-8') as file:
        batch = []
        line_count = 0
        for line in file:
            if max_lines is not None and line_count >= max_lines:
                break  # Stop loading if max_lines is reached
            line = line.strip()
            if line:
                # Split by tab and keep only the sentence part (ignore the ID part)
                parts = line.split('\t')
                if len(parts) > 1:  # Ensure that there are two parts (ID and sentence)
                    sentence = parts[1]
                    batch.append(sentence)
                    line_count += 1

            # If the batch size is reached, yield the batch
            if len(batch) == batch_size:
                logging.info(f"Loaded a batch of {len(batch)} sentences.")
                yield batch
                batch = []  # Clear the batch after yielding

        # Yield any remaining sentences in the last batch
        if batch:
            logging.info(f"Loaded the last batch of {len(batch)} sentences.")
            yield batch

def preprocess_text_in_batches(input_file, pos_output_file, tokenized_output_file, batch_size=500, log_every=50, max_lines=None):
    """
    Process the input dataset in batches by tokenizing and performing POS tagging,
    writing the POS tags to a CSV file and tokenized sentences to a separate text file.
    """
    total_tagged_sentences = 0
    log_counter = 0  # To track logging after every 50 sentences
    
    # Open the CSV file for POS tags and the text file for tokenized sentences
    with open(pos_output_file, 'a', encoding='utf-8', newline='') as csvfile, open(tokenized_output_file, 'a', encoding='utf-8') as txtfile:
        writer = csv.writer(csvfile)
        writer.writerow(['General POS', 'Detailed POS'])  # CSV header

        # Load the dataset in batches
        for batch in load_dataset(input_file, batch_size, max_lines):
            tokenized_sentences = []
            sentence_identifiers = []

            # Process each sentence in the current batch
            for sentence in batch:
                sentences_batch = tokenize(sentence)
                tokenized_sentences.extend(sentences_batch)

                # Write each tokenized sentence to the text file (one sentence per line)
                txtfile.write("\n".join(sentences_batch) + "\n")

            # Process the batch once loaded and write POS tags to CSV
            process_batch(tokenized_sentences, writer)

            # Update total tagged sentences and log after every 50 sentences
            total_tagged_sentences += len(tokenized_sentences)
            while total_tagged_sentences >= log_counter + log_every:
                log_counter += log_every
                logging.info(f"Tagged {log_counter} sentences so far.")

        # Final log for any remaining sentences
        if total_tagged_sentences % log_every != 0:
            logging.info(f"Final batch: tagged {total_tagged_sentences} sentences total.")

    logging.info(f"Preprocessed data saved to {pos_output_file}")
    logging.info(f"Total sentences tagged and saved: {total_tagged_sentences}")

def process_batch(tokenized_sentences, csv_writer):
    """
    Helper function to process a batch of tokenized sentences and write only the POS tags to the CSV file.
    """
    general_pos_tagged_batch = []
    detailed_pos_tagged_batch = []
    
    # Perform POS tagging for each sentence
    for sentence in tokenized_sentences:
        if sentence:
            general_pos_tagged_batch.append(pos_rtag(sentence))  # Rough POS tagging
            detailed_pos_tagged_batch.append(pos_dtag(sentence))  # Detailed POS tagging
        else:
            general_pos_tagged_batch.append('')
            detailed_pos_tagged_batch.append('')

    # Write only the POS tags to the CSV file (no original sentence or identifier)
    for gen_pos, det_pos in zip(general_pos_tagged_batch, detailed_pos_tagged_batch):
        csv_writer.writerow([gen_pos, det_pos])

    logging.info(f"Processed and saved {len(tokenized_sentences)} sentences' POS tags in the batch.")

    # Clear lists after processing to save memory
    general_pos_tagged_batch.clear()
    detailed_pos_tagged_batch.clear()

# Main function to execute the preprocessing
def main():
    input_file = 'dataset/ALT-Parallel-Corpus-20191206/data_fil.txt'   # Input file with raw text
    pos_output_file = 'MPoSM/pos_tags_output.csv'  # CSV file for POS tags
    tokenized_output_file = 'MPoSM/tokenized_sentences.txt'  # Text file for tokenized sentences

    # Specify the max_lines argument if you want to limit how many lines are processed
    max_lines = 5000  # Set this to the number of lines you want to process (or None for no limit)

    preprocess_text_in_batches(input_file, pos_output_file, tokenized_output_file, max_lines=max_lines)  # Apply preprocessing and POS tagging

if __name__ == "__main__":
    main()
