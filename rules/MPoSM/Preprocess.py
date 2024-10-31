import sys
import os
import logging
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules.Modules.preprocessing.Tokenizer import tokenize
from rules.Modules.preprocessing.POSDTagger import pos_tag as pos_dtag
from rules.Modules.preprocessing.POSRTagger import pos_tag as pos_rtag

def load_dataset(file_path, batch_size=5000, max_lines=None):
    logging.info(f"Loading dataset from {file_path} in batches of {batch_size}")
    with open(file_path, 'r', encoding='utf-8') as file:
        batch = []
        line_count = 0
        for line in tqdm(file, desc="Loading dataset", total=max_lines):
            if max_lines is not None and line_count >= max_lines:
                break
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) > 1:
                    sentence = parts[1]
                    batch.append(sentence)
                    line_count += 1

            if len(batch) == batch_size:
                logging.info(f"Loaded a batch of {len(batch)} sentences.")
                yield batch
                batch = []

        if batch:
            logging.info(f"Loaded the last batch of {len(batch)} sentences.")
            yield batch

def parallel_pos_tagging(sentences):
    """
    Apply POS tagging in parallel using ProcessPoolExecutor.
    Returns tuples of (general_pos_tags, detailed_pos_tags).
    """
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(pos_rtag, sentence): sentence for sentence in sentences}
        general_pos_tags = []
        detailed_pos_tags = []
        
        for future in as_completed(futures):
            sentence = futures[future]
            try:
                general_pos_tags.append(future.result())  # Rough POS tagging result
                detailed_pos_tags.append(pos_dtag(sentence))  # Detailed POS tagging result
            except Exception as e:
                logging.error(f"Error tagging sentence: {sentence}. Error: {e}")
                general_pos_tags.append('')
                detailed_pos_tags.append('')
        
        return general_pos_tags, detailed_pos_tags

def preprocess_text_in_batches(input_file, pos_output_file, tokenized_output_file, batch_size=500, log_every=50, max_lines=None):
    total_tagged_sentences = 0
    log_counter = 0

    with open(pos_output_file, 'a', encoding='utf-8', newline='') as csvfile, open(tokenized_output_file, 'a', encoding='utf-8') as txtfile:
        writer = csv.writer(csvfile)
        writer.writerow(['General POS', 'Detailed POS'])

        for batch in load_dataset(input_file, batch_size, max_lines):
            tokenized_sentences = []
            
            for sentence in batch:
                sentences_batch = tokenize(sentence)
                tokenized_sentences.extend(sentences_batch)
                txtfile.write("\n".join(sentences_batch) + "\n")

            general_pos, detailed_pos = parallel_pos_tagging(tokenized_sentences)

            for gen_pos, det_pos in zip(general_pos, detailed_pos):
                writer.writerow([gen_pos, det_pos])

            total_tagged_sentences += len(tokenized_sentences)
            while total_tagged_sentences >= log_counter + log_every:
                log_counter += log_every
                logging.info(f"Tagged {log_counter} sentences so far.")

        if total_tagged_sentences % log_every != 0:
            logging.info(f"Final batch: tagged {total_tagged_sentences} sentences total.")

    logging.info(f"Preprocessed data saved to {pos_output_file}")
    logging.info(f"Total sentences tagged and saved: {total_tagged_sentences}")

# Main function to execute the preprocessing
def main():
    input_file = 'rules/dataset/ALT-Parallel-Corpus-20191206/data_fil.txt'
    pos_output_file = 'rules/MPoSM/pos_tags_output.csv'
    tokenized_output_file = 'rules/MPoSM/tokenized_sentences.txt'

    max_lines = 5000  # Adjust to None for processing the entire dataset

    preprocess_text_in_batches(input_file, pos_output_file, tokenized_output_file, max_lines=max_lines)

if __name__ == "__main__":
    main()
