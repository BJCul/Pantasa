import sys
import os
import logging
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add the project root directory to sys.path to locate modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rules.Modules.Tokenizer import tokenize
from rules.Modules.POSRTagger import pos_tag

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
                else:
                    sentence = parts[0]
                batch.append(sentence)
                line_count += 1
            else:
                logging.warning(f"Skipping empty line at index {line_count}")

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
        futures = {executor.submit(pos_tag, sentence): sentence for sentence in sentences}
        general_pos_tags = []
        detailed_pos_tags = []
        
        for future in as_completed(futures):
            sentence = futures[future]
            try:
                rough_tags, detailed_tags = future.result()  # Rough and detailed POS tagging result
                general_pos_tags.append(rough_tags)
                detailed_pos_tags.append(detailed_tags)
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
                txtfile.flush()  # Ensure data is written to file

            general_pos, detailed_pos = parallel_pos_tagging(tokenized_sentences)
            logging.info(f"General POS: {general_pos}, Detailed POS: {detailed_pos}")  # Debugging line

            for gen_pos, det_pos in zip(general_pos, detailed_pos):
                writer.writerow([gen_pos, det_pos])
            csvfile.flush()  # Ensure data is written to file

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
    input_file = 'data/raw/casual_tagalog/carlo_personal_essays.txt'
    pos_output_file = 'rules/MPoSM/pos_tags_output.csv'
    tokenized_output_file = 'rules/MPoSM/tokenized_sentences.txt'

    max_lines = None # Adjust to None for processing the entire dataset

    preprocess_text_in_batches(input_file, pos_output_file, tokenized_output_file, max_lines=max_lines)

if __name__ == "__main__":
    main()
