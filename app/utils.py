# app/utils.py

import csv

NGRAM_SIZE_UPPER = 5  # Maximum n-gram size for rule-based matching
NGRAM_MAX_RULE_SIZE = 7  # Maximum n-gram size for predefined rule-based patterns
NGRAM_SIZE_LOWER = 2  # Minimum n-gram size

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def write_file(file_path, data):
    with open(file_path, 'w') as file:
        file.write(data)

def load_hybrid_ngram_patterns(file_path):
    """
    Loads hybrid n-gram patterns from a CSV file.

    Args:
    - file_path: The path to the CSV file containing hybrid n-grams.

    Returns:
    - A list of dictionaries, where each dictionary contains:
        - 'pattern_id': The ID of the pattern.
        - 'ngram_pattern': The list of POS tags (hybrid n-gram).
    """
    hybrid_ngrams = []

    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            ngram_pattern = row['Final_Hybrid_N-Gram'].split()
            hybrid_ngrams.append({
                'pattern_id': row['Pattern_ID'],
                'ngram_pattern': ngram_pattern
            })

    return hybrid_ngrams

def generate_ngrams(tokens, ngram_size):
    """
    Generates N-grams from a list of tokens with a given N-gram size.
    """
    return [tokens[i:i + ngram_size] for i in range(len(tokens) - ngram_size + 1)]

def process_sentence_with_dynamic_ngrams(tokens):
    """
    Processes the input sentence using dynamically sized N-grams based on constants.
    """
    ngram_collections = {}

    sentence_length = len(tokens)

    # Determine appropriate N-gram sizes based on sentence length
    for ngram_size in range(NGRAM_SIZE_LOWER, min(NGRAM_SIZE_UPPER + 1, sentence_length + 1)):
        ngram_collections[f'{ngram_size}-gram'] = generate_ngrams(tokens, ngram_size)

    # Handling larger N-gram sizes for predefined rules
    if sentence_length >= NGRAM_MAX_RULE_SIZE:
        ngram_collections[f'{NGRAM_MAX_RULE_SIZE}-gram'] = generate_ngrams(tokens, NGRAM_MAX_RULE_SIZE)

    return ngram_collections

# app/utils.py

def extract_ngrams(tokens):
    """
    Generates N-grams from the input tokens using dynamic N-gram sizes.
    The size of N-grams is defined by constants NGRAM_SIZE_LOWER, NGRAM_SIZE_UPPER, and NGRAM_MAX_RULE_SIZE.
    
    Args:
    - tokens: List of POS tags or words from the input sentence.
    
    Returns:
    - List of generated N-grams.
    """
    ngrams = []

    sentence_length = len(tokens)

    # Generate n-grams for sizes between NGRAM_SIZE_LOWER and NGRAM_SIZE_UPPER
    for ngram_size in range(NGRAM_SIZE_LOWER, min(NGRAM_SIZE_UPPER + 1, sentence_length + 1)):
        ngrams.extend(generate_ngrams(tokens, ngram_size))

    # Handle special case for NGRAM_MAX_RULE_SIZE
    if sentence_length >= NGRAM_MAX_RULE_SIZE:
        ngrams.extend(generate_ngrams(tokens, NGRAM_MAX_RULE_SIZE))

    return ngrams

import logging
import os

# Create a directory for logs if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up logging configuration
logging.basicConfig(
    filename='logs/pantasa.log',  # Log to a file
    level=logging.DEBUG,  # Log level, adjust to INFO or ERROR in production
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format in logs
)

# Create a logger for the module
logger = logging.getLogger(__name__)

def log_message(level, message):
    if level == "debug":
        logger.debug(message)
    elif level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "critical":
        logger.critical(message)


# Example usage
if __name__ == "__main__":
    file_path = 'data/processed/hngrams.csv'
    hybrid_ngrams = load_hybrid_ngram_patterns(file_path)
    print(hybrid_ngrams)
