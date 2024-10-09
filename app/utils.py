# app/utils.py

import csv

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
