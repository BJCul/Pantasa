# app/preprocess.py
import re
import tempfile
import subprocess
import sys
import os
import logging
from utils import log_message

# Import necessary functions from the Morphinas lemmatizer
from morphinas_project.lemmatizer_client import initialize_stemmer, lemmatize_multiple_words

# Initialize the Morphinas Stemmer
stemmer = initialize_stemmer()

logger = logging.getLogger(__name__)

def tokenize_sentence(sentence):
    """
    Tokenizes a sentence into words and punctuation using regex.
    """
    token_pattern = re.compile(r'\w+|[^\w\s]')
    return token_pattern.findall(sentence)

def pos_tagging(tokens, jar_path, model_path):
    """
    Tags tokens using the FSPOST Tagger via subprocess.

    Args:
    - tokens: List of tokens to tag.
    - jar_path: Path to the Stanford POS Tagger jar file.
    - model_path: Path to the Tagalog POS Tagger model file.

    Returns:
    - List of (token, pos_tag) tuples.
    """
    sentence = ' '.join(tokens)
    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(sentence)
            temp_file_path = temp_file.name

        command = [
            'java', '-mx1g',
            '-cp', jar_path,
            'edu.stanford.nlp.tagger.maxent.MaxentTagger',
            '-model', model_path,
            '-textFile', temp_file_path
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

        os.unlink(temp_file_path)  # Delete the temporary file

        if process.returncode != 0:
            raise Exception(f"POS tagging process failed: {error.decode('utf-8')}")

        tagged_output = output.decode('utf-8').strip().split()
        tagged_tokens = [tuple(tag.split('|')) for tag in tagged_output if '|' in tag]

        return tagged_tokens

    except Exception as e:
        print(f"Error during POS tagging: {e}")
        return []

def preprocess_text(text_input, jar_path, model_path):
    """
    Preprocesses the input text by tokenizing, POS tagging, and lemmatizing.

    Args:
    - text_input: The input sentence(s) to preprocess.
    - jar_path: Path to the FSPOST Tagger jar file.
    - model_path: Path to the FSPOST Tagger model file.

    Returns:
    - List of tuples containing (tokens, lemmas, pos_tags).
    """
    tokens = tokenize_sentence(text_input)
    tagged_tokens = pos_tagging(tokens, jar_path, model_path)

    if not tagged_tokens:
        log_message("Error: Tagged tokens are empty.")
        return []

    words = [word for word, pos in tagged_tokens]
    gateway, lemmatizer = stemmer
    lemmatized_words = lemmatize_multiple_words(words, gateway, lemmatizer)
    log_message("info", f"Lemmatized Words: {lemmatized_words}")


    preprocessed_output = (tokens, lemmatized_words, tagged_tokens)
    return [preprocessed_output]

# Example usage
if __name__ == "__main__":
    jar_path = r'C:\Projects\Pantasa\rules\Libraries\FSPOST\stanford-postagger.jar'
    model_path = r'C:\Projects\Pantasa\rules\Libraries\FSPOST\filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'

    sentence = "kumain ang bata ng mansanas"

    preprocessed_text = preprocess_text(sentence, jar_path, model_path)
    print(preprocessed_text)
