import re
import tempfile
import subprocess
import os
import logging
from utils import log_message
from spell_checker import spell_check_sentence
from morphinas_project.lemmatizer_client import initialize_stemmer, lemmatize_multiple_words

# Initialize the Morphinas Stemmer
stemmer = initialize_stemmer()

logger = logging.getLogger(__name__)

def tokenize_sentence(sentence):
    """
    Tokenizes a sentence into words and punctuation using regex.
    Words inside << >> are tokenized as-is, including any punctuation.
    """
    # Define pattern to detect words wrapped in << >>
    wrapped_pattern = re.compile(r'<<[^<>]+>>')

    # Find all words wrapped in << >>
    wrapped_tokens = wrapped_pattern.findall(sentence)

    # Remove wrapped tokens from the sentence temporarily to avoid splitting them
    for wrapped in wrapped_tokens:
        sentence = sentence.replace(wrapped, '__WRAPPED__')

    # Tokenize the rest of the sentence (normal tokenization)
    token_pattern = re.compile(r'\w+|[^\w\s]')
    tokens = token_pattern.findall(sentence)

    # Reinsert the wrapped tokens back into the tokenized sentence
    final_tokens = []
    for token in tokens:
        if token == '__WRAPPED__':
            final_tokens.append(wrapped_tokens.pop(0))  # Put the wrapped token back
        else:
            final_tokens.append(token)

    return final_tokens


import subprocess
import tempfile
import os

jar = 'rules/Libraries/FSPOST/stanford-postagger.jar'
model = 'rules/Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'

def pos_tagging(tokens, jar_path=jar, model_path=model):
    """
    Tags tokens using the FSPOST Tagger via subprocess.
    """
    # Prepare tokens for tagging
    java_tokens = []
    tagged_tokens = []

    for token in tokens:
        # Check if the token is a tuple (e.g., (word, pos_tag)) and extract the word
        if isinstance(token, tuple):
            token = token[0]  # Extract the first element, which is the actual word

        java_tokens.append(token)  # Send to Java POS tagger for normal tagging

    if java_tokens:
        # Only call the Java POS tagger if there are tokens to tag
        sentence = ' '.join(java_tokens)
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
            java_tagged_tokens = [tuple(tag.split('|')) for tag in tagged_output if '|' in tag]

            # Append the tagged tokens from Java POS tagger
            tagged_tokens.extend(java_tagged_tokens)

        except Exception as e:
            log_message("error", f"Error during POS tagging: {e}")
            return []

    return tagged_tokens


def preprocess_text(text_input, jar_path, model_path):
    """
    Preprocesses the input text by tokenizing, POS tagging, lemmatizing, and checking spelling.
    Args:
    - text_input: The input sentence to preprocess.
    - jar_path: Path to the FSPOST Tagger jar file.
    - model_path: Path to the FSPOST Tagger model file.
    """
    # Step 1: Spell check the sentence
    checked_sentence = spell_check_sentence(text_input)

    # Step 2: Tokenize the sentence
    tokens = tokenize_sentence(checked_sentence)

    # Step 3: POS tagging using the provided jar and model paths
    tagged_tokens = pos_tagging(tokens, jar_path=jar_path, model_path=model_path)

    if not tagged_tokens:
        log_message("error", "Tagged tokens are empty.")
        return []

    words = [word for word, pos in tagged_tokens]

    # Step 4: Lemmatization
    gateway, lemmatizer = stemmer
    lemmatized_words = lemmatize_multiple_words(words, gateway, lemmatizer)
    log_message("info", f"Lemmatized Words: {lemmatized_words}")

    # Step 5: Prepare the preprocessed output
    preprocessed_output = (tokens, lemmatized_words, tagged_tokens, checked_sentence)
    
    # Log the final preprocessed output for better traceability
    log_message("info", f"Preprocessed Output: {preprocessed_output}")

    return [preprocessed_output]


# Example usage
if __name__ == "__main__":
    

    sentence = "kumain ang bata ng mansana na asda dasfa"

    preprocessed_text = preprocess_text(sentence, jar_path=jar, model_path=model)
    print(preprocessed_text)
