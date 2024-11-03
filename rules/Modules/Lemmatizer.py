import sys
import os

# Add the path to morphinas_project
sys.path.append('C:/Users/Carlo Agas/Documents/GitHub/Pantasaa/morphinas_project')

from lemmatizer_client import initialize_stemmer, lemmatize_multiple_words

# Initialize the Morphinas lemmatizer once to reuse across function calls
gateway, lemmatizer = initialize_stemmer()

def escape_unwrapped_quotes(sentence):
    """Escape double quotes if they are not at the start and end of the sentence."""
    if '"' in sentence and not (sentence.startswith('"') and sentence.endswith('"')):
        sentence = sentence.replace('"', '""')
    return sentence

def wrap_sentence_with_commas(sentence):
    """Wrap sentence with double quotes if it contains a comma and isn't already wrapped."""
    if ',' in sentence and not (sentence.startswith('"') and sentence.endswith('"')):
        sentence = f'"{sentence}"'
    return sentence

def redo_escape_and_wrap(sentence):
    """Reapply double quotes and wrapping rules if conditions are met, avoiding redundant escapes."""
    sentence = escape_unwrapped_quotes(sentence)
    return wrap_sentence_with_commas(sentence)

def undo_escape_and_wrap(sentence):
    """Revert double quotes and wrapping for processing."""
    if sentence.startswith('"') and sentence.endswith('"'):
        sentence = sentence[1:-1]
    return sentence.replace('""', '"')

def lemmatize_sentence(sentence):
    """
    Calls the Morphinas lemmatizer to lemmatize a sentence and returns the lemmatized string.
    """
    try:
        # Unwrap and unescape the sentence before processing
        sentence = undo_escape_and_wrap(sentence)

        # Tokenize the sentence into words (you can also use the tokenize_sentence function you already have)
        words = sentence.split()

        # Use the Morphinas lemmatizer to lemmatize the words
        lemmatized_words = lemmatize_multiple_words(words, gateway, lemmatizer)

        # Join the lemmatized words back into a single string
        lemmatized_string = ' '.join(lemmatized_words)

        # Reapply escape and wrapping rules after lemmatization
        lemmatized_string = redo_escape_and_wrap(lemmatized_string)

        return lemmatized_string

    except Exception as e:
        print(f"Exception occurred during lemmatization: {e}")
        return sentence
