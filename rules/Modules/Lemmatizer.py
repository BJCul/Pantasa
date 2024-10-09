import sys
import os

# Add the path to morphinas_project
sys.path.append(r'C:\Projects\Pantasa\app')

from morphinas_project.lemmatizer_client import initialize_stemmer, lemmatize_multiple_words

# Initialize the Morphinas lemmatizer once to reuse across function calls
gateway, lemmatizer = initialize_stemmer()

def lemmatize_sentence(sentence):
    """
    Calls the Morphinas lemmatizer to lemmatize a sentence and returns the lemmatized string.
    """
    try:
        # Check if the sentence is enclosed in single quotation marks with a comma before the closing mark
        if sentence.startswith('"') and sentence.endswith('"') and ',' in sentence:
            sentence = sentence[1:-2]  # Remove the opening and closing quotation marks and the comma

        # Tokenize the sentence into words (you can also use the tokenize_sentence function you already have)
        words = sentence.split()

        # Use the Morphinas lemmatizer to lemmatize the words
        lemmatized_words = lemmatize_multiple_words(words, gateway, lemmatizer)

        # Join the lemmatized words back into a single string
        lemmatized_string = ' '.join(lemmatized_words)

        # Add back the quotation marks if they were removed
        if ',' in sentence:
            lemmatized_string = '"' + lemmatized_string + '"'

        return lemmatized_string

    except Exception as e:
        print(f"Exception occurred during lemmatization: {e}")
        return sentence



