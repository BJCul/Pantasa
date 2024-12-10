import sys
import re

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

terminator = '.!?,;:'

def tokenize_with_terminator(sentence):
    """
    Tokenizes the sentence while preserving punctuation in the `terminator` list
    and ensuring their original placement is maintained.
    """
    # Regex pattern to match words, numbers, and terminators
    pattern = fr"[a-zA-Z0-9]+|[{re.escape(terminator)}]|[\s]+"
    tokens = re.findall(pattern, sentence)
    return tokens

def lemmatize_sentence(sentence):
    """
    Calls the Morphinas lemmatizer to lemmatize a sentence while preserving punctuation
    in its original position without adding spaces around them.
    """
    try:
        # Unwrap and unescape the sentence before processing
        sentence = undo_escape_and_wrap(sentence)

        # Tokenize the sentence into words and preserve punctuation with placement
        tokens = tokenize_with_terminator(sentence)

        # Use the Morphinas lemmatizer to lemmatize only alphanumeric tokens
        lemmatized_tokens = []
        for token in tokens:
            if token.strip() and token.isalnum():  # Only lemmatize words
                lemmatized_tokens.extend(lemmatize_multiple_words([token], gateway, lemmatizer))
            else:
                lemmatized_tokens.append(token)  # Preserve spaces and punctuation as is

        # Join tokens without adding extra spaces
        lemmatized_string = ''.join(lemmatized_tokens)

        # Reapply escape and wrapping rules after lemmatization
        lemmatized_string = redo_escape_and_wrap(lemmatized_string)

        return lemmatized_string

    except Exception as e:
        print(f"Exception occurred during lemmatization: {e}")
        return sentence

test_sentences = [
    "Kapag walang bulakbol cops na sa Metro, magdadalawang-isip din ang mga kriminal na tumira dahil t’yak sa kulungan ang bagsak nila.",
    "Makikita na ang politikal na aktibidad patungkol sa press con na ito ang nabalitang kasong hit-and-run ng isang Jose Antonio Sanvicente sa isang security",
    "Ang tinutukoy ko mga kosa ay itong    mga pulis na  pagkatapos mag-check atten­dance­ ay sasakay sa kanilang motor at lilisanin na ang duty area nila",
]

for test_sentence in test_sentences:
    lemma = lemmatize_sentence(test_sentence)
    print(lemma)