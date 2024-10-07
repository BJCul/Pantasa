import re
import tempfile
import subprocess
import sys
import os

# Add the parent directory (Pantasa root) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary functions and modules
from morphinas_project.lemmatizer_client import initialize_stemmer, lemmatize_multiple_words
from rule_checker import process_input_ngram

# Initialize Morphinas Stemmer
stemmer = initialize_stemmer()

# ----------------------- Tokenization Function ---------------------------
def tokenize_sentence(sentence):
    """
    Tokenizes a sentence into words and punctuation using regex.
    """
    token_pattern = re.compile(r'\w+|[^\w\s]')  # Tokenizes words and punctuation
    return token_pattern.findall(sentence)

# ----------------------- FSPOST POS Tagging Function ----------------------
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
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write(sentence)
        temp_file.flush()  # Ensure the content is written to the file
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

    if process.returncode != 0:
        print(f"Error in POS tagging process: {error.decode('utf-8')}")
        return []

    tagged_output = output.decode('utf-8').strip().split()
    tagged_tokens = [tuple(tag.split('|')) for tag in tagged_output if '|' in tag]

    return tagged_tokens

# ----------------------- Preprocessing Function ---------------------------
def preprocess_text(text_input, jar_path, model_path):
    """
    Preprocesses the input text by tokenizing, POS tagging, and lemmatizing.

    Args:
    - text_input: The input sentence(s) to preprocess.
    - jar_path: Path to the FSPOST Tagger jar file.
    - model_path: Path to the FSPOST Tagger model file.

    Returns:
    - Tokenized, Lemmatized, POS-tagged text as a tuple (tokens, lemmas, pos_tags).
    """
    sentences = [text_input]  # Wrap text_input in a list if it's a single sentence
    preprocessed_sentences = []

    for sentence in sentences:
        tokens = tokenize_sentence(sentence)
        print(f"Tokens: {tokens}")

        tagged_tokens = pos_tagging(tokens, jar_path, model_path)
        print(f"Tagged Tokens: {tagged_tokens}")

        if not tagged_tokens:
            print("Error: Tagged tokens are empty, returning an empty list.")
            return []  # Early return in case of tagging failure

        words = [word for word, pos in tagged_tokens]
        gateway, lemmatizer = stemmer
        lemmatized_words = lemmatize_multiple_words(words, gateway, lemmatizer)
        print(f"Lemmatized Words: {lemmatized_words}")

        # Check to ensure the correct tuple structure
        preprocessed_sentences.append((tokens, lemmatized_words, tagged_tokens))
        print(f"Preprocessed Sentence Structure: {preprocessed_sentences[-1]}")

    return preprocessed_sentences

# ----------------------- Integration with Rule Checking --------------------
def run_rule_checker_on_preprocessed_text(preprocessed_output):
    """
    Processes the preprocessed text with the rule checker and generates correction suggestions.
    
    Args:
    - preprocessed_output: The output from preprocess_text function (tokens, lemmas, pos_tags).
    """
    for preprocessed_item in preprocessed_output:
        if len(preprocessed_item) != 3:
            print(f"Error: Expected 3 items in the tuple, but got {len(preprocessed_item)}.")
            continue

        tokens, lemmas, pos_tags = preprocessed_item
        print(f"Processing tokens: {tokens}, lemmas: {lemmas}, pos_tags: {pos_tags}")
        
        # Combine the POS tags into a single n-gram format (example: "NN VB DT")
        ngram_input = ' '.join([f"{pos}" for word, pos in pos_tags])
        print(f"Generated N-gram: {ngram_input}")
        
        # Pass the generated n-gram to the rule-checking module
        corrections = process_input_ngram(ngram_input)

        # Display the corrections and suggestions
        print("Corrections:")
        for correction in corrections:
            print(f"Pattern ID: {correction['pattern_id']}, Distance: {correction['distance']}, Tags: {correction['correction_tags']}")

# ----------------------- Main Workflow Example ----------------------------
if __name__ == "__main__":
    jar_path = r'C:\Projects\Pantasa\rules\Libraries\FSPOST\stanford-postagger.jar'
    model_path = r'C:\Projects\Pantasa\rules\Libraries\FSPOST\filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'

    sentence = "Andaming mga kabataan ang lulong sa pinagbabawal na gamot."
    
    # Step 1: Preprocess the text (Tokenize, POS tag, and Lemmatize)
    preprocessed_text = preprocess_text(sentence, jar_path, model_path)

    # Step 2: Pass the preprocessed output to the rule checker
    run_rule_checker_on_preprocessed_text(preprocessed_text)
