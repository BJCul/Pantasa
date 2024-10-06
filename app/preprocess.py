import re
import tempfile
import subprocess
import sys
import os

# Add the parent directory (Pantasa root) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morphinas_project.lemmatizer_client import initialize_stemmer, lemmatize_multiple_words

# Initialize Morphinas Stemmer
stemmer = initialize_stemmer()

# ----------------------- Tokenization Function ---------------------------
def tokenize_sentence(sentence):
    """
    Tokenizes a sentence into words and punctuation using regex.
    """
    token_pattern = re.compile(r'\w+|[^\w\s]')  # Tokenizes words and punctuation
    return token_pattern.findall(sentence)

# ----------------------- FSPOST POS Tagging Function ---------------------------
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
    # Prepare the input sentence
    sentence = ' '.join(tokens)

    # Use a temporary file to simulate the command-line behavior
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write(sentence)
        temp_file.flush()  # Ensure the content is written to the file
        
        temp_file_path = temp_file.name

    # Command to run the Stanford POS Tagger (FSPOST)
    command = [
        'java', '-mx1g',  # Increase memory allocation here
        '-cp', jar_path,
        'edu.stanford.nlp.tagger.maxent.MaxentTagger',
        '-model', model_path,
        '-textFile', temp_file_path  # Pass the temp file as input
    ]

    # Execute the command and capture the output and error
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    # Check if the Java process returned an error
    if process.returncode != 0:
        print(f"Error in POS tagging process: {error.decode('utf-8')}")
        return []

    # Process the raw output by splitting each word|tag pair
    tagged_output = output.decode('utf-8').strip().split()
    tagged_tokens = [tuple(tag.split('|')) for tag in tagged_output if '|' in tag]  # Correctly split by '|'

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
    - Preprocessed text as a string.
    """

    # Step 1: Sentence Tokenization
    sentences = [text_input]  # Wrap text_input in a list if it's a single sentence

    preprocessed_sentences = []

    # Step 2: Process each sentence individually
    for sentence in sentences:
        # Tokenize sentence into words
        tokens = tokenize_sentence(sentence)
        
        print(f"Tokens: {tokens}")

        # Step 3: POS Tagging using FSPOST Tagger
        tagged_tokens = pos_tagging(tokens, jar_path, model_path)
        
        print(f"Tagged Tokens: {tagged_tokens}")

        # Step 4: Lemmatization using Morphinas
        words = [word for word, pos in tagged_tokens]
        
        # Unpack the stemmer tuple into gateway and lemmatizer
        gateway, lemmatizer = stemmer
        
        # Pass both the gateway and lemmatizer to the lemmatization function
        lemmatized_words = lemmatize_multiple_words(words, gateway, lemmatizer)

        # Combine the lemmatized words back into a sentence
        lemmatized_sentence = ' '.join(lemmatized_words)
        preprocessed_sentences.append(lemmatized_sentence)
    
    # Return the preprocessed text (lemmatized sentences combined)
    return ' '.join(preprocessed_sentences)


# ----------------------- Main Workflow Example ---------------------------
if __name__ == "__main__":
    # File paths for the FSPOST Tagger and Morphinas
    jar_path = r'C:\Projects\Pantasa\rules\Libraries\FSPOST\stanford-postagger.jar'  # Adjust the path to the JAR file
    model_path = r'C:\Projects\Pantasa\rules\Libraries\FSPOST\filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'  # Adjust the path to the model file

    # Input sentence
    sentence = "kakain kumakain nagkakainan kakainin"
    
    # Step 1: Preprocessing
    preprocessed_text = preprocess_text(sentence, jar_path, model_path)
    print(f"Preprocessed Text: {preprocessed_text}")

