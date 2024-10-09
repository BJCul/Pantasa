import re
import tempfile
import subprocess
import sys
import os
import csv

# Add the parent directory (Pantasa root) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary functions and modules
from morphinas_project.lemmatizer_client import initialize_stemmer, lemmatize_multiple_words
from rule_checker import process_input_ngram, weighted_levenshtein

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

# ----------------------- Error Detection with Suggestions ---------------------------
class ErrorDetectionService:
    def __init__(self, ngram_database, stemmer):
        self.ngram_database = ngram_database
        self.stemmer = stemmer
    
    def detect_errors(self, input_sentence, jar_path, model_path):
        # Step 1: Preprocess the sentence (tokenize, tag, lemmatize)
        preprocessed_sentences = preprocess_text(input_sentence, jar_path, model_path)
        if not preprocessed_sentences:
            return "Error: Preprocessing failed."

        for tokens, lemmas, tagged_tokens in preprocessed_sentences:
            pos_tags = [pos for word, pos in tagged_tokens]

            # Step 2: Compare POS Tags with Hybrid N-Grams
            matching_ngrams = self.ngram_database.get_matching_ngrams(pos_tags)

            if matching_ngrams:
                return False, "No error detected."

            # Step 3: If no match, generate suggestions based on Levenshtein distance
            suggestions = self.generate_suggestions(tokens, lemmas, pos_tags)
            return True, suggestions

    def generate_suggestions(self, tokens, lemmas, pos_tags):
        suggestions = []
        
        # Implement the use of manually created rules (like na/-ng, affix merging/unmerging, etc.)
        for i in range(len(tokens)):
            # Rule 1: Use of "na/-ng"
            if i < len(tokens) - 1 and tokens[i + 1] == "na" and tokens[i][-1] in "aeiou":
                suggestions.append(f"Replace 'na' with '-ng' in '{tokens[i]} na' -> '{tokens[i]}ng'")
            if i < len(tokens) - 1 and tokens[i + 1] == "na" and tokens[i][-1] == "n":
                suggestions.append(f"Replace 'na' with '-g' in '{tokens[i]} na' -> '{tokens[i]}g'")

            # Rule 2: Merging Incorrectly Unmerged Affixes
            if tokens[i].startswith("pinag") and tokens[i] == "pinag" and i < len(tokens) - 1:
                suggestions.append(f"Merge '{tokens[i]}' with '{tokens[i + 1]}' -> 'pinagsikapan'")

            # Rule 3: Removing Incorrect Hyphenation
            if "-" in tokens[i] and tokens[i].replace("-", "") in lemmas:
                suggestions.append(f"Remove hyphen from '{tokens[i]}' -> '{tokens[i].replace('-', '')}'")

            # Rule 4: Unmerging "mas" from Verbs
            if tokens[i].startswith("mas") and len(tokens[i]) > 3:
                remaining_word = tokens[i][3:]
                if remaining_word in lemmas:
                    suggestions.append(f"Unmerge 'mas' from '{tokens[i]}' -> 'mas {remaining_word}'")

            # Suggestion via Levenshtein distance (edit-based correction)
            for hybrid_ngram in self.ngram_database.hybrid_ngrams:
                hybrid_pattern = hybrid_ngram['ngram_pattern']
                for j, token in enumerate(tokens):
                    if j < len(hybrid_pattern):
                        dist = weighted_levenshtein(token, hybrid_pattern[j])
                        if dist < 2:  # Threshold for suggesting corrections
                            suggestions.append(f"Replace '{token}' with '{hybrid_pattern[j]}'")

        return suggestions

# ----------------------- N-Gram Database Handling ---------------------------
class NGramDatabase:
    def __init__(self):
        self.hybrid_ngrams = []

    def load_from_csv(self, file_path):
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                pattern_id = row['Pattern_ID']
                hybrid_ngram = row['Final_Hybrid_N-Gram'].split()
                self.hybrid_ngrams.append({
                    'pattern_id': pattern_id,
                    'ngram_pattern': hybrid_ngram
                })
    
    def get_matching_ngrams(self, pos_tags):
        matching_ngrams = []
        for hybrid_ngram in self.hybrid_ngrams:
            if len(pos_tags) == len(hybrid_ngram['ngram_pattern']):
                match = True
                for i in range(len(pos_tags)):
                    if not re.match(hybrid_ngram['ngram_pattern'][i], pos_tags[i]):
                        match = False
                        break
                if match:
                    matching_ngrams.append(hybrid_ngram['pattern_id'])
        return matching_ngrams

# ----------------------- Main Function ----------------------------
def main():
    # Initialize the database and load hybrid n-grams from a CSV file
    ngram_database = NGramDatabase()
    ngram_database.load_from_csv('data/processed/hngrams.csv')  # Assumed CSV file with hybrid n-grams

    # Initialize the error detection service
    error_detection_service = ErrorDetectionService(ngram_database, stemmer)

    # Input sentence
    input_sentence = "kumakain ng mga mga mga bata mga mga mga"

    jar_path = r'C:\Projects\Pantasa\rules\Libraries\FSPOST\stanford-postagger.jar'
    model_path = r'C:\Projects\Pantasa\rules\Libraries\FSPOST\filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'

    # Detect errors
    has_error, message = error_detection_service.detect_errors(input_sentence, jar_path, model_path)

    # Output the result
    if has_error:
        print("Error detected:")
        for suggestion in message:
            print(suggestion)
    else:
        print(message)

if __name__ == "__main__":
    main()
