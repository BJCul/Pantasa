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
    token_pattern = re.compile(r'\w+|[^\w\s]')  # Tokenizes words and punctuation
    return token_pattern.findall(sentence)

# ----------------------- FSPOST POS Tagging Function ----------------------
def pos_tagging(tokens, jar_path, model_path):
    sentence = ' '.join(tokens)
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write(sentence)
        temp_file.flush()
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
    sentences = [text_input]
    preprocessed_sentences = []

    for sentence in sentences:
        tokens = tokenize_sentence(sentence)
        tagged_tokens = pos_tagging(tokens, jar_path, model_path)
        if not tagged_tokens:
            return []  # Early return in case of tagging failure

        words = [word for word, pos in tagged_tokens]
        gateway, lemmatizer = stemmer
        lemmatized_words = lemmatize_multiple_words(words, gateway, lemmatizer)

        preprocessed_sentences.append((tokens, lemmatized_words, tagged_tokens))

    return preprocessed_sentences

# ----------------------- Error Detection with Suggestions ---------------------------
class ErrorDetectionService:
    def __init__(self, ngram_database, stemmer):
        self.ngram_database = ngram_database
        self.stemmer = stemmer
    
    def detect_errors(self, input_sentence, jar_path, model_path):
        preprocessed_sentences = preprocess_text(input_sentence, jar_path, model_path)
        if not preprocessed_sentences:
            return "Error: Preprocessing failed."

        suggestions = []
        for tokens, lemmas, tagged_tokens in preprocessed_sentences:
            pos_tags = [pos for word, pos in tagged_tokens]
            matching_ngrams = self.ngram_database.get_matching_ngrams(pos_tags)

            if matching_ngrams:
                return False, "No error detected."

            # Apply manually created rules
            suggestions.extend(self.apply_manual_rules(tokens))

            # Generate suggestions using Levenshtein
            suggestions.extend(self.generate_suggestions(tokens, lemmas, pos_tags))
        
        return True, suggestions

    def apply_manual_rules(self, tokens):
        suggestions = []
        for i, token in enumerate(tokens):
            if i + 1 < len(tokens):
                token_a, token_b = tokens[i], tokens[i + 1]

                # Rule 1: na/ng correction
                if token_a.endswith("n") and token_b == "na":
                    suggestions.append(f"Replace '{token_a} na' with '{token_a}-ng'")
                elif token_a.endswith("a") and token_b == "na":
                    suggestions.append(f"Replace '{token_a} na' with '{token_a}-ng'")

                # Rule 2: Unmerge 'mas' from verbs
                if token_a.startswith("mas") and token_a[3:] in dictionary:  # Simple check
                    suggestions.append(f"Unmerge 'mas' from '{token_a}' -> 'mas {token_a[3:]}'")

                # Rule 3: Merge incorrectly unmerged affixes
                if token_a == "pinag" and token_b in dictionary:
                    suggestions.append(f"Merge 'pinag' with '{token_b}' -> 'pinags{token_b}'")

                # Rule 4: Removing incorrect hyphenation
                if '-' in token_a and token_a.replace('-', '') in dictionary:
                    suggestions.append(f"Remove hyphen from '{token_a}' -> '{token_a.replace('-', '')}'")
        return suggestions

    def generate_suggestions(self, tokens, lemmas, pos_tags):
        suggestions = []
        for hybrid_ngram in self.ngram_database.hybrid_ngrams:
            hybrid_pattern = hybrid_ngram['ngram_pattern']
            for i, token in enumerate(tokens):
                if i < len(hybrid_pattern):
                    dist = weighted_levenshtein(token, hybrid_pattern[i])
                    if dist < 2:  # Threshold for suggesting corrections
                        suggestions.append(f"Suggestion: Replace '{token}' with '{hybrid_pattern[i]}'")
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
    input_sentence = "Ang mga tao ay hndi na talaga natuto."

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
