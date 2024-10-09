import re
import os
import sys
import subprocess
import tempfile

# Add the parent directory (Pantasa root) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morphinas_project.lemmatizer_client import initialize_stemmer, lemmatize_multiple_words  # Import Morphinas client

# ----------------------- Preprocessing Module ---------------------------
class SentenceTokenizer:
    def __init__(self):
        self.token_pattern = re.compile(r'\w+|[^\w\s]')  # Tokenizes words and punctuation

    def tokenize(self, sentence):
        return self.token_pattern.findall(sentence)


from py4j.java_gateway import JavaGateway

class MorphinasLemmatizer:
    def __init__(self):
        # Initialize the Morphinas stemmer using Py4J once
        self.gateway = JavaGateway()
        self.stemmer = self.gateway.entry_point.initialize_stemmer()

    def lemmatize(self, words):
        try:
            # Create a Java String array for the words
            java_words_array = self.gateway.new_array(self.gateway.jvm.String, len(words))
            
            # Fill the Java array with the words
            for i, word in enumerate(words):
                java_words_array[i] = word

            # Use Py4J to call the lemmatizer for multiple words
            lemmas_java_array = self.stemmer.lemmatizeMultiple(java_words_array)

            # Convert the Java array back to a Python list
            lemmas = [lemmas_java_array[i] for i in range(len(lemmas_java_array))]

        except Exception as e:
            print(f"Error during lemmatization: {e}")
            lemmas = words  # If an error occurs, keep the original words
        return lemmas


class FSPOSTagger:
    def __init__(self, jar_path, model_path):
        self.jar_path = jar_path
        self.model_path = model_path

    def tag(self, tokens):
        # Prepare the input sentence
        sentence = ' '.join(tokens)

        # Use a temporary file to simulate the command-line behavior
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(sentence)
            temp_file.flush()  # Ensure the content is written to the file
            
            temp_file_path = temp_file.name

        # Command to run the Stanford POS Tagger (FSPOST)
        command = [
            'java', '-mx300m', '-cp', self.jar_path,
            'edu.stanford.nlp.tagger.maxent.MaxentTagger',
            '-model', self.model_path,
            '-textFile', temp_file_path  # Pass the temp file as input
        ]

        # Execute the command and capture the output
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

        if process.returncode != 0:
            print(f"POS Tagger Error: {error.decode('utf-8')}")
            return []

        # Process the raw output by splitting each word|tag pair
        tagged_output = output.decode('utf-8').strip().split()
        tagged_tokens = [tuple(tag.split('|')) for tag in tagged_output if '|' in tag]  # Correctly split by '|'

        return tagged_tokens


class PreprocessingModule:
    def __init__(self, lemmatizer, pos_tagger):
        self.tokenizer = SentenceTokenizer()
        self.lemmatizer = lemmatizer
        self.pos_tagger = pos_tagger

    def process(self, sentence):
        # Step 1: Tokenize the sentence
        tokens = self.tokenizer.tokenize(sentence)

        # Step 2: Lemmatize the tokens
        lemmas = self.lemmatizer.lemmatize(tokens)

        # Step 3: POS Tag the tokens
        pos_tags = self.pos_tagger.tag(tokens)

        return tokens, lemmas, pos_tags


# ----------------------- Main Workflow Example ---------------------------
if __name__ == "__main__":
    # File paths for the FSPOST Tagger and Morphinas
    jar_path = r'C:/Projects/Pantasa/rules/Libraries/FSPOST/stanford-postagger.jar'  # Adjust the path to the JAR file
    model_path = r'C:/Projects/Pantasa/rules/Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'  # Adjust the path to the model file

    # Instantiate the Morphinas lemmatizer (no need for the JAR path, using Py4J)
    lemmatizer = MorphinasLemmatizer()

    # Initialize FSPOST POS Tagger
    pos_tagger = FSPOSTagger(jar_path, model_path)

    # Instantiate the Preprocessing Module
    preprocessor = PreprocessingModule(lemmatizer, pos_tagger)

    # Input sentence
    sentence = "siya ay kumain nang mansanas"
    
    # Step 1: Preprocessing
    tokens, lemmas, pos_tags = preprocessor.process(sentence)
    print(f"Tokens: {tokens}\nLemmas: {lemmas}\nPOS Tags: {pos_tags}")
