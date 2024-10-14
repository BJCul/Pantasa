import pandas as pd
import re
from collections import Counter
import tempfile
import subprocess
import os
import logging
from app.utils import log_message
from app.spell_checker import spell_check_sentence
from app.morphinas_project.lemmatizer_client import initialize_stemmer, lemmatize_multiple_words
from app.predefined_rules.rule_main import  apply_predefined_rules

# Initialize the Morphinas Stemmer
stemmer = initialize_stemmer()

logger = logging.getLogger(__name__)

jar = 'rules/Libraries/FSPOST/stanford-postagger.jar'
model = 'rules/Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'

def tokenize_sentence(sentence):
    """
    Tokenizes an input sentence into words and punctuation using regex.
    Handles words, punctuation, and special cases like numbers or abbreviations.
    
    Args:
    - sentence: The input sentence as a string.
    
    Returns:
    - A list of tokens.
    """
    
    # Tokenization pattern
    # Matches words, abbreviations, numbers, and punctuation
    token_pattern = re.compile(r'\w+|[^\w\s]')
    
    # Find all tokens in the sentence
    tokens = token_pattern.findall(sentence)
    
    return tokens


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
    mispelled_words, checked_sentence = spell_check_sentence(text_input)

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
    preprocessed_output = (tokens, lemmatized_words, tagged_tokens, checked_sentence, mispelled_words)
    
    # Log the final preprocessed output for better traceability
    log_message("info", f"Preprocessed Output: {preprocessed_output}")

    return [preprocessed_output]

# Load and create the rule pattern bank
def rule_pattern_bank():
    file_path = 'data/processed/hngrams.csv'  # Update with actual path
    hybrid_ngrams_df = pd.read_csv(file_path)

    # Create a dictionary to store the Rule Pattern Bank (Hybrid N-Grams + Predefined Rules)
    rule_pattern_bank = {}

    # Store the hybrid n-grams from the CSV file into the rule pattern bank
    for index, row in hybrid_ngrams_df.iterrows():
        pattern_id = row['Pattern_ID']
        hybrid_ngram = row['Final_Hybrid_N-Gram']
        rule_pattern_bank[pattern_id] = {'hybrid_ngram': hybrid_ngram}
    
    return rule_pattern_bank

# Step 2: Define the weighted Levenshtein distance function
def edit_weighted_levenshtein(input_ngram, pattern_ngram):
    input_tokens = input_ngram.strip().split()
    pattern_tokens = pattern_ngram.strip().split()
    
    len_input = len(input_tokens)
    len_pattern = len(pattern_tokens)
    
    # Create a matrix to store the edit distances
    distance_matrix = [[0] * (len_pattern + 1) for _ in range(len_input + 1)]
    
    # Initialize base case values (costs of insertions and deletions)
    for i in range(len_input + 1):
        distance_matrix[i][0] = i
    for j in range(len_pattern + 1):
        distance_matrix[0][j] = j

    # Define weights for substitution, insertion, and deletion
    substitution_weight = 0.8
    insertion_weight = 1.0
    deletion_weight = 1.0

    # Compute the distances
    for i in range(1, len_input + 1):
        for j in range(1, len_pattern + 1):
            input_token = input_tokens[i - 1]
            pattern_token = pattern_tokens[j - 1]
            # Use regex to check if input_token matches pattern_token
            try:
                if re.match(pattern_token, input_token): 
                    cost = 0
                else:
                    cost = substitution_weight  # Apply substitution weight if tokens differ
            except re.error as e:
                print(f"Regex error: {e} with pattern_token: '{pattern_token}' and input_token: '{input_token}'")

            distance_matrix[i][j] = min(
                distance_matrix[i - 1][j] + deletion_weight,    # Deletion
                distance_matrix[i][j - 1] + insertion_weight,  # Insertion
                distance_matrix[i - 1][j - 1] + cost           # Substitution
            )
    
    return distance_matrix[len_input][len_pattern]

# Step 3: Function to generate correction token tags based on the Levenshtein distance
def generate_correction_tags(input_ngram, pattern_ngram):
    input_tokens = input_ngram.split()
    pattern_tokens = pattern_ngram.split()
    
    tags = []
    
    for i, (input_token, pattern_token) in enumerate(zip(input_tokens, pattern_tokens)):
        if input_token == pattern_token:
            tags.append(f'KEEP_{input_token}_{pattern_token}')
        if input_token != pattern_token:
            if len(input_token) >= 2 and len(pattern_token) >= 2 and input_token.startswith(pattern_token[:2]): # For substitution errors (same POS group)
                tags.append(f'KEEP_{input_token}_{pattern_token}')
            else:  # General substitution (different POS group)
                tags.append(f'SUBSTITUTE_{input_token}_{pattern_token}')
    
    # Handle if there are extra tokens in the input or pattern
    if len(input_tokens) > len(pattern_tokens):
        extra_tokens = input_tokens[len(pattern_tokens):]
        for token in extra_tokens:
            tags.append(f'DELETE_{token}')
    elif len(pattern_tokens) > len(input_tokens):
        extra_tokens = pattern_tokens[len(input_tokens):]
        for token in extra_tokens:
            tags.append(f'INSERT_{token}')
    
    return tags

# Step 4: Function to generate n-grams of different lengths (from 3 to 7) from the input sentence
def generate_ngrams(input_tokens, n_min=3, n_max=7):
    ngrams = []
    for n in range(n_min, n_max + 1):
        for i in range(len(input_tokens) - n + 1):
            ngram = input_tokens[i:i+n]
            ngrams.append((" ".join(ngram), i))  # Track the starting index
    return ngrams

# Step 5: Suggestion phase - generate suggestions for corrections without applying them
def generate_suggestions(pos_tags):

    input_tokens = [pos_tag for word, pos_tag in pos_tags]
    
    # Generate token-level correction tracker
    token_suggestions = [{"token": token, "suggestions": [], "distances": []} for token in input_tokens]
    
    # Generate 3-gram to 7-gram sequences from the input sentence
    input_ngrams_with_index = generate_ngrams(input_tokens)
    
    # Iterate over each n-gram and compare it to the rule pattern bank
    for input_ngram, start_idx in input_ngrams_with_index:
        min_distance = float('inf')
        rule_bank = rule_pattern_bank()
        best_match = None

        for pattern_id, pattern_data in rule_bank.items():
            # Compare input n-gram with each pattern n-gram from the rule pattern bank
            pattern_ngram = pattern_data.get('hybrid_ngram')
            if pattern_ngram:
                distance = edit_weighted_levenshtein(input_ngram, pattern_ngram)
                if distance < min_distance:
                    min_distance = distance
                    best_match = pattern_ngram
            
        if best_match:
            correction_tags = generate_correction_tags(input_ngram, best_match)
            print(f"CORRECTION TAGS {correction_tags}")
            
            # Populate the token-level correction tracker
            input_ngram_tokens = input_ngram.split()
            for i, tag in enumerate(correction_tags):
                if start_idx + i < len(token_suggestions):  # Correctly map to original token index
                    token_suggestions[start_idx + i]["suggestions"].append(tag)
                    token_suggestions[start_idx + i]["distances"].append(min_distance)
    
    return token_suggestions

# Step 6: Correction phase - apply the suggestions to correct the input sentence
def apply_pos_corrections(token_suggestions):
    final_sentence = []
    
    # Iterate through the token_suggestions and apply the corrections
    for token_info in token_suggestions:
        suggestions = token_info["suggestions"]
        distances = token_info["distances"]
        
        # Step 1: Count the frequency of each exact suggestion (e.g., SUBSTITUTE_DTC_CC.*, KEEP_DTC_DT.*)
        suggestion_count = Counter(suggestions)  # Count occurrences of each exact suggestion
        print(f"COUNTER {suggestion_count}")

        # Step 2: Find the most frequent exact suggestion(s)
        most_frequent_suggestion = suggestion_count.most_common(1)[0][0]  # Get the most frequent exact suggestion
        
        # Step 3: Filter suggestions to only those matching the most frequent exact suggestion
        filtered_indices = [i for i, s in enumerate(suggestions) if s == most_frequent_suggestion]
        
        # Step 4: If multiple suggestions have the same frequency, pick the one with the lowest distance
        if len(filtered_indices) > 1:
            # Get the distances of the filtered suggestions
            filtered_distances = [distances[i] for i in filtered_indices]
            # Find the index of the smallest distance among the filtered suggestions
            best_filtered_index = filtered_distances.index(min(filtered_distances))
            # Use this index to get the corresponding best suggestion index
            best_index = filtered_indices[best_filtered_index]
        else:
            best_index = filtered_indices[0]
        
        best_suggestion = suggestions[best_index]

        # Apply the suggestion based on its type
        suggestion_type = best_suggestion.split("_")[0]

        if suggestion_type == "KEEP":
            final_sentence.append(token_info["token"])
        elif suggestion_type == "SUBSTITUTE":
            final_sentence.append(best_suggestion.split("_")[2])  # Apply substitution
        elif suggestion_type == "DELETE":
            continue  # Skip the token if DELETE is chosen
        elif suggestion_type == "INSERT":
            # Handle insertion if needed, but this should only be for missing tokens
            pass

    return " ".join(final_sentence)

def pantasa_checker(input_sentence, jar_path, model_path, rule_bank):
    
    # Preprocess the sentence (spell checking, tokenize, POS tag, and lemmatize)
    preprocessed_output = preprocess_text(input_sentence, jar_path, model_path)
    if not preprocessed_output:
        logger.error("Error during preprocessing.")
        return []
    
    tokens, lemmas, pos_tags, checked_sentence, misspelled_words = preprocessed_output[0]

    # POS matching to Hybrid n-gram
    log_message("info", f"POS TAGS: {pos_tags}")
    suggestions = generate_suggestions(pos_tags)

    # Rule corrected sentences obtain from applying predefined rules
    rule_corrected_text = apply_predefined_rules(checked_sentence)

    # POS Correction
    corrected_sentence = apply_pos_corrections(suggestions)
    print(f"Input Sentence: {input_sentence}")
    print(f"POS Corrected Sentence: {corrected_sentence}")
    print(f"Misspelled word: {misspelled_words}")
    print(f"Rule Corrected text: {rule_corrected_text}")

    return corrected_sentence, misspelled_words, rule_corrected_text

if __name__ == "__main__":
    input_text = "magtanim ay hindi biro"
    jar_path = 'rules/Libraries/FSPOST/stanford-postagger.jar'
    model_path = 'rules/Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    rule_bank = rule_pattern_bank()         

    corrected_sentence, misspelled_words, rule_corrected_text = pantasa_checker(input_text, jar_path, model_path, rule_bank)
    print(f"Final Corrected Sentence: {corrected_sentence}")
    print(f"Misspelled word: {misspelled_words}")
    print(f"Rule Corrected text: {rule_corrected_text}")
