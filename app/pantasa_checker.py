import pandas as pd
import re
from collections import Counter
import tempfile
import subprocess
import os
import logging
from app.grammar_checker import spell_check_incorrect_words
from app.utils import log_message
from app.spell_checker import load_dictionary, spell_check_sentence
from app.morphinas_project.lemmatizer_client import initialize_stemmer, lemmatize_multiple_words
from app.predefined_rules.rule_main import  apply_predefined_rules, apply_predefined_rules_post, apply_predefined_rules_pre

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
    logger.debug(f"Tokens: {tokens}")
    
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
                'java', '-mx300m',
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
            logger.debug(f"POS Tagged Tokens: {tagged_tokens}")


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
def rule_pattern_bank(rule_path):
    hybrid_ngrams_df = pd.read_csv(rule_path)

    rule_pattern_bank = {}

    for index, row in hybrid_ngrams_df.iterrows():
        pattern_id = row['Pattern_ID']
        hybrid_ngram = row['Hybrid_N-Gram']
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

# Step 3: Function to generate correction token tags based on the Levenshtein distance
def generate_suggestions(pos_tags, rule_path):
    input_tokens = [pos_tag for word, pos_tag in pos_tags]

    # Generate token-level correction tracker
    token_suggestions = [{"token": token, "suggestions": [], "distances": []} for token in input_tokens]


    # Initialize insertion suggestions dictionary
    insertion_suggestions = {i: [] for i in range(len(input_tokens) + 1)}

    # Generate n-grams
    input_ngrams_with_index = generate_ngrams(input_tokens)

    # Load the rule bank
    rule_bank = rule_pattern_bank(rule_path)

    # Iterate over each n-gram
    for input_ngram, start_idx in input_ngrams_with_index:
        min_distance = float('inf')
        best_matches = []

        for pattern_id, pattern_data in rule_bank.items():
            pattern_ngram = pattern_data.get('hybrid_ngram')

            if pattern_ngram:
                distance = edit_weighted_levenshtein(input_ngram, pattern_ngram)

                if distance < min_distance:
                    min_distance = distance
                    best_matches = [(pattern_ngram, distance)]
                elif distance == min_distance:
                    best_matches.append((pattern_ngram, distance))

        # Collect suggestions from all best matches
        for best_match, distance in best_matches:
            correction_tags = generate_correction_tags(input_ngram, best_match)
            print(f"CORRECTION TAGS {correction_tags}")

            input_ngram_tokens = input_ngram.split()
            for i, tag in enumerate(correction_tags):
                if tag.startswith("INSERT"):
                    insert_token = tag.split("_")[1]
                    position = start_idx + i
                    insertion_suggestions[position].append(insert_token)
                else:
                    if start_idx + i < len(token_suggestions):
                        token_suggestions[start_idx + i]["suggestions"].append(tag)
                        token_suggestions[start_idx + i]["distances"].append(distance)

    return token_suggestions, insertion_suggestions



def load_pos_tag_dictionary(pos_tag, pos_path):
    """
    Load the POS tag dictionary based on the specific or generalized POS tag.
    
    Args:
    - pos_tag (str): The POS tag to search for.
    - pos_path (str): The base path where the CSV files are stored.
    
    Returns:
    - words (list): List of words from the corresponding POS tag CSV files.
    """
    
    # Print all files in the directory for debugging purposes
    print(f"Available files in {pos_path}: {os.listdir(pos_path)}")
    
    # 1. If the tag is an exact match (e.g., VBAF), load the corresponding file
    csv_file_name = f"{pos_tag}_words.csv"
    exact_file_path = os.path.join(pos_path, csv_file_name)
    
    if os.path.exists(exact_file_path):
        print(f"Loading file for exact POS tag: {pos_tag}")
        return load_csv_words(exact_file_path)
    
    # 2. If the tag is generalized (e.g., VB.*), load all matching files
    if '.*' in pos_tag:
        generalized_tag_pattern = re.sub(r'(.*)\.\*', r'\1', pos_tag)
        matching_words = []

        # List all files in the directory and find files starting with the generalized POS tag (e.g., vbaf_words.csv, vbof_words.csv, vbtr_words.csv)
        for file_name in os.listdir(pos_path):
            if file_name.startswith(generalized_tag_pattern):
                file_path = os.path.join(pos_path, file_name)
                print(f"Loading file for generalized POS tag: {file_name}")
                matching_words.extend(load_csv_words(file_path))  # Add words from all matching files

        # If no matching files were found, raise an error
        if not matching_words:
            raise FileNotFoundError(f"No files found for POS tag pattern: {pos_tag}")
        
        return matching_words
    
    # 3. If no generalized or exact match, raise an error
    raise FileNotFoundError(f"CSV file for POS tag '{pos_tag}' not found")

def load_csv_words(file_path):
    """
    Load the words from a CSV file.
    
    Args:
    - file_path (str): The file path to the CSV file.
    
    Returns:
    - words (list): List of words from the CSV file.
    """
    # Load the CSV into a pandas DataFrame and return the first column as a list
    df = pd.read_csv(file_path, header=None)
    words = df[0].dropna().tolist()  # Assuming words are in the first column
    return words

def weighted_levenshtein_word(word1, word2):
    len_word1 = len(word1)
    len_word2 = len(word2)
    
    # Initialize the matrix
    distance_matrix = [[0] * (len_word2 + 1) for _ in range(len_word1 + 1)]
    
    # Initialize base cases
    for i in range(len_word1 + 1):
        distance_matrix[i][0] = i
    for j in range(len_word2 + 1):
        distance_matrix[0][j] = j
    
    # Define weights
    substitution_weight = 1.0
    insertion_weight = 1.0
    deletion_weight = 1.0
    
    # Compute distances
    for i in range(1, len_word1 + 1):
        for j in range(1, len_word2 + 1):
            cost = substitution_weight if word1[i-1] != word2[j-1] else 0
            distance_matrix[i][j] = min(
                distance_matrix[i-1][j] + deletion_weight,
                distance_matrix[i][j-1] + insertion_weight,
                distance_matrix[i-1][j-1] + cost
            )
    return distance_matrix[len_word1][len_word2]

def get_closest_words(word, dictionary, num_suggestions=1):
    """
    Find the closest words in the dictionary to the input word using Levenshtein distance.
    Args:
    - word: The misspelled word.
    - dictionary: A set of correct words.
    - num_suggestions: The number of suggestions to return.
    Returns:
    - A list of tuples (word, distance).
    """
    word_distances = []
    for dict_word in dictionary:
        distance = weighted_levenshtein_word(word, dict_word)
        word_distances.append((dict_word, distance))
    
    # Sort the words by distance
    word_distances.sort(key=lambda x: x[1])
    
    # Return the top suggestions
    return word_distances[:num_suggestions]

def get_closest_words_by_pos(input_word, words_list, num_suggestions=3):
    """
    Get the closest words to the input word from a list of words.

    Args:
    - input_word: The word to find replacements for.
    - words_list: List of words corresponding to the target POS tag.
    - num_suggestions: Number of suggestions to return.

    Returns:
    - A list of tuples: (word, distance)
    """
    if not words_list:
        return []

    # Compute distances
    word_distances = []
    for word in words_list:
        distance = weighted_levenshtein_word(input_word, word)
        word_distances.append((word, distance))

    # Sort by distance
    word_distances.sort(key=lambda x: x[1])
    
    # If fewer words are available than num_suggestions, return all available words
    suggestions = word_distances[:min(len(word_distances), num_suggestions)]

    return suggestions


# Step 6: Correction phase - apply the suggestions to correct the input sentence
from collections import Counter

def apply_pos_corrections(token_suggestions, pos_tags, pos_path, insertion_suggestions):
    """
    This function applies the best corrections based on both Levenshtein distance and frequency.
    It also handles token insertions and applies the best suggestion for each word.

    Args:
    - token_suggestions: A list of dictionaries where each dictionary contains the token's original word,
                         suggestions for corrections, and their respective distances.
    - pos_tags: List of (word, POS) tuples representing the sentence to be corrected.
    - pos_path: Path to the POS tag dictionary files.
    - insertion_suggestions: Dictionary that contains possible insertions for each token index.

    Returns:
    - corrected_sentence: The corrected sentence as a string.
    - word_suggestions: A dictionary mapping original words to their suggested replacements.
    """
    final_sentence = []
    word_suggestions = {}
    pos_tag_dict = {}
    idx = 0
    max_distance = 1.0  # Set the maximum acceptable edit distance

    while idx <= len(token_suggestions):
        # Handle insertions at the current position
        if idx in insertion_suggestions and insertion_suggestions[idx]:
            insert_counter = Counter(insertion_suggestions[idx])
            most_common_insertion_pos, _ = insert_counter.most_common(1)[0]

            # Map the POS tag to an actual word using your POS tag dictionary
            if most_common_insertion_pos not in pos_tag_dict:
                word_list = load_pos_tag_dictionary(most_common_insertion_pos, pos_path)
                pos_tag_dict[most_common_insertion_pos] = word_list
            else:
                word_list = pos_tag_dict[most_common_insertion_pos]

            # Choose a word to insert
            if word_list:
                inserted_word = word_list[0]
            else:
                inserted_word = most_common_insertion_pos

            final_sentence.append(inserted_word)

        if idx < len(token_suggestions):
            token_info = token_suggestions[idx]
            suggestions = token_info["suggestions"]
            distances = token_info["distances"]

            if not suggestions:
                # No suggestions; keep the original word
                word = pos_tags[idx][0]
                final_sentence.append(word)
            else:
                # Combine suggestions and distances into tuples, then filter out duplicates
                suggestions_with_distances = list(zip(suggestions, distances))

                # Use Counter to group similar suggestions and count their frequencies
                suggestion_counter = Counter(suggestions_with_distances)

                # Create a list of unique suggestions sorted by frequency and distance
                unique_suggestions = []
                for suggestion, count in suggestion_counter.items():
                    suggestion_text, suggestion_distance = suggestion
                    unique_suggestions.append((suggestion_text, suggestion_distance, count))

                # Sort by frequency (highest first) and by distance (smallest first)
                unique_suggestions.sort(key=lambda x: (-x[2], x[1]))  # Sort by count then by distance

                # Process the best suggestion
                best_suggestion, best_distance, _ = unique_suggestions[0]

                suggestion_parts = best_suggestion.split("_")
                suggestion_type = suggestion_parts[0]

                if suggestion_type == "KEEP":
                    word = pos_tags[idx][0]
                    final_sentence.append(word)
                elif suggestion_type == "SUBSTITUTE":
                    input_word = pos_tags[idx][0]
                    target_pos = suggestion_parts[2]

                    # Load the dictionary for the target POS tag if not already loaded
                    if target_pos not in pos_tag_dict:
                        word_list = load_pos_tag_dictionary(target_pos, pos_path)
                        pos_tag_dict[target_pos] = word_list
                    else:
                        word_list = pos_tag_dict[target_pos]

                    # Get closest words by POS
                    suggestions_list = get_closest_words_by_pos(input_word, word_list, num_suggestions=3)

                    if suggestions_list:
                        # Apply a distance threshold for word substitution
                        word_max_distance = 2  # Adjust as needed
                        filtered_word_suggestions = [s for s in suggestions_list if s[1] <= word_max_distance]

                        if filtered_word_suggestions:
                            replacement_word = filtered_word_suggestions[0][0]
                            final_sentence.append(replacement_word)
                            word_suggestions[input_word] = [word for word, _ in filtered_word_suggestions]
                        else:
                            # Keep the original word if no suitable replacement
                            final_sentence.append(input_word)
                    else:
                        # If no suggestions found, keep the original word
                        final_sentence.append(input_word)
                elif suggestion_type == "DELETE":
                    idx += 1  # Move to the next token (skip current token)
                    continue  # Skip adding the current token
                else:
                    # Handle any other suggestion types
                    final_sentence.append(pos_tags[idx][0])

        idx += 1  # Move to the next position

    corrected_sentence = " ".join(final_sentence)
    print(f"CORRECTED SENTENCE: {corrected_sentence}")
    return corrected_sentence, word_suggestions

def check_words_in_dictionary(words, directory_path):
    """
    Check if words exist in the dictionary.
    Args:
    - words: List of words to check.
    Returns:
    - List of incorrect words.
    """
    incorrect_words = []
    dictionary = load_dictionary(directory_path)
    
    # Check each word against the dictionary
    for word in words:
        if word.lower() not in dictionary:
            incorrect_words.append(word)
    
    has_incorrect_word = len(incorrect_words) > 0
    logger.debug(f"Incorrect Words: {incorrect_words}")
    
    return incorrect_words, has_incorrect_word

def spell_check_word(word, directory_path):
    """
    Check if the word is spelled correctly and provide a correction if not.
    """
    dictionary = load_dictionary(directory_path)
    word_lower = word.lower()
    
    if word_lower in dictionary:
        # Word is spelled correctly
        return word, None
    else:
        # Word is misspelled; find the closest match
        suggestions = get_closest_words(word_lower, dictionary)
        if suggestions:
            corrected_word = suggestions[0][0]  # Get the best suggestion
            return word, corrected_word
        else:
            # No suggestions found
            return word, None

def spell_check_incorrect_words(text, incorrect_words, directory_path):
    """
    Spell check only the words tagged as incorrect.
    """
    corrected_text = text
    for word in incorrect_words:
        # Get suggestions from your spell checker
        misspelled_word, corrected_word = spell_check_word(word, directory_path)
        if corrected_word:
            # Replace the word with the corrected version
            corrected_text = re.sub(r'\b{}\b'.format(re.escape(word)), corrected_word, corrected_text)
            log_message("info", f"Replaced '{word}' with '{corrected_word}'")
        else:
            log_message("warning", f"No suggestions found for '{word}'")
    return corrected_text

def pantasa_checker(input_sentence, jar_path, model_path, rule_path, directory_path, pos_path):
    """
    Steps: 
    1. /Preprocess: Tokenize and POS Tag 
    2. Dictionary Check: Case-insensitive matching, lemmatization for dic matching
    3. PRE rule: contextualized rules, rule prioritization
    4. Re-check dic: conditional re-check, consistent tokenization 
    5. Spell check: context-aware spell chcking
    6. POST rule: Selective application
    7. Re-tokenize and Re-POS tag: minimized retagging
    8. Generate suggestions: Optimize n-gram generation, advance similarity metrics
    9. Apply pos correction: contet word selection, meaning preservation 
    10. OUTPUT
    """
        
    # Step 1: Preprocess the input text
    log_message("info", "Starting preprocessing")
    tokens = tokenize_sentence(input_sentence)
    pos_tags = pos_tagging(tokens, jar_path, model_path)
    if not pos_tags:
        log_message("error", "POS tagging failed during preprocessing")
        return [], [], {}

    # Step 2: Check if words exist in the dictionary and tag those that don't
    log_message("info", "Checking words against the dictionary")
    words = [word for word, _ in pos_tags]
    incorrect_words, has_incorrect_words  = check_words_in_dictionary(words, directory_path)
    if has_incorrect_words:
        log_message("info", f"The sentence has incorrect words")
    else:
        log_message("info", f"The sentence doesn't have incorrect words")

    # Step 3: Apply pre-defined rules before any modification
    log_message("info", "Applying pre-defined rules (pre)")
    pre_rules_corrected_text = apply_predefined_rules_pre(input_sentence)
    log_message("info", f"Text after pre-defined rules (pre): {pre_rules_corrected_text}")

    # Step 4: Check the dictionary again for any remaining incorrect words
    log_message("info", "Re-checking words against the dictionary after pre-defined rules (pre)")
    
    pre_words = re.findall(r'\w+', pre_rules_corrected_text)
    incorrect_words_after_pre, has_incorrect_words = check_words_in_dictionary(pre_words, directory_path)
    log_message("info", f"Incorrect words after pre-defined rules (pre): {incorrect_words_after_pre}")
    if has_incorrect_words:
        log_message("info", f"The sentence has incorrect words. Incorrect word: {incorrect_words_after_pre}")
    else:
        log_message("info", "The sentence doesn't have incorrect words")

    # Step 5: Spell check the words tagged as incorrect
    log_message("info", "Spell checking incorrect words")
    spell_checked_text = spell_check_incorrect_words(pre_rules_corrected_text, incorrect_words_after_pre, directory_path)
    log_message("info", f"Text after spell checking: {spell_checked_text}")

    # Step 6: Apply pre-defined rules after modifications
    log_message("info", "Applying pre-defined rules (post)")
    post_rules_corrected_text = apply_predefined_rules_post(spell_checked_text)
    log_message("info", f"Text after pre-defined rules (post): {post_rules_corrected_text}")

    # Step 7: Retokenize and re-tag the text after modifications
    log_message("info", "Retokenizing and re-tagging after modifications")
    tokens = tokenize_sentence(post_rules_corrected_text)
    pos_tags = pos_tagging(tokens, jar_path=jar_path, model_path=model_path)
    if not pos_tags:
        log_message("error", "POS tagging failed after modifications")
        return [], [], []
    words = [word for word, _ in pos_tags]
    # Step 8: Generate suggestions using n-gram matching
    log_message("info", "Generating suggestions")
    token_suggestions, insertion_suggestions = generate_suggestions(pos_tags, rule_path)
    log_message("info", f"Token Suggestions: {token_suggestions}")
    log_message("info", f"Insertion Suggestions: {insertion_suggestions}")

    # Step 9: Apply POS corrections
    log_message("info", "Applying POS corrections")
    corrected_sentence, word_suggestions = apply_pos_corrections(token_suggestions, pos_tags, pos_path, insertion_suggestions)

    log_message("info", f"Final Corrected Sentence: {corrected_sentence}")
    # Return the corrected sentence and any suggestions
    return corrected_sentence

if __name__ == "__main__":
    input_text = "magtanim ay hindi biro"
    jar_path = 'rules/Libraries/FSPOST/stanford-postagger.jar'
    model_path = 'rules/Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    
    rule_bank = rule_pattern_bank()         

    corrected_sentence= pantasa_checker(input_text, jar_path, model_path, rule_bank)
    
