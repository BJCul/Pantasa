"""
Program Title: Pantasa: Iterative rule-based using Hybrid ngram
Programmers:    Agas, Carlo
                Alcantara, James Patrick
                Figueroa, Jarlson
                Tapia, Rachelle
            
Where the program fits in the general system designs:
    This program is part of the backend system of Pantasa for grammar error detection and correction using hybrid n-grams 
    rule based system and the iterative processs. This is where the input sentence go after the user input a sentence.
Date written and revised: October 10, 2024
Purpose:
    - To correct input text using a hybrid n-gram rule bank.
    - To suggest and apply corrections for text tokens based on pattern
      matching, Levenshtein distance, and rule frequency.
Data Structures, Algorithms, and Control:
    - Data Structures:
        - Dictionary: Used for the rule pattern bank and token suggestions.
        - List: Stores n-grams and their indices.
        - Defaultdict & Counter: Track insert suggestions and their counts.
    - Algorithms:
        - Weighted Levenshtein distance for measuring similarity.
        - N-gram generation for varying lengths.
        - Correction tag generation for substitutions, deletions, and insertions.
        - Majority rule for insertions based on suggestions.
        - Iterative process
    - Control:
        - Iterative loops for n-gram generation and suggestion evaluation.
        - Conditional logic to apply corrections based on tag type and frequency.
"""
import pandas as pd
import re
from collections import Counter, defaultdict
import tempfile
import subprocess
import os
import logging
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

    # Create a dictionary to store the Rule Pattern Bank (Hybrid N-Grams + Predefined Rules)
    rule_pattern_bank = {}

    # Store the hybrid n-grams from the CSV file into the rule pattern bank
    for index, row in hybrid_ngrams_df.iterrows():
        hybrid_ngram = row['Hybrid_N-Gram']
        pattern_frequency = row['Frequency']
        
        # Add the hybrid_ngram and its frequency to the dictionary
        if hybrid_ngram and pattern_frequency:
            rule_pattern_bank[index] = {
                'hybrid_ngram': hybrid_ngram,
                'frequency': pattern_frequency
            }

    return rule_pattern_bank

def generalize_pos_tag(pos_tag):
    """
    Generalizes a POS tag by mapping it to its base form.
    Handles composite POS tags by generalizing each component.

    Args:
    - pos_tag: The POS tag to generalize.

    Returns:
    - A set of generalized POS tags.
    """
    # Split composite POS tags
    pos_tags = pos_tag.split('_')
    generalized_tags = set()
    for tag in pos_tags:
        # Generalize each tag individually
        if tag.startswith('NN'):
            generalized_tags.add('NN.*')
        elif tag.startswith('VB'):
            generalized_tags.add('VB.*')
        elif tag.startswith('CC'):
            generalized_tags.add('CC.*')
        elif tag.startswith('DT'):
            generalized_tags.add('DT.*')
        elif tag.startswith('PR'):
            generalized_tags.add('PR.*')
        elif tag.startswith('JJ'):
            generalized_tags.add('JJ.*')
        elif tag.startswith('RB'):
            generalized_tags.add('RB.*')
        elif tag == 'LM':
            generalized_tags.add('LM')
        elif tag == 'TS':
            generalized_tags.add('TS')
        elif tag.startswith('CD'):
            generalized_tags.add('CD.*')
        elif tag.startswith('PM'):
            generalized_tags.add('PM.*')
        else:
            generalized_tags.add(tag)  # Return as is if no mapping is defined
    return generalized_tags

def edit_weighted_levenshtein(input_ngram_tokens, pattern_ngram):
    pattern_tokens = pattern_ngram.strip().split()
    
    len_input = len(input_ngram_tokens)
    len_pattern = len(pattern_tokens)
    
    # Create a matrix to store the edit distances
    distance_matrix = [[0] * (len_pattern + 1) for _ in range(len_input + 1)]

    # Define weights for substitution, insertion, and deletion
    substitution_weight = 10.0  # Increased to avoid overcorrection
    insertion_weight = 8.0 
    deletion_weight = 1.2
    
    # Initialize base case values (costs of insertions and deletions)
    for i in range(len_input + 1):
        distance_matrix[i][0] = i * deletion_weight
    for j in range(len_pattern + 1):
        distance_matrix[0][j] = j * insertion_weight
    
    # Compute the distances
    for i in range(1, len_input + 1):
        for j in range(1, len_pattern + 1):
            input_word, input_pos = input_ngram_tokens[i - 1]
            pattern_token = pattern_tokens[j - 1]
            
            # Determine comparison basis
            if pattern_token.islower() or pattern_token in ["ay", "ng", "na"]:
                # Specific word matching
                if input_word.lower() == pattern_token:
                    cost = 0
                else:
                    cost = substitution_weight
            else:
                # POS tag matching with possible wildcards
                generalized_input_pos_set = generalize_pos_tag(input_pos)
                try:
                    if any(re.fullmatch(pattern_token, gen_pos) for gen_pos in generalized_input_pos_set):
                        cost = 0
                    else:
                        cost = substitution_weight
                except re.error as e:
                    print(f"Regex error: {e} with pattern_token: '{pattern_token}' and input_pos: '{generalized_input_pos_set}'")
                    cost = substitution_weight
    
            distance_matrix[i][j] = min(
                distance_matrix[i - 1][j] + deletion_weight,    # Deletion
                distance_matrix[i][j - 1] + insertion_weight,  # Insertion
                distance_matrix[i - 1][j - 1] + cost           # Substitution
            )
    
    return distance_matrix[len_input][len_pattern]

def generate_correction_tags(input_ngram_tokens, pattern_ngram):
    """
    Generates correction tags that detail the operations needed to transform
    an input n-gram into the matching pattern n-gram.

    Args:
    - input_ngram_tokens: List of tuples (word, pos_tag) for the input n-gram.
    - pattern_ngram: String representing the pattern n-gram.

    Returns:
    - tags: List of correction tags.
    """
    pattern_tokens = pattern_ngram.strip().split()
    
    tags = []
    input_idx = 0
    pattern_idx = 0
    
    while input_idx < len(input_ngram_tokens) and pattern_idx < len(pattern_tokens):
        input_word, input_pos = input_ngram_tokens[input_idx]
        pattern_token = pattern_tokens[pattern_idx].strip('_')  # Strip any underscores
        
        if pattern_token.islower() or pattern_token in ["ay", "ng", "na"]:
            # Specific word matching
            if input_word.lower() == pattern_token:
                tags.append(f'KEEP_{input_word}_{input_pos}')
            else:
                tags.append(f'SUBSTITUTE_WORD_{input_word}_{pattern_token}')
            input_idx += 1
            pattern_idx += 1
        else:
            # POS tag matching with possible wildcards
            generalized_input_pos_set = generalize_pos_tag(input_pos)
            if any(re.fullmatch(pattern_token, gen_pos) for gen_pos in generalized_input_pos_set):
                tags.append(f'KEEP_{input_word}_{input_pos}')
            else:
                tags.append(f'SUBSTITUTE_POS_{input_pos}_{pattern_token}')
            input_idx += 1
            pattern_idx += 1
    
    # Handle extra tokens in input (deletion)
    while input_idx < len(input_ngram_tokens):
        input_word, input_pos = input_ngram_tokens[input_idx]
        tags.append(f'DELETE_{input_word}')
        input_idx += 1
    
    # Handle extra tokens in pattern (insertion)
    while pattern_idx < len(pattern_tokens):
        pattern_token = pattern_tokens[pattern_idx]
        tags.append(f'INSERT_{pattern_token}')
        pattern_idx += 1
    
    return tags

def generate_ngrams(input_tokens):
    ngrams = []
    length = len(input_tokens)

    # Determine minimum n-gram size based on input length
    if length < 4:
        n_min = 3
    elif length < 5:
        n_min = 4
    elif length < 6:
        n_min = 5
    elif length > 7:
        n_min = 6
    else:
        n_min = 5

    n_max = min(7, length)

    # Generate n-grams within the dynamic range
    for n in range(n_min, n_max + 1):
        for i in range(len(input_tokens) - n + 1):
            ngram_tokens = input_tokens[i:i+n]
            ngrams.append((ngram_tokens, i))  # ngram_tokens is a list of (word, pos_tag) tuples
    return ngrams

def generate_suggestions(pos_tags, rule_path):
    """
    Generates suggestions for corrections based on matching input POS-tagged tokens
    with patterns from the rule pattern bank.

    Args:
    - pos_tags: List of tuples (word, pos_tag) representing the input tokens.
    - rule_path: Path to the rule pattern bank CSV file.

    Returns:
    - token_suggestions: List of dictionaries with suggestions for each token.
    """
    # input_tokens is a list of (word, pos_tag) tuples
    input_tokens = pos_tags

    # Initialize token suggestions
    token_suggestions = [{"token": token[0], "suggestions": [], "distances": []} for token in input_tokens]

    # Initialize insert suggestions
    insert_suggestions = defaultdict(list)

    # Generate n-grams from input tokens (as lists of (word, pos_tag) tuples)
    input_ngrams_with_index = generate_ngrams(input_tokens)

    # Load the rule pattern bank
    rule_bank = rule_pattern_bank(rule_path)

    # Set a threshold for acceptable edit distances
    threshold_distance = 5.0  # Adjust as needed

    # Iterate over each input n-gram
    for input_ngram_tokens, start_idx in input_ngrams_with_index:
        min_distance = float('inf')
        best_match = None
        highest_frequency = 0

        # Iterate over each pattern in the rule bank
        for pattern_id, pattern_data in rule_bank.items():
            pattern_ngram = pattern_data.get('hybrid_ngram')
            frequency = pattern_data.get('frequency')

            if pattern_ngram:
                # Compute distance between input n-gram and pattern n-gram
                distance = edit_weighted_levenshtein(input_ngram_tokens, pattern_ngram)
                if distance < min_distance or (distance == min_distance and frequency > highest_frequency):
                    min_distance = distance
                    best_match = pattern_ngram
                    highest_frequency = frequency

        if best_match and min_distance <= threshold_distance:
            # Generate correction tags
            correction_tags = generate_correction_tags(input_ngram_tokens, best_match)

            # Process correction tags
            input_idx = 0  # Index in input_ngram_tokens
            token_idx = start_idx  # Index in token_suggestions

            for tag in correction_tags:
                if tag.startswith("INSERT"):
                    # Handle insertion
                    inserted_token = tag.split("_")[1]
                    insert_suggestions[token_idx].append(inserted_token)
                    # Do not increment token_idx
                elif tag.startswith("DELETE"):
                    # Handle deletion
                    if token_idx < len(token_suggestions):
                        token_suggestions[token_idx]["suggestions"].append(tag)
                        token_suggestions[token_idx]["distances"].append(min_distance)
                    # Move to next token
                    token_idx += 1
                    input_idx += 1
                else:
                    # Handle KEEP and SUBSTITUTE
                    if token_idx < len(token_suggestions):
                        token_suggestions[token_idx]["suggestions"].append(tag)
                        token_suggestions[token_idx]["distances"].append(min_distance)
                    # Move to next token
                    token_idx += 1
                    input_idx += 1

    # Handle insertions
    # For each position where an insertion is suggested, decide whether to insert and what to insert
    # Here, we will insert the most common suggestion at each position

    # Collect all insertion positions and sort them
    insertion_positions = sorted(insert_suggestions.keys())

    # Initialize an offset to adjust indices after insertions
    offset = 0

    for pos in insertion_positions:
        adjusted_pos = pos + offset
        inserts = insert_suggestions[pos]
        # Get the most common insertion suggestion
        insert_counter = Counter(inserts)
        most_common_insert, _ = insert_counter.most_common(1)[0]
        # Insert into token_suggestions at adjusted_pos
        token_suggestions.insert(adjusted_pos, {
            "token": most_common_insert,
            "suggestions": [f'INSERT_{most_common_insert}'],
            "distances": [0]  # Distance can be set to 0 for insertions
        })
        # Update offset
        offset += 1

    return token_suggestions

def load_pos_tag_dictionary(pos_tag, pos_path):
    """
    Load the POS tag dictionary based on the specific or generalized POS tag.

    Args:
    - pos_tag (str): The POS tag to search for.
    - pos_path (str): The base path where the CSV files are stored.

    Returns:
    - words (list): List of words from the corresponding POS tag CSV files.
    """
    if not pos_tag:
        raise ValueError("POS tag is empty. Cannot load POS tag dictionary.")

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

        # List all files in the directory and find files starting with the generalized POS tag
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

def get_closest_words(word, dictionary, num_suggestions=5):
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

def get_closest_words_by_pos(input_word, words_list, num_suggestions=1):
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

def apply_pos_corrections(token_suggestions, pos_tags, pos_path):
    """
    Applies corrections to the input sentence based on the token suggestions.

    Args:
    - token_suggestions: List of dictionaries containing suggestions for each token.
    - pos_tags: List of tuples (word, pos_tag) representing the original tokens.
    - pos_path: Path to the directory containing POS tag dictionaries.

    Returns:
    - corrected_sentence: The corrected sentence as a string.
    """
    final_sentence = []
    word_suggestions = {}  # To keep track of suggestions for each word
    pos_tag_dict = {}      # Cache for loaded POS tag word lists
    idx = 0                # Index for iterating through pos_tags
    token_idx = 0          # Index for iterating through token_suggestions

    while token_idx < len(token_suggestions):
        token_info = token_suggestions[token_idx]
        suggestions = token_info["suggestions"]
        distances = token_info["distances"]

        if not suggestions:
            # No suggestions; keep the original word
            if idx < len(pos_tags):
                word = pos_tags[idx][0]
                final_sentence.append(word)
                idx += 1
            else:
                # This might be an inserted token without suggestions
                final_sentence.append(token_info["token"])
            token_idx += 1
            continue

        # Count the frequency of each exact suggestion
        suggestion_count = Counter(suggestions)

        if suggestion_count:
            # Get the most frequent suggestion
            most_frequent_suggestion = suggestion_count.most_common(1)[0][0]
            suggestion_parts = most_frequent_suggestion.split("_", 2)  # Split only on the first two underscores
            suggestion_type = suggestion_parts[0]

            if suggestion_type == "KEEP":
                if idx < len(pos_tags):
                    word = pos_tags[idx][0]
                    final_sentence.append(word)
                    idx += 1
                token_idx += 1

            elif suggestion_type == "SUBSTITUTE":
                substitution_kind = suggestion_parts[1]
                remaining_parts = suggestion_parts[2]

                if substitution_kind == "WORD":
                    # Word substitution
                    # remaining_parts: input_word_pattern_token
                    input_word, pattern_token = remaining_parts.rsplit('_', 1)
                    final_sentence.append(pattern_token)
                    idx += 1
                elif substitution_kind == "POS":
                    # POS substitution
                    # remaining_parts: input_pos_pattern_token
                    input_pos, pattern_token = remaining_parts.rsplit('_', 1)
                    input_word = pos_tags[idx][0]

                    # Strip any trailing underscores from pattern_token
                    pattern_token = pattern_token.strip('_')

                    # Validate pattern_token
                    if not pattern_token:
                        print(f"Warning: Empty pattern_token in suggestion '{most_frequent_suggestion}'")
                        token_idx += 1
                        continue

                    # Load or retrieve the POS tag dictionary
                    if pattern_token not in pos_tag_dict:
                        word_list = load_pos_tag_dictionary(pattern_token, pos_path)
                        pos_tag_dict[pattern_token] = word_list
                    else:
                        word_list = pos_tag_dict[pattern_token]

                    # Get closest words by POS
                    suggestions_list = get_closest_words_by_pos(input_word, word_list, num_suggestions=1)

                    if suggestions_list:
                        replacement_word = suggestions_list[0][0]
                        final_sentence.append(replacement_word)
                        word_suggestions[input_word] = [word for word, dist in suggestions_list]
                    else:
                        # Prioritize 'KEEP' if no suitable substitution found
                        final_sentence.append(input_word)
                    idx += 1
                else:
                    # Handle other substitution types if necessary
                    if idx < len(pos_tags):
                        word = pos_tags[idx][0]
                        final_sentence.append(word)
                        idx += 1
                token_idx += 1

            elif suggestion_type == "DELETE":
                # Skip the word
                idx += 1
                token_idx += 1

            elif suggestion_type == "INSERT":
                pattern_token = suggestion_parts[1]

                # Check if the pattern_token is a POS tag or a specific word
                if pattern_token.islower() or pattern_token in ["ay", "ng", "na"]:
                    # Insert the specific word
                    final_sentence.append(pattern_token)
                else:
                    # It's a POS tag; load the corresponding dictionary
                    target_pos = pattern_token
                    if target_pos not in pos_tag_dict:
                        word_list = load_pos_tag_dictionary(target_pos, pos_path)
                        pos_tag_dict[target_pos] = word_list
                    else:
                        word_list = pos_tag_dict[target_pos]
                    inserted_token = word_list[0] if word_list else "[UNK]"
                    final_sentence.append(inserted_token)
                token_idx += 1

            else:
                # Fallback: Append the original word
                if idx < len(pos_tags):
                    word = pos_tags[idx][0]
                    final_sentence.append(word)
                    idx += 1
                token_idx += 1
        else:
            # No valid suggestions; prioritize 'KEEP'
            if idx < len(pos_tags):
                word = pos_tags[idx][0]
                final_sentence.append(word)
                idx += 1
            token_idx += 1

    corrected_sentence = " ".join(final_sentence)
    return corrected_sentence

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

def spell_check_word(word, directory_path, num_suggestions=5):
    """
    Check if the word is spelled correctly and provide up to `num_suggestions` corrections if not.
    """
    dictionary = load_dictionary(directory_path)
    word_lower = word.lower()
    
    if word_lower in dictionary:
        # Word is spelled correctly
        return word, None
    else:
        # Word is misspelled; find the closest matches
        suggestions = get_closest_words(word_lower, dictionary, num_suggestions=num_suggestions)
        if suggestions:
            # Return the word and all closest suggestions
            return word, [suggestion[0] for suggestion in suggestions]  # Get the top suggestions
        else:
            # No suggestions found
            return word, None

def spell_check_incorrect_words(text, incorrect_words, directory_path, num_suggestions=5):
    """
    Spell check only the words tagged as incorrect and provide multiple suggestions.
    Replaces incorrect words with the 3rd suggestion if available.
    """
    corrected_text = text
    suggestions_dict = {}  # Store suggestions for each incorrect word

    # Loop through each incorrect word
    for word in incorrect_words:
        # Get suggestions from your spell checker
        misspelled_word, suggestions = spell_check_word(word, directory_path, num_suggestions)
        if suggestions:
            # Log the suggestions and store them
            log_message("info", f"Suggestions for '{word}': {suggestions}")
            suggestions_dict[word] = suggestions

            # Replace the word with the 3rd suggestion if it exists
            if len(suggestions) >= 3:
                corrected_word = suggestions[2]  # Get the 3rd suggestion (index 2)
            else:
                corrected_word = suggestions[0]  # If less than 3 suggestions, use the first one
            
            # Replace the word in the text
            corrected_text = re.sub(r'\b{}\b'.format(re.escape(word)), corrected_word, corrected_text)
            log_message("info", f"Replaced '{word}' with '{corrected_word}'")
        else:
            log_message("warning", f"No suggestions found for '{word}'")
            suggestions_dict[word] = []  # If no suggestions, leave an empty list

    # Return the corrected text, suggestions, and incorrect words
    return corrected_text, suggestions_dict, incorrect_words

def pantasa_checker(input_sentence, jar_path, model_path, rule_path, directory_path, pos_path):
    """
    Step 1: Check for misspelled words using dictionary
    Step 2: Apply pre-defined rules for possible word corrections
    Step 3: Re-check dictionary after pre-rules
    Step 4: If still misspelled words, suggest spell corrections
    Step 5: Else, proceed with grammar checking
    """
    
    # Step 1: Check if words exist in the dictionary
    log_message("info", "Checking words against the dictionary")
    tokens = tokenize_sentence(input_sentence)
    words = [word for word in tokens]
    incorrect_words, has_incorrect_words  = check_words_in_dictionary(words, directory_path)
    
    if has_incorrect_words:
        # There are misspelled words, proceed with spell checking pipeline

        # Step 2: Apply pre-defined rules before any modification
        log_message("info", "Applying pre-defined rules (pre) to resolve misspelled words")
        pre_rules_corrected_text = apply_predefined_rules_pre(input_sentence)
        log_message("info", f"Text after pre-defined rules (pre): {pre_rules_corrected_text}")

        # Step 3: Re-check the dictionary after applying pre-rules
        pre_words = re.findall(r'\w+', pre_rules_corrected_text)
        incorrect_words_after_pre, has_incorrect_words = check_words_in_dictionary(pre_words, directory_path)
        log_message("info", f"Incorrect words after pre-defined rules (pre): {incorrect_words_after_pre}")
        
        if has_incorrect_words:
            # Step 4: Spell check the words tagged as incorrect
            log_message("info", "Spell checking remaining incorrect words")
            spell_checked_text, spell_suggestions, final_incorrect_words = spell_check_incorrect_words(
                pre_rules_corrected_text, incorrect_words_after_pre, directory_path
            )
            log_message("info", f"Text after spell checking: {spell_checked_text}")
            # Output misspelled words and suggestions
            return spell_checked_text, spell_suggestions, final_incorrect_words
        else:
            # If pre-rules resolved all misspelled words, return with no further spell checking needed
            log_message("info", "Pre-rules resolved all misspelled words")
            return pre_rules_corrected_text, {}, []

    else:
        # Proceed with grammar checking (no misspelled words found)
        log_message("info", "No misspelled words found, proceeding with grammar checking")

        # Step 6: Apply post-defined rules after POS tagging
        log_message("info", "Applying post-defined rules (post)")
        post_rules_corrected_text = apply_predefined_rules_post(input_sentence)
        log_message("info", f"Text after post-defined rules (post): {post_rules_corrected_text}")

        # Step 7: Re-tokenize and re-POS tag after post rules
        log_message("info", "Retokenizing and re-POS tagging after post modifications")
        tokens = tokenize_sentence(post_rules_corrected_text)
        pos_tags = pos_tagging(tokens, jar_path, model_path)
        if not pos_tags:
            log_message("error", "POS tagging failed after modifications")
            return [], [], []

        # Step 8: Generate suggestions using n-gram matching
        log_message("info", "Generating suggestions using n-gram matching")
        token_suggestions = generate_suggestions(pos_tags, rule_path)
        log_message("info", f"Token Suggestions: {token_suggestions}")

        # Step 9: Apply POS corrections
        log_message("info", "Applying POS corrections")
        corrected_sentence = apply_pos_corrections(token_suggestions, pos_tags, pos_path)

        log_message("info", f"Final Corrected Sentence: {corrected_sentence}")
        # Return the corrected sentence and token suggestions
        return corrected_sentence, token_suggestions, []

def test():
    input_sentence = "kaya ng ng tao na"
    jar_path = 'rules/Libraries/FSPOST/stanford-postagger.jar'
    model_path = 'rules/Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    rule_path = 'data/processed/hngrams_test.csv'
    directory_path = 'data/raw/dictionary.csv'
    pos_path = 'data/processed/pos_dic'

    # Step 1: Tokenize the input sentence
    tokens = tokenize_sentence(input_sentence)
    print("Tokens:", tokens)

    # Step 2: POS tag the tokens
    pos_tags = pos_tagging(tokens, jar_path=jar_path, model_path=model_path)
    print("POS Tags:", pos_tags)

    # Step 3: Load the rule pattern bank
    rule_bank = rule_pattern_bank(rule_path)
    print("Rule Pattern Bank Loaded.")

    # Step 4: Generate suggestions using n-gram matching
    token_suggestions = generate_suggestions(pos_tags, rule_path)
    print("Token Suggestions:")
    for idx, suggestion in enumerate(token_suggestions):
        print(f"Token {idx}: {suggestion}")

    # Step 5: Apply POS corrections
    corrected_sentence = apply_pos_corrections(token_suggestions, pos_tags, pos_path)
    print("Corrected Sentence:", corrected_sentence)

test()