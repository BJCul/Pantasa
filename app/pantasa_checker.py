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
from .utils import log_message
from .spell_checker import load_dictionary, spell_check_sentence
from .morphinas_project.lemmatizer_client import initialize_stemmer, lemmatize_multiple_words
from .predefined_rules.rule_main import  apply_predefined_rules, apply_predefined_rules_post, apply_predefined_rules_pre

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
        pattern_frequency =row['Frequency']
        
        # Add the hybrid_ngram and its frequency to the dictionary
        if hybrid_ngram and pattern_frequency:
            rule_pattern_bank[index] = {
                'hybrid_ngram': hybrid_ngram,
                'frequency': pattern_frequency
            }

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
    substitution_weight = 7.0
    insertion_weight = 8.0 
    deletion_weight = 1.2

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

"""
Function Title: Generate Correction Token Tags
Programmers:    Agas, Carlo
                Alcantara, James Patrick
                Figueroa, Jarlson
                Tapia, Rachelle
Where the function fits in the general system designs:
    This function is a part of the Pantasa system program.
    It identifies the necessary operations (keep, substitute, insert, or delete)
    required to transform an input n-gram into a matching pattern n-gram.
Date written and revised: October 10, 2024
Purpose:
    - To compare an input n-gram with a pattern n-gram and generate a sequence
      of correction tags that detail the operations needed for transformation.
    - The correction tags include:
        1. `KEEP`: Indicates that the token is already correct.
        2. `SUBSTITUTE`: Replaces an incorrect token with a correct one.
        3. `INSERT`: Adds a missing token to the input.
        4. `DELETE`: Removes an extra token from the input.
Data Structures, Algorithms, and Control:
    - **Data Structures**:
        - List: Used to store and return the correction tags.
    - **Algorithms**:
        - Two-pointer technique to iterate through input and pattern tokens.
        - Conditional logic to determine the type of operation (keep, substitute,
          insert, or delete).
    - **Control**:
        - A `while` loop to process the tokens in both input and pattern n-grams.
        - Additional `while` loops to handle leftover tokens (extra tokens in input
          or pattern).
"""

# Step 3: Function to generate correction token tags based on the Levenshtein distance
def generate_correction_tags(input_ngram, pattern_ngram):
    input_tokens = input_ngram.split()
    pattern_tokens = pattern_ngram.split()
    
    tags = []
    
    input_idx = 0
    pattern_idx = 0
    
    while input_idx < len(input_tokens) and pattern_idx < len(pattern_tokens):
        input_token = input_tokens[input_idx]
        pattern_token = pattern_tokens[pattern_idx]
        if input_token == pattern_token:
            tags.append(f'KEEP_{input_token}_{pattern_token}')
            input_idx += 1
            pattern_idx += 1
        else:
            # Check for insertion or deletion by looking ahead
            if input_idx < len(input_tokens) - 1 and pattern_token == input_tokens[input_idx + 1]:
                tags.append(f'DELETE_{input_token}')
                input_idx += 1
            elif pattern_idx < len(pattern_tokens) - 1 and input_token == pattern_tokens[pattern_idx + 1]:
                tags.append(f'INSERT_{pattern_token}')
                pattern_idx += 1
            else:
                tags.append(f'SUBSTITUTE_{input_token}_{pattern_token}')
                input_idx += 1
                pattern_idx += 1
    
    # Handle extra tokens in input or pattern (deletion or insertion)
    while input_idx < len(input_tokens):
        tags.append(f'DELETE_{input_tokens[input_idx]}')
        input_idx += 1
    
    while pattern_idx < len(pattern_tokens):
        tags.append(f'INSERT_{pattern_tokens[pattern_idx]}')
        pattern_idx += 1
    
    return tags

# Step 4: Function to generate n-grams of different lengths (from 3 to 7) from the input sentence
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
    
    n_max = min(7, length)  # Set the maximum n-gram size but ensure it's capped by the input length
    
    # Generate n-grams within the dynamic range
    for n in range(n_min, n_max + 1):
        for i in range(len(input_tokens) - n + 1):
            ngram = input_tokens[i:i+n]
            ngrams.append((" ".join(ngram), i))  # Track the starting index
    return ngrams

# Step 5: Suggestion phase - generate suggestions for corrections without applying them
def generate_suggestions(pos_tags, rule_path):

    input_tokens = [pos_tag for word, pos_tag in pos_tags]
    
    # Generate token-level correction tracker
    token_suggestions = [{"token": token, "suggestions": [], "distances": []} for token in input_tokens]
    
    # Track potential inserts (insertion suggestions for each index)
    insert_suggestions = defaultdict(list)
    
    # Generate 3-gram to 7-gram sequences from the input sentence
    input_ngrams_with_index = generate_ngrams(input_tokens)
    
    # Iterate over each n-gram and compare it to the rule pattern bank
    for input_ngram, start_idx in input_ngrams_with_index:
        min_distance = float('inf')
        rule_bank = rule_pattern_bank(rule_path)
        best_match = None
        highest_frequency = 0

        for pattern_id, pattern_data in rule_bank.items():
            # Compare input n-gram with each pattern n-gram from the rule pattern bank
            pattern_ngram = pattern_data.get('hybrid_ngram')
            frequency = pattern_data.get('frequency')  # Correct key for frequency

            if pattern_ngram:
                distance = edit_weighted_levenshtein(input_ngram, pattern_ngram)
                if distance < min_distance or (distance == min_distance and frequency > highest_frequency):
                    min_distance = distance
                    best_match = pattern_ngram
                    highest_frequency = frequency  # Update to use the more frequent pattern
            
        if best_match:
            correction_tags = generate_correction_tags(input_ngram, best_match)
            print(f"CORRECTION TAGS {correction_tags}")
            
            # Populate the token-level correction tracker
            input_ngram_tokens = input_ngram.split()
            token_shift = 0
            for i, tag in enumerate(correction_tags):
                token_idx = start_idx + i + token_shift

                if tag.startswith("INSERT"):
                    # Track the insert suggestion
                    inserted_token = tag.split("_")[1]
                    insert_suggestions[token_idx].append(inserted_token)
                    token_shift = -1

                else:
                    # Ensure token_suggestions don't skip after insertion
                    if token_idx < len(token_suggestions):
                        token_suggestions[token_idx]["suggestions"].append(tag)
                        token_suggestions[token_idx]["distances"].append(min_distance)
    
    # Step 6: Handle inserts based on the majority rule
    for token_idx, inserts in insert_suggestions.items():
        # Count occurrences of each insertion suggestion
        insert_counter = Counter(inserts)
        most_common_insert, insert_count = insert_counter.most_common(1)[0]

        if len(inserts) > 1:
            num_corrections = len(insert_counter)
            threshold = num_corrections / 2
            
            # Only insert the token if the majority of patterns suggest it
            if insert_count > threshold:
                token_suggestions.insert(token_idx, {"token": most_common_insert, "suggestions": [f'INSERT_{most_common_insert}'], "distances": [0.8]})
        else:
            # Skip insertions if there's only one suggestion
            continue
        
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

"""
Function Title: POS-Based Correction Phase
Programmers:    Agas, Carlo
                Alcantara, James Patrick
                Figueroa, Jarlson
                Tapia, Rachelle
Where the function fits in the general system designs:
    This function is part of the text correction module, focusing on applying token-level
    corrections based on suggestions generated during the suggestion phase. It integrates
    grammatical rules and POS dictionaries to correct the input sentence effectively.
Date written and revised: October 10, 2024
Purpose:
    - To process and apply token-level suggestions for corrections, including:
        1. Keeping the token unchanged (`KEEP`).
        2. Substituting the token with a more appropriate one (`SUBSTITUTE`).
        3. Deleting unnecessary tokens (`DELETE`).
        4. Inserting missing tokens based on POS tag rules (`INSERT`).
    - To utilize part-of-speech (POS) dictionaries for substitution and insertion operations.
    - To construct a grammatically corrected output sentence.
Data Structures, Algorithms, and Control:
    - **Data Structures**:
        - `final_sentence`: List to store the corrected sentence tokens.
        - `word_suggestions`: Dictionary to track suggestions for each word.
        - `pos_tag_dict`: Dictionary to cache loaded POS tag word lists.
        - `inserted_tokens`: Set to track inserted tokens and prevent duplication.
    - **Algorithms**:
        - Iterative approach to process `token_suggestions` and apply corrections.
        - Conditional logic for handling different suggestion types (`KEEP`, `SUBSTITUTE`,
          `DELETE`, `INSERT`).
        - Leverages external helper functions for POS-related operations:
          - `load_pos_tag_dictionary`: Loads POS-tagged word lists from a file.
          - `get_closest_words_by_pos`: Retrieves the closest matching words for a target POS.
    - **Control**:
        - Iterates over `token_suggestions` using a loop and updates indices as suggestions are applied.
        - Uses counters (`idx`) to sync corrections with `pos_tags`.
"""

# Step 6: Correction phase - apply the suggestions to correct the input sentence
def apply_pos_corrections(token_suggestions, pos_tags, pos_path):
    final_sentence = []
    word_suggestions = {}  # To keep track of suggestions for each word
    pos_tag_dict = {}      # Cache for loaded POS tag dictionaries
    idx = 0                # Index for iterating through pos_tags and token_suggestions
    inserted_tokens = set()

    # Iterate through the token_suggestions and apply the corrections
    for token_info in token_suggestions:
        suggestions = token_info["suggestions"]
        distances = token_info["distances"]

        if not suggestions:
            # No suggestions; keep the original word
            word = pos_tags[idx][0]
            final_sentence.append(word)
            idx += 1
            continue

        # Count the frequency of each exact suggestion
        suggestion_count = Counter(suggestions)
        print(f"COUNTER {suggestion_count}")

        if suggestion_count:
            # Step 2: Find the most frequent exact suggestion(s)
            most_frequent_suggestion = suggestion_count.most_common(1)[0][0]  # Get the most frequent exact suggestion

            # Extract suggestion type, current POS, and target POS correctly
            suggestion_parts = most_frequent_suggestion.split("_")
            suggestion_type = suggestion_parts[0]
            current_pos = "_".join(suggestion_parts[1:-1])  # Join all parts except the last one as current POS
            target_pos = suggestion_parts[-1]  # Last part is the target POS

            print(f"SUGGESTION TYPE: {suggestion_type}, CURRENT POS: {current_pos}, TARGET POS: {target_pos}")

            if suggestion_type == "KEEP":
                # Append the original word
                word = pos_tags[idx][0]
                final_sentence.append(word)
                idx += 1  # Increment idx to move to the next word

            elif suggestion_type == "SUBSTITUTE":
                # Extract input word and target POS tag
                input_word = pos_tags[idx][0]

                # Load the dictionary for the target POS tag if not already loaded
                if target_pos not in pos_tag_dict:
                    word_list = load_pos_tag_dictionary(target_pos, pos_path)
                    pos_tag_dict[target_pos] = word_list
                else:
                    word_list = pos_tag_dict[target_pos]

                # Get closest words by POS
                suggestions_list = get_closest_words_by_pos(input_word, word_list, num_suggestions=1)

                if suggestions_list:
                    # Pick the best suggestion (smallest distance)
                    replacement_word = suggestions_list[0][0]
                    final_sentence.append(replacement_word)
                    print(f"Replaced '{input_word}' with '{replacement_word}' for target POS '{target_pos}'")

                    # Store suggestions for the word
                    word_suggestions[input_word] = [word for word, dist in suggestions_list]
                else:
                    # If no suggestions found, keep the original word
                    final_sentence.append(input_word)

                idx += 1  # Increment idx to move to the next word

            elif suggestion_type == "DELETE":
                # Skip the word (do not append it)
                idx += 1  # Move to the next word

            elif suggestion_type == "INSERT":
                target_pos = target_pos  # The extracted target POS tag for insertion

                # Load the dictionary for the target POS tag if not already loaded
                if target_pos not in pos_tag_dict:
                    word_list = load_pos_tag_dictionary(target_pos, pos_path)
                    pos_tag_dict[target_pos] = word_list
                else:
                    word_list = pos_tag_dict[target_pos]

                # Choose the most frequent word from the POS tag dictionary for insertion
                if word_list:
                    inserted_token = word_list[0]  # Assuming the list is sorted by frequency
                else:
                    inserted_token = "[UNK]"  # Fallback if no words are found

                # Only insert if not already inserted at this position
                if inserted_token not in inserted_tokens:
                    final_sentence.append(inserted_token)
                    inserted_tokens.add(inserted_token)  # Track the insertion to avoid duplication
                else:
                    continue  # Skip if already inserted

            else:
                # Handle any other suggestion types (fallback)
                word = pos_tags[idx][0]
                final_sentence.append(word)
                idx += 1  # Increment idx to move to the next word
        else:
            # Default case: Append the word if no valid suggestions
            word = pos_tags[idx][0]
            final_sentence.append(word)
            idx += 1

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
    
    # # Step 1: Check if words exist in the dictionary
    # log_message("info", "Checking words against the dictionary")
    # tokens = tokenize_sentence(input_sentence)
    # words = [word for word in tokens]
    # incorrect_words, has_incorrect_words  = check_words_in_dictionary(words, directory_path)
    
    # if has_incorrect_words:
    #     # There are misspelled words, proceed with spell checking pipeline

    #     # Step 2: Apply pre-defined rules before any modification
    #     log_message("info", "Applying pre-defined rules (pre) to resolve misspelled words")
    #     pre_rules_corrected_text = apply_predefined_rules_pre(input_sentence)
    #     log_message("info", f"Text after pre-defined rules (pre): {pre_rules_corrected_text}")

    #     # Step 3: Re-check the dictionary after applying pre-rules
    #     pre_words = re.findall(r'\w+', pre_rules_corrected_text)
    #     incorrect_words_after_pre, has_incorrect_words = check_words_in_dictionary(pre_words, directory_path)
    #     log_message("info", f"Incorrect words after pre-defined rules (pre): {incorrect_words_after_pre}")
        
    #     if has_incorrect_words:
    #         # Step 4: Spell check the words tagged as incorrect
    #         log_message("info", "Spell checking remaining incorrect words")
    #         spell_checked_text, spell_suggestions, final_incorrect_words = spell_check_incorrect_words(
    #             pre_rules_corrected_text, incorrect_words_after_pre, directory_path
    #         )
    #         log_message("info", f"Text after spell checking: {spell_checked_text}")
    #         # Output misspelled words and suggestions
    #         return spell_checked_text, spell_suggestions, final_incorrect_words
    #     else:
    #         # If pre-rules resolved all misspelled words, return with no further spell checking needed
    #         log_message("info", "Pre-rules resolved all misspelled words")
    #         return pre_rules_corrected_text, {}, []

    # else:
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



    
