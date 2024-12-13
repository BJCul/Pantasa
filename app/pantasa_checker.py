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

import Levenshtein
import pandas as pd
import re
from collections import Counter, defaultdict
import tempfile
import pickle
import subprocess
import os
import logging
from .utils import log_message
from .spell_checker import load_dictionary, spell_check_sentence
from .morphinas_project.lemmatizer_client import initialize_stemmer, lemmatize_multiple_words
from .predefined_rules.rule_main import apply_predefined_rules, apply_predefined_rules_post, apply_predefined_rules_pre

# Initialize the Morphinas Stemmer
stemmer = initialize_stemmer()

logger = logging.getLogger(__name__)

jar = 'rules/Libraries/FSPOST/stanford-postagger.jar'
model = 'rules/Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'

MAX_LEVENSHTEIN_DISTANCE = 2

def tokenize_sentence(sentence):
    """
    Tokenizes an input sentence into words and punctuation using regex.
    This updated version allows hyphenated words to remain as single tokens.
    
    Args:
    - sentence: The input sentence as a string.
    
    Returns:
    - A list of tokens.
    """
    # Modified token pattern to keep hyphenated words together
    token_pattern = re.compile(r'\w+(?:-\w+)*|[^\w\s]')
    tokens = token_pattern.findall(sentence)
    logger.debug(f"Tokens: {tokens}")
    return tokens

def pos_tagging(tokens, jar_path=jar, model_path=model):
    """
    Tags tokens using the FSPOST Tagger via subprocess.
    """
    java_tokens = []
    tagged_tokens = []

    for token in tokens:
        if isinstance(token, tuple):
            token = token[0]  # Extract the actual word if tuple
        java_tokens.append(token)

    if java_tokens:
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

            os.unlink(temp_file_path)

            if process.returncode != 0:
                raise Exception(f"POS tagging process failed: {error.decode('utf-8')}")

            tagged_output = output.decode('utf-8').strip().split()
            java_tagged_tokens = [tuple(tag.split('|')) for tag in tagged_output if '|' in tag]

            tagged_tokens.extend(java_tagged_tokens)
            logger.debug(f"POS Tagged Tokens: {tagged_tokens}")

        except Exception as e:
            log_message("error", f"Error during POS tagging: {e}")
            return []

    return tagged_tokens

def preprocess_text(text_input, jar_path, model_path):
    """
    Preprocess the input text by tokenizing, POS tagging, lemmatizing, and checking spelling.
    """
    mispelled_words, checked_sentence = spell_check_sentence(text_input)
    tokens = tokenize_sentence(checked_sentence)
    tagged_tokens = pos_tagging(tokens, jar_path=jar_path, model_path=model_path)

    if not tagged_tokens:
        log_message("error", "Tagged tokens are empty.")
        return []

    words = [word for word, pos in tagged_tokens]
    gateway, lemmatizer = stemmer
    lemmatized_words = lemmatize_multiple_words(words, gateway, lemmatizer)
    log_message("info", f"Lemmatized Words: {lemmatized_words}")

    preprocessed_output = (tokens, lemmatized_words, tagged_tokens, checked_sentence, mispelled_words)
    log_message("info", f"Preprocessed Output: {preprocessed_output}")
    return [preprocessed_output]

def rule_pattern_bank(rule_path, cache_path="rule_pattern_bank.pkl"):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    hybrid_ngrams_df = pd.read_csv(rule_path, low_memory=False)
    rule_pattern_bank = {}

    for index, row in hybrid_ngrams_df.iterrows():
        hybrid_ngram = row['DetailedPOS_N-Gram']
        pattern_frequency = row['Frequency']
        if hybrid_ngram and pattern_frequency:
            rule_pattern_bank[index] = {
                'hybrid_ngram': hybrid_ngram,
                'frequency': pattern_frequency
            }

    with open(cache_path, "wb") as f:
        pickle.dump(rule_pattern_bank, f)

    return rule_pattern_bank

def edit_weighted_levenshtein(input_ngram, pattern_ngram):
    input_tokens = input_ngram.strip().split()
    pattern_tokens = pattern_ngram.strip().split()
    
    len_input = len(input_tokens)
    len_pattern = len(pattern_tokens)
    distance_matrix = [[0] * (len_pattern + 1) for _ in range(len_input + 1)]

    for i in range(len_input + 1):
        distance_matrix[i][0] = i
    for j in range(len_pattern + 1):
        distance_matrix[0][j] = j

    substitution_weight = 0.7
    insertion_weight = 0.8
    deletion_weight = 1.2

    for i in range(1, len_input + 1):
        for j in range(1, len_pattern + 1):
            input_token = input_tokens[i - 1]
            pattern_token = pattern_tokens[j - 1]
            try:
                if re.match(pattern_token, input_token): 
                    cost = 0
                else:
                    cost = substitution_weight
            except re.error as e:
                print(f"Regex error: {e} with pattern_token: '{pattern_token}' and input_token: '{input_token}'")

            distance_matrix[i][j] = min(
                distance_matrix[i - 1][j] + deletion_weight,
                distance_matrix[i][j - 1] + insertion_weight,
                distance_matrix[i - 1][j - 1] + cost
            )
    
    return distance_matrix[len_input][len_pattern]

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

    while input_idx < len(input_tokens):
        tags.append(f'DELETE_{input_tokens[input_idx]}')
        input_idx += 1
    
    while pattern_idx < len(pattern_tokens):
        tags.append(f'INSERT_{pattern_tokens[pattern_idx]}')
        pattern_idx += 1
    
    return tags

def generate_ngrams(input_tokens):
    ngrams = []
    length = len(input_tokens)

    if length < 4:
        n_min = 3
    elif length < 5:
        n_min = 4
    elif length < 6:
        n_min = 5
    elif length > 7:
        n_min = 4
    else:
        n_min = 5
    
    n_max = min(6, length)
    
    for n in range(n_min, n_max + 1):
        for i in range(len(input_tokens) - n + 1):
            ngram = input_tokens[i:i+n]
            ngrams.append((" ".join(ngram), i))
    return ngrams

def token_levenshtein_distance(s1, s2):
    # Split the strings into tokens
    tokens1 = s1.split()
    tokens2 = s2.split()

    # Ensure we treat each token as a unit for the distance calculation
    # Create a map for all unique tokens encountered in both token lists
    unique_tokens = list(set(tokens1 + tokens2))
    token_to_index = {token: str(i) for i, token in enumerate(unique_tokens)}

    # Map each token in the lists to its index in the unique_tokens list
    indexed_str1 = ' '.join(token_to_index[token] for token in tokens1)
    indexed_str2 = ' '.join(token_to_index[token] for token in tokens2)
    
    # Calculate the Levenshtein distance treating indices as "words"
    dist = Levenshtein.distance(indexed_str1, indexed_str2)

    # Return the distance and the original token lists as formatted strings
    return dist

def generate_suggestions(pos_tags, rule_bank):
    input_tokens = [pos_tag for word, pos_tag in pos_tags]
    token_suggestions = [{"token": token, "suggestions": [], "distances": []} for token in input_tokens]
    insert_suggestions = defaultdict(list)
    
    input_ngrams_with_index = generate_ngrams(input_tokens)
    
    for input_ngram, start_idx in input_ngrams_with_index:
        min_distance = float('inf')
        best_match = None
        highest_frequency = 0
        ngram_length = len(input_ngram.split())
                
        for pattern_id, pattern_data in rule_bank.items():
            pattern_ngram = pattern_data.get('hybrid_ngram')
            frequency = pattern_data.get('frequency')
    
            if pattern_ngram:
                distance = token_levenshtein_distance(input_ngram, pattern_ngram)
                
                if distance < min_distance or (distance == min_distance and frequency > highest_frequency):
                    min_distance = distance
                    best_match = pattern_ngram
                    highest_frequency = frequency

                    print(f"Pattern N-gram '{input_ngram}' Distance'{distance}' Best Match: '{best_match}' Frequency '{frequency}'")
    
        if best_match:
            correction_tags = generate_correction_tags(input_ngram, best_match)
            logger.debug(f"CORRECTION TAGS {correction_tags}")
            
            input_ngram_tokens = input_ngram.split()
            token_shift = 0
            for i, tag in enumerate(correction_tags):
                token_idx = start_idx + i + token_shift

                if tag.startswith("INSERT"):
                    inserted_token = tag.split("_")[1]
                    insert_suggestions[token_idx].append(inserted_token)
                    token_shift = -1
                else:
                    if token_idx < len(token_suggestions):
                        token_suggestions[token_idx]["suggestions"].append(tag)
                        token_suggestions[token_idx]["distances"].append(min_distance)

    token_appearance_count = Counter()

    for _, start_idx in input_ngrams_with_index:
        for i in range(start_idx, start_idx + ngram_length):
            token_appearance_count[i] += 1
    max_count = max(token_appearance_count.values())
    
    # Handle inserts based on majority rule
    for token_idx, inserts in insert_suggestions.items():
        insert_counter = Counter(inserts)
        most_common_insert, insert_count = insert_counter.most_common(1)[0]
    
        if insert_count > 1:
            num_corrections = len(insert_counter)
            threshold = max_count / 2
            
            if insert_count > threshold:
                token_suggestions.insert(token_idx, {"token": most_common_insert, "suggestions": [f'INSERT_{most_common_insert}'], "distances": [0.8]})
        else:
            continue
    
    return token_suggestions

def load_pos_tag_dictionary(pos_tag, pos_path):
    csv_file_name = f"{pos_tag}_words.csv"
    exact_file_path = os.path.join(pos_path, csv_file_name)
    
    if os.path.exists(exact_file_path):
        print(f"Loading file for exact POS tag: {pos_tag}")
        return load_csv_words(exact_file_path)
    
    if '.*' in pos_tag:
        generalized_tag_pattern = re.sub(r'(.*)\.\*', r'\1', pos_tag)
        matching_words = []
        for file_name in os.listdir(pos_path):
            if file_name.startswith(generalized_tag_pattern):
                file_path = os.path.join(pos_path, file_name)
                print(f"Loading file for generalized POS tag: {file_name}")
                matching_words.extend(load_csv_words(file_path))
        
        if not matching_words:
            raise FileNotFoundError(f"No files found for POS tag pattern: {pos_tag}")
        
        return matching_words
    
    raise FileNotFoundError(f"CSV file for POS tag '{pos_tag}' not found")

def load_csv_words(file_path):
    df = pd.read_csv(file_path, header=None)
    words = df[0].dropna().tolist()
    return words

def weighted_levenshtein_word(word1, word2):
    len_word1 = len(word1)
    len_word2 = len(word2)
    distance_matrix = [[0] * (len_word2 + 1) for _ in range(len_word1 + 1)]
    
    for i in range(len_word1 + 1):
        distance_matrix[i][0] = i
    for j in range(len_word2 + 1):
        distance_matrix[0][j] = j
    
    substitution_weight = 1.0
    insertion_weight = 1.0
    deletion_weight = 1.0
    
    for i in range(1, len_word1 + 1):
        for j in range(1, len_word2 + 1):
            cost = 0 if word1[i-1] == word2[j-1] else substitution_weight
            distance_matrix[i][j] = min(
                distance_matrix[i-1][j] + deletion_weight,
                distance_matrix[i][j-1] + insertion_weight,
                distance_matrix[i-1][j-1] + cost
            )
    return distance_matrix[len_word1][len_word2]

def get_closest_words(word, dictionary, num_suggestions=5):
    word_distances = []
    for dict_word in dictionary:
        distance = weighted_levenshtein_word(word, dict_word)
        word_distances.append((dict_word, distance))
    
    word_distances.sort(key=lambda x: x[1])
    return word_distances[:num_suggestions]

def get_closest_words_by_pos(input_word, words_list, num_suggestions=1):
    if not words_list:
        return []
    word_distances = []
    for w in words_list:
        distance = weighted_levenshtein_word(input_word, w)
        word_distances.append((w, distance))
    word_distances.sort(key=lambda x: x[1])
    suggestions = word_distances[:min(len(word_distances), num_suggestions)]
    return suggestions

def apply_pos_corrections(token_suggestions, pos_tags, pos_path):
    final_sentence = []
    word_suggestions = {}
    pos_tag_dict = {}
    idx = 0
    inserted_tokens = set()

    for token_info in token_suggestions:
        suggestions = token_info["suggestions"]
        distances = token_info["distances"]

        if not suggestions:
            word = pos_tags[idx][0]
            final_sentence.append(word)
            idx += 1
            continue

        suggestion_count = Counter(suggestions)
        print(f"COUNTER {suggestion_count}")

        if suggestion_count:
            most_frequent_suggestion = suggestion_count.most_common(1)[0][0]
            suggestion_parts = most_frequent_suggestion.split("_")
            suggestion_type = suggestion_parts[0]
            current_pos = "_".join(suggestion_parts[1:-1])
            target_pos = suggestion_parts[-1]

            print(f"SUGGESTION TYPE: {suggestion_type}, CURRENT POS: {current_pos}, TARGET POS: {target_pos}")

            if suggestion_type == "KEEP":
                word = pos_tags[idx][0]
                final_sentence.append(word)
                idx += 1

            elif suggestion_type == "SUBSTITUTE":
                input_word = pos_tags[idx][0]
                if target_pos not in pos_tag_dict:
                    word_list = load_pos_tag_dictionary(target_pos, pos_path)
                    pos_tag_dict[target_pos] = word_list
                else:
                    word_list = pos_tag_dict[target_pos]

                suggestions_list = get_closest_words_by_pos(input_word, word_list, num_suggestions=1)

                if suggestions_list:
                    replacement_word = suggestions_list[0][0]
                    final_sentence.append(replacement_word)
                    print(f"Replaced '{input_word}' with '{replacement_word}' for target POS '{target_pos}'")
                    word_suggestions[input_word] = [w for w, dist in suggestions_list]
                else:
                    final_sentence.append(input_word)

                idx += 1

            elif suggestion_type == "DELETE":
                idx += 1

            elif suggestion_type == "INSERT":
                if target_pos not in pos_tag_dict:
                    word_list = load_pos_tag_dictionary(target_pos, pos_path)
                    pos_tag_dict[target_pos] = word_list
                else:
                    word_list = pos_tag_dict[target_pos]

                if word_list:
                    inserted_token = word_list[0]
                else:
                    inserted_token = "[UNK]"

                if inserted_token not in inserted_tokens:
                    final_sentence.append(inserted_token)
                    inserted_tokens.add(inserted_token)
                else:
                    # If already inserted, skip
                    pass

            else:
                word = pos_tags[idx][0]
                final_sentence.append(word)
                idx += 1
        else:
            word = pos_tags[idx][0]
            final_sentence.append(word)
            idx += 1

    corrected_sentence = " ".join(final_sentence)
    return corrected_sentence

def check_words_in_dictionary(words, directory_path):
    incorrect_words = []
    dictionary = load_dictionary(directory_path)
    
    for word in words:
        if word.lower() not in dictionary:
            incorrect_words.append(word)
    
    has_incorrect_word = len(incorrect_words) > 0
    logger.debug(f"Incorrect Words: {incorrect_words}")
    return incorrect_words, has_incorrect_word

def spell_check_word(word, directory_path, num_suggestions=5):
    dictionary = load_dictionary(directory_path)
    word_lower = word.lower()
    
    if word_lower in dictionary:
        return word, None
    else:
        suggestions = get_closest_words(word_lower, dictionary, num_suggestions=num_suggestions)
        if suggestions:
            return word, [suggestion[0] for suggestion in suggestions]
        else:
            return word, None

def spell_check_incorrect_words(text, incorrect_words, directory_path, num_suggestions=5):
    corrected_text = text
    suggestions_dict = {}

    for word in incorrect_words:
        misspelled_word, suggestions = spell_check_word(word, directory_path, num_suggestions)
        if suggestions:
            log_message("info", f"Suggestions for '{word}': {suggestions}")
            suggestions_dict[word] = suggestions

            if len(suggestions) >= 3:
                corrected_word = suggestions[2]
            else:
                corrected_word = suggestions[0]
            
            corrected_text = re.sub(r'\b{}\b'.format(re.escape(word)), corrected_word, corrected_text)
            log_message("info", f"Replaced '{word}' with '{corrected_word}'")
        else:
            log_message("warning", f"No suggestions found for '{word}'")
            suggestions_dict[word] = []

    return corrected_text, suggestions_dict, incorrect_words

def pantasa_checker(input_sentence, jar_path, model_path, rule_path, directory_path, pos_path):
    log_message("info", "No misspelled words found, proceeding with grammar checking")

    # Load the rule bank once here
    global_rule_bank = rule_pattern_bank(rule_path)

    log_message("info", "Applying post-defined rules (post)")
    post_rules_corrected_text = apply_predefined_rules_post(input_sentence)
    log_message("info", f"Text after post-defined rules (post): {post_rules_corrected_text}")

    log_message("info", "Retokenizing and re-POS tagging after post modifications")
    tokens = tokenize_sentence(post_rules_corrected_text)
    pos_tags = pos_tagging(tokens, jar_path, model_path)
    if not pos_tags:
        log_message("error", "POS tagging failed after modifications")
        return [], [], []

    log_message("info", "Generating suggestions using n-gram matching")
    # Use the pre-loaded rule_bank here
    token_suggestions = generate_suggestions(pos_tags, global_rule_bank)
    log_message("info", f"Token Suggestions: {token_suggestions}")

    log_message("info", "Applying POS corrections")
    corrected_sentence = apply_pos_corrections(token_suggestions, pos_tags, pos_path)

    log_message("info", f"Final Corrected Sentence: {corrected_sentence}")
    return corrected_sentence, token_suggestions, []
