from collections import Counter
import logging
import pandas as pd

def load_pos_tag_dictionary(csv_path):
    """
    Load a POS tag dictionary from a CSV file.
    
    Args:
    - csv_path (str): Path to the CSV file containing the POS tag dictionary.

    Returns:
    - pos_tag_dict (dict): Dictionary where keys are POS tags and values are lists of words.
    """
    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # Initialize an empty dictionary to store POS tags and words
    pos_tag_dict = {}
    
    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        # The POS_TAG is in the first column and WORDS are in the subsequent columns
        pos_tag = row[0]  # First column as POS tag
        words = row[1:].dropna().tolist()  # All other columns are words, drop NaN values
        
        # Add the POS tag and corresponding words to the dictionary
        pos_tag_dict[pos_tag] = words
    
    return pos_tag_dict

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
    substitution_weight = 0.8
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

def get_closest_words_by_pos(input_word, target_pos_tag, pos_tag_dict, num_suggestions=4):
    """
    Get the closest words to the input word from the dictionary of words with the target POS tag.

    Args:
    - input_word: The word to find replacements for.
    - target_pos_tag: The POS tag of the words to search for.
    - pos_tag_dict: Dictionary mapping POS tags to lists of words.
    - num_suggestions: Number of suggestions to return.

    Returns:
    - A list of tuples: (word, distance)
    """
    words_list = pos_tag_dict.get(target_pos_tag, [])
    if not words_list:
        return []

    # Compute distances
    word_distances = []
    for word in words_list:
        distance = weighted_levenshtein_word(input_word, word)
        word_distances.append((word, distance))

    # Sort by distance
    word_distances.sort(key=lambda x: x[1])

    # Get top N suggestions
    suggestions = word_distances[:num_suggestions]

    return suggestions


# Modification from the apply_pos_corrections function from pantasa checker
def apply_pos_corrections(token_suggestions, pos_tags, directory_path):
    final_sentence = []
    word_suggestions = {}  # To keep track of suggestions for each word
    pos_tag_dict = {}  # Cache for loaded POS tag dictionaries

    # Iterate through the token_suggestions and apply the corrections
    for idx, token_info in enumerate(token_suggestions):
        suggestions = token_info["suggestions"]
        distances = token_info["distances"]
        
        if not suggestions:
            # No suggestions; keep the original word
            word = pos_tags[idx][0]
            final_sentence.append(word)
            continue  # Move to the next token

        # Count the frequency of each exact suggestion
        suggestion_count = Counter(suggestions)
        print(f"COUNTER {suggestion_count}")
    
        # Find the most frequent suggestion
        most_frequent_suggestion = suggestion_count.most_common(1)[0][0]
        print(f"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        # Filter suggestions matching the most frequent suggestion
        filtered_indices = [i for i, s in enumerate(suggestions) if s == most_frequent_suggestion]

        # Pick the suggestion with the lowest distance if there's a tie
        if len(filtered_indices) > 1:
            filtered_distances = [distances[i] for i in filtered_indices]
            best_filtered_index = filtered_distances.index(min(filtered_distances))
            best_index = filtered_indices[best_filtered_index]
            print(f"BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")

        else:
            best_index = filtered_indices[0]
            print(f"CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")

                
        best_suggestion = suggestions[best_index]
        suggestion_parts = best_suggestion.split("_")
        suggestion_type = suggestion_parts[0]
        print(f"CDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")

        if suggestion_type == "KEEP":
            # Append the original word
            word = pos_tags[idx][0]
            final_sentence.append(word)
            print(f"EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        elif suggestion_type == "SUBSTITUTE":
            # Extract input word and target POS tag
            input_word = pos_tags[idx][0]
            input_pos = suggestion_parts[1]
            target_pos = suggestion_parts[2]

            print(f"INPUT WORD: {input_word}")
            print(f"INPUT POS: {input_pos}")
            print(f"TARGET POS: {target_pos}")

            # Load the dictionary for the target POS tag if not already loaded
            if target_pos not in pos_tag_dict:
                word_list = load_pos_tag_dictionary(target_pos, directory_path)
                pos_tag_dict[target_pos] = word_list
            else:
                word_list = pos_tag_dict[target_pos]

            # Get closest words by POS
            suggestions_list = get_closest_words_by_pos(input_word, word_list, num_suggestions=3)

            if suggestions_list:
                # For now, pick the best suggestion (smallest distance)
                replacement_word = suggestions_list[0][0]
                final_sentence.append(replacement_word)
                
                # Store suggestions for the word
                word_suggestions[input_word] = [word for word, dist in suggestions_list]
            else:
                # If no suggestions found, keep the original word
                final_sentence.append(input_word)
        elif suggestion_type == "DELETE":
            continue  # Skip the token
        elif suggestion_type == "INSERT":
            # Handle insertion if necessary
            pass
    
    corrected_sentence = " ".join(final_sentence)
    return corrected_sentence, word_suggestions



def pantasa_checker(input_sentence, jar_path, model_path, rule_bank, pos_tag_dict):
    # Preprocess the sentence
    preprocessed_output = preprocess_text(input_sentence, jar_path, model_path)
    if not preprocessed_output:
        logger.error("Error during preprocessing.")
        return []
    
    tokens, lemmas, pos_tags, checked_sentence, misspelled_words = preprocessed_output[0]

    # Generate suggestions
    log_message("info", f"POS TAGS: {pos_tags}")
    token_suggestions = generate_suggestions(pos_tags)

    # Apply corrections
    corrected_sentence, word_suggestions = apply_pos_corrections(token_suggestions, pos_tags, pos_tag_dict)
    print(f"Input Sentence: {input_sentence}")
    print(f"POS Corrected Sentence: {corrected_sentence}")
    print(f"Misspelled word: {misspelled_words}")

    # Return corrected sentence and suggestions
    return corrected_sentence, misspelled_words, word_suggestions
    
