import pandas as pd
import re
from collections import defaultdict, Counter

# Step 1: Load and create the rule pattern bank
file_path = 'data/processed/hngrams.csv'  # Update with actual path
hybrid_ngrams_df = pd.read_csv(file_path)

# Create a dictionary to store the Rule Pattern Bank (Hybrid N-Grams + Predefined Rules)
rule_pattern_bank = {}

# Store the hybrid n-grams from the CSV file into the rule pattern bank
for index, row in hybrid_ngrams_df.iterrows():
    hybrid_ngram = row['Hybrid_N-Gram']
    pattern_frequency = row['Frequency']

# Step 2: Define the weighted Levenshtein distance function
def weighted_levenshtein(input_ngram, pattern_ngram):
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
    substitution_weight = 1.0
    insertion_weight = 0.5 
    deletion_weight = 0.8

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
    if length < 5:
        n_min = 3
    elif length < 6:
        n_min = 4
    elif length < 7:
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
def generate_suggestions(input_pos):
    input_tokens = input_pos.strip().split()
    
    # Generate token-level correction tracker
    token_suggestions = [{"token": token, "suggestions": [], "distances": []} for token in input_tokens]
    
    # Track potential inserts (insertion suggestions for each index)
    insert_suggestions = defaultdict(list)
    
    # Generate 3-gram to 7-gram sequences from the input sentence
    input_ngrams_with_index = generate_ngrams(input_tokens)
    
    # Iterate over each n-gram and compare it to the rule pattern bank
    for input_ngram, start_idx in input_ngrams_with_index:
        min_distance = float('inf')
        best_match = None
        highest_frequency = 0  # Track the highest frequency pattern

        for index, row in hybrid_ngrams_df.iterrows():
            pattern_ngram = row['Hybrid_N-Gram']
            pattern_frequency = row['Frequency']

            # Compare input n-gram with each pattern n-gram from the rule pattern bank
            if pattern_ngram:
                distance = weighted_levenshtein(input_ngram, pattern_ngram)
                # Prioritize by both minimal distance and high frequency
                if distance < min_distance or (distance == min_distance and pattern_frequency > highest_frequency):
                    min_distance = distance
                    best_match = pattern_ngram
                    highest_frequency = pattern_frequency  # Update to use the more frequent pattern
            
        if best_match:
            correction_tags = generate_correction_tags(input_ngram, best_match)
            print(f"CORRECTION TAGS {correction_tags}")
            
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


# Step 6: Correction phase - apply the suggestions to correct the input sentence
def apply_corrections(token_suggestions):
    final_sentence = []
    
    # Track inserted tokens to avoid duplication
    inserted_tokens = set()

    # Iterate through the token_suggestions and apply the corrections
    for token_info in token_suggestions:
        suggestions = token_info["suggestions"]
        distances = token_info["distances"]
        
        if not suggestions:
            # If no suggestions, keep the original token
            final_sentence.append(token_info["token"])
            continue
        
        # Step 1: Count the frequency of each exact suggestion (e.g., SUBSTITUTE_DTCP_DTC, KEEP_DTCP_DTCP)
        suggestion_count = Counter(suggestions)  # Count occurrences of each exact suggestion
        
        if suggestion_count:
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
                inserted_token = best_suggestion.split("_")[1]
                # Only insert if not already inserted at this position
                if inserted_token not in inserted_tokens:
                    final_sentence.append(inserted_token)  # Insert new token
                    inserted_tokens.add(inserted_token)  # Track the insertion to avoid duplication
                else:
                    continue  # Skip the token if it's already inserted
        else:
            # If no valid suggestions, append the original token
            final_sentence.append(token_info["token"])

    return " ".join(final_sentence)


input_pos = 'LM VBTS CCB DTCP'

corrected = generate_suggestions(input_pos)

print(corrected)