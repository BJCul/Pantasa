# app/error_detection.py

import re
from utils import load_hybrid_ngram_patterns
from preprocess import preprocess_text

def match_pos_to_hybrid_ngram(input_pos_tags, hybrid_ngram, flexibility=False):
    """
    Checks if the input POS tag sequence matches the hybrid n-gram pattern.

    Args:
    - input_pos_tags: List of POS tags from the input sentence.
    - hybrid_ngram: List of POS tag patterns from the hybrid n-gram.

    Returns:
    - True if the input POS tag sequence matches the hybrid n-gram pattern, otherwise False.
    """
    if len(input_pos_tags) != len(hybrid_ngram):
        if flexibility and abs(len(input_pos_tags) - len(hybrid_ngram)) <= 1:
            print("Warning: Minor length difference tolerated.")
        else:
            return False  # Length mismatch

    for i, pattern in enumerate(hybrid_ngram):
        input_pos_tag = input_pos_tags[i][1]  # Extract the POS tag from the tuple
        
        if "_" in pattern:  # Handle hybrid patterns with underscores
            parts = pattern.split("_")
            input_parts = input_pos_tag.split("_")
            
            if len(input_parts) < 2:
                print(f"Mismatch at position {i}: Expected underscore in tag but not found.")
                return False  # No match if the input lacks expected parts

            # Match both parts of the pattern with flexibility if enabled
            if not re.match(parts[0], input_parts[0]) or not re.match(parts[1], input_parts[1]):
                print(f"Mismatch at position {i}: {parts} does not match {input_parts}")
                return False
        else:
            # Match general POS tags (e.g., VB.* matches VB, VBAF, VBTS)
            general_pattern = pattern.replace('.*', '')
            if not re.match(re.escape(general_pattern), input_pos_tag):
                if flexibility:
                    # Allow for small differences (e.g., NN vs NNS)
                    if input_pos_tag.startswith(general_pattern):
                        print(f"Flexibility allowed: {input_pos_tag} loosely matches {general_pattern}")
                        continue
                    else:
                        print(f"Mismatch at position {i}: {input_pos_tag} does not match {pattern}")
                        return False
                else:
                    print(f"Mismatch at position {i}: {input_pos_tag} does not match {pattern}")
                    return False

    return True


def compare_with_hybrid_ngrams(input_pos_tags, hybrid_ngram_patterns):
    matching_patterns = []

    for hybrid_ngram in hybrid_ngram_patterns:
        if match_pos_to_hybrid_ngram(input_pos_tags, hybrid_ngram['ngram_pattern']):
            matching_patterns.append(hybrid_ngram['pattern_id'])

    return matching_patterns

def compare_pos_sequences(input_pos_tags, hybrid_ngram_tags):
    mismatches = 0
    min_len = min(len(input_pos_tags), len(hybrid_ngram_tags))

    for i in range(min_len):
        input_pos_tag = input_pos_tags[i][1]
        hybrid_ngram_tag = hybrid_ngram_tags[i]

        if not re.match(re.escape(hybrid_ngram_tag), re.escape(input_pos_tag)):
            mismatches += 1

    mismatches += abs(len(input_pos_tags) - len(hybrid_ngram_tags))
    return mismatches

def generate_substitution_suggestion(input_pos_tags, hybrid_ngram_tags):
    suggestions = []
    for i in range(len(hybrid_ngram_tags)):
        if i >= len(input_pos_tags):
            suggestions.append(f"insert {hybrid_ngram_tags[i]}")
            continue

        input_pos_tag = input_pos_tags[i][1]
        if not re.match(re.escape(hybrid_ngram_tags[i]), re.escape(input_pos_tag)):
            suggestions.append(f"replace {input_pos_tag} with {hybrid_ngram_tags[i]}")

    return ", ".join(suggestions)

def generate_suggestions(input_pos_tags, hybrid_ngram_patterns):
    closest_matches = []
    suggestions = []

    for hybrid_ngram in hybrid_ngram_patterns:
        hybrid_ngram_tags = hybrid_ngram['ngram_pattern']
        similarity = compare_pos_sequences(input_pos_tags, hybrid_ngram_tags)
        if similarity <= 2:  # Adjust threshold as needed
            closest_matches.append((hybrid_ngram['pattern_id'], hybrid_ngram_tags, similarity))

    if closest_matches:
        closest_matches.sort(key=lambda x: x[2])  # Sort by similarity
        for match in closest_matches:
            pattern_id, ngram_tags, distance = match
            suggestion = generate_substitution_suggestion(input_pos_tags, ngram_tags)
            suggestions.append(f"Pattern ID {pattern_id}: Suggest {suggestion}")
    else:
        suggestions.append("No suggestions available.")

    return suggestions

def detect_errors_with_pantasa(input_sentence, jar_path, model_path, hybrid_ngram_patterns):
    preprocessed_output = preprocess_text(input_sentence)
    if not preprocessed_output:
        return True, "Error during preprocessing."

    tokens, lemmas, pos_tags = preprocessed_output[0]
    pos_tag_list = pos_tags

    matching_patterns = compare_with_hybrid_ngrams(pos_tag_list, hybrid_ngram_patterns)

    if not matching_patterns:
        suggestions = generate_suggestions(pos_tag_list, hybrid_ngram_patterns)
        print(pos_tag_list)
        return True, f"Error detected: No matching hybrid n-gram pattern found.\nSuggestions:\n" + "\n".join(suggestions)

    return False, "No error detected: Sentence is grammatically correct."

# Example usage
if __name__ == "__main__":
    hybrid_ngram_patterns = load_hybrid_ngram_patterns('data/processed/hngrams.csv')

    jar_path = r'C:\Projects\Pantasa\rules\Libraries\FSPOST\stanford-postagger.jar'
    model_path = r'C:\Projects\Pantasa\rules\Libraries\FSPOST\filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'

    input_sentence = "Isang pamilya tayo"

    has_error, message = detect_errors_with_pantasa(input_sentence, jar_path, model_path, hybrid_ngram_patterns)
    print(message)
