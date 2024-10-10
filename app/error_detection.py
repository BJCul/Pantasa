import re
from utils import load_hybrid_ngram_patterns, process_sentence_with_dynamic_ngrams, extract_ngrams
from preprocess import preprocess_text

def match_pos_to_hybrid_ngram(input_pos_tags, hybrid_ngram, flexibility=True):
    """
    Checks if the input POS tag sequence matches the hybrid n-gram pattern with flexibility.
    """
    if len(input_pos_tags) != len(hybrid_ngram):
        if flexibility and abs(len(input_pos_tags) - len(hybrid_ngram)) <= 1:
            pass  # Minor length difference tolerated
        else:
            return False  # Length mismatch

    for i, pattern in enumerate(hybrid_ngram):
        input_pos_tag = input_pos_tags[i][1]  # Extract the POS tag from the tuple
        
        if "_" in pattern:  # Handle hybrid patterns with underscores
            parts = pattern.split("_")
            input_parts = input_pos_tag.split("_")
            
            if len(input_parts) < 2:
                return False  # No match if the input lacks expected parts

            # Match both parts of the pattern with flexibility if enabled
            if not re.match(parts[0], input_parts[0]) or not re.match(parts[1], input_parts[1]):
                return False
        else:
            # Match general POS tags (e.g., VB.* matches VB, VBAF, VBTS)
            general_pattern = pattern.replace('.*', '')
            if not re.match(re.escape(general_pattern), input_pos_tag):
                if flexibility:
                    # Allow for small differences (e.g., NN vs NNS)
                    if input_pos_tag.startswith(general_pattern):
                        continue
                    else:
                        return False
                else:
                    return False

    return True


def compare_with_hybrid_ngrams(input_pos_tags, hybrid_ngram_patterns):
    """
    Compare the input sentence's POS tags against hybrid n-gram patterns using dynamically sized N-grams.

    Args:
    - input_pos_tags: List of POS tags from the input sentence.
    - hybrid_ngram_patterns: List of hybrid n-gram patterns.

    Returns:
    - List of matching pattern IDs.
    """
    matching_patterns = []
    sentence_tokens = [tag[1] for tag in input_pos_tags]  # Extract POS tags

    # Use dynamic N-gram size generation for matching
    ngrams = extract_ngrams(sentence_tokens)

    total_comparisons = 0  # To count total comparisons made

    print(f"Total n-grams from input sentence: {len(ngrams)}")  # Log total n-grams generated

    # Iterate through each n-gram generated from the input sentence
    for ngram_index, ngram in enumerate(ngrams):
        print(f"Processing n-gram {ngram_index + 1}/{len(ngrams)}: {ngram}")

        # Iterate through each hybrid n-gram pattern
        for hybrid_index, hybrid_ngram in enumerate(hybrid_ngram_patterns):
            print(f"Comparing n-gram {ngram} with hybrid n-gram {hybrid_index + 1}/{len(hybrid_ngram_patterns)}")

            total_comparisons += 1  # Increment the comparison counter

            # Perform the comparison between the current n-gram and hybrid n-gram
            if match_pos_to_hybrid_ngram(ngram, hybrid_ngram['ngram_pattern']):
                print(f"Match found: {ngram} matches {hybrid_ngram['ngram_pattern']}")
                matching_patterns.append(hybrid_ngram['pattern_id'])

    print(f"Total comparisons made: {total_comparisons}")

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
            
            # Show the comparison between input n-gram and hybrid n-gram
            input_ngram_str = " ".join([tag[1] for tag in input_pos_tags])
            hybrid_ngram_str = " ".join(ngram_tags)
            suggestions.append(f"Pattern ID {pattern_id}: Suggest {suggestion}")
            suggestions.append(f"Comparing input n-gram: {input_ngram_str} with hybrid n-gram: {hybrid_ngram_str}")
    else:
        suggestions.append("No suggestions available.")

    return suggestions


def detect_errors_with_pantasa(input_sentence, jar_path, model_path, hybrid_ngram_patterns):
    # Step 1: Preprocess the sentence (tokenize, POS tag, and lemmatize)
    preprocessed_output = preprocess_text(input_sentence, jar_path, model_path)
    if not preprocessed_output:
        return True, "Error during preprocessing."

    tokens, lemmas, pos_tags = preprocessed_output[0]
    
    # Step 2: Extract dynamically sized N-grams from the tokens
    ngram_collections = process_sentence_with_dynamic_ngrams(tokens)
    
    # Use the original POS tags to generate the pos_tag_list for each n-gram
    for ngram_size, ngrams in ngram_collections.items():
        for ngram in ngrams:
            # Match tokens in n-gram with their POS tags from the original pos_tags list
            pos_tag_list = []
            for token in ngram:
                for original_token, pos_tag in pos_tags:
                    if original_token == token:
                        pos_tag_list.append((original_token, pos_tag))
                        break

            # Now compare the pos_tag_list with hybrid n-gram patterns
            matching_patterns = compare_with_hybrid_ngrams(pos_tag_list, hybrid_ngram_patterns)
            
            if matching_patterns:
                return False, f"No error detected with {ngram_size}: {ngram}"

    suggestions = generate_suggestions(pos_tag_list, hybrid_ngram_patterns)
    return True, f"Error detected: No matching hybrid n-gram pattern found.\nSuggestions:\n" + "\n".join(suggestions)


# Example usage
if __name__ == "__main__":

    hybrid_ngram_patterns = load_hybrid_ngram_patterns('data/processed/hngrams.csv')

    jar_path = r'C:\Projects\Pantasa\rules\Libraries\FSPOST\stanford-postagger.jar'
    model_path = r'C:\Projects\Pantasa\rules\Libraries\FSPOST\filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'

    input_sentence = "kumain ang mga bata ng mansanas"

    has_error, message = detect_errors_with_pantasa(input_sentence, jar_path, model_path, hybrid_ngram_patterns)
    print(message)
