import re
from utils import load_hybrid_ngram_patterns, process_sentence_with_dynamic_ngrams, extract_ngrams
from preprocess import preprocess_text
import logging

logger = logging.getLogger(__name__)

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
    - A list of matched n-grams and their corresponding pattern IDs.
    """
    sentence_tokens = [tag[1] for tag in input_pos_tags]  # Extract POS tags
    matched_patterns = []  # To store matched n-grams and their patterns

    # Use dynamic N-gram size generation for matching
    ngrams = extract_ngrams(sentence_tokens)

    total_comparisons = 0  # To count total comparisons made

    logger.info(f"Total n-grams from input sentence: {len(ngrams)}")  # Log total n-grams generated

    # Iterate through each n-gram generated from the input sentence
    for ngram_index, ngram in enumerate(ngrams):
        logger.info(f"Processing n-gram {ngram_index + 1}/{len(ngrams)}: {ngram}")

        # Iterate through each hybrid n-gram pattern
        for hybrid_index, hybrid_ngram in enumerate(hybrid_ngram_patterns):
            total_comparisons += 1  # Increment the comparison counter

            # Perform the comparison between the current n-gram and hybrid n-gram
            if match_pos_to_hybrid_ngram(ngram, hybrid_ngram['ngram_pattern']):
                logger.info(f"Match found: Input n-gram {ngram} matches hybrid n-gram {hybrid_ngram['ngram_pattern']} (Pattern ID: {hybrid_ngram['pattern_id']})")
                matched_patterns.append({'ngram': ngram, 'pattern_id': hybrid_ngram['pattern_id']})
            else:
                logger.debug(f"No match: Input n-gram {ngram} does not match hybrid n-gram {hybrid_ngram['ngram_pattern']}")

    logger.info(f"Total comparisons made: {total_comparisons}")
    
    return matched_patterns

def ngram_matching(input_sentence, jar_path, model_path, hybrid_ngram_patterns):
    # Step 1: Preprocess the sentence (tokenize, POS tag, and lemmatize)
    preprocessed_output = preprocess_text(input_sentence, jar_path, model_path)
    if not preprocessed_output:
        logger.error("Error during preprocessing.")
        return []

    print(f"f{preprocessed_output}")
    tokens, lemmas, pos_tags, checked_sentence = preprocessed_output[0]
    
    # Step 2: Extract dynamically sized N-grams from the tokens
    ngram_collections = process_sentence_with_dynamic_ngrams(tokens)
    
    matches = []  # To store all matching patterns

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
            matched_patterns = compare_with_hybrid_ngrams(pos_tag_list, hybrid_ngram_patterns)
            if matched_patterns:
                matches.extend(matched_patterns)

    return matches, ngram_collections, preprocessed_output

# Example usage
if __name__ == "__main__":

    hybrid_ngram_patterns = load_hybrid_ngram_patterns('data/processed/hngrams.csv')

    jar_path = r'C:\Projects\Pantasa\rules\Libraries\FSPOST\stanford-postagger.jar'
    model_path = r'C:\Projects\Pantasa\rules\Libraries\FSPOST\filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'

    input_sentence = "kumain ang mga bata ng mansana"

    matched_ngrams = ngram_matching(input_sentence, jar_path, model_path, hybrid_ngram_patterns)

    if matched_ngrams:
        print("Matched n-grams and their patterns:")
        for match in matched_ngrams:
            print(match)
    else:
        print("No n-gram matched with any hybrid n-gram pattern.")
