import re
from difflib import get_close_matches
from utils import load_hybrid_ngram_patterns
from preprocess import preprocess_text
from error_detection import match_pos_to_hybrid_ngram, compare_pos_sequences, generate_substitution_suggestion

# Load Hybrid N-Grams from CSV
hybrid_ngram_patterns = load_hybrid_ngram_patterns('data/processed/hngrams.csv')

# Function to compare input sentence's POS tags to hybrid n-grams and detect errors
def detect_errors_with_pantasa(input_sentence, jar_path, model_path, hybrid_ngram_patterns):
    # Step 1: Preprocess the input sentence (tokenization, POS tagging, and lemmatization)
    preprocessed_output = preprocess_text(input_sentence, jar_path, model_path)
    if not preprocessed_output:
        return False, "Error during preprocessing"
    
    tokens, lemmas, pos_tag_list = preprocessed_output[0]  # Unpack the preprocessed output
    print(f"POS Tag Sequence: {pos_tag_list}")
    
    # Step 2: Match POS tags with stored hybrid n-grams
    matching_patterns = compare_with_hybrid_ngrams(pos_tag_list, hybrid_ngram_patterns)
    
    if not matching_patterns:
        # No exact match, use Levenshtein distance or closest match suggestions
        suggestions = generate_suggestions(pos_tag_list, hybrid_ngram_patterns)
        return True, f"Error detected: No matching hybrid n-gram pattern found.\nSuggestions: {suggestions}"
    
    return False, "No error detected: Sentence is grammatically correct."

# Matching POS Tags with Hybrid N-Grams
def compare_with_hybrid_ngrams(input_pos_tags, hybrid_ngram_patterns):
    """
    Compare the POS tags of the input sentence with the hybrid n-gram patterns.
    
    Args:
    - input_pos_tags: List of POS tags from the input sentence (as tuples).
    - hybrid_ngram_patterns: List of hybrid n-gram patterns.

    Returns:
    - List of matching pattern IDs.
    """
    matching_patterns = []
    
    for hybrid_ngram in hybrid_ngram_patterns:
        if match_pos_to_hybrid_ngram(input_pos_tags, hybrid_ngram['ngram_pattern']):
            matching_patterns.append(hybrid_ngram['pattern_id'])
    
    return matching_patterns


# Generate suggestions based on closest matching hybrid n-grams
def generate_suggestions(input_pos_tags, hybrid_ngram_patterns):
    closest_matches = []
    suggestions = []
    
    # Find the closest hybrid n-grams by comparing POS tags
    for hybrid_ngram in hybrid_ngram_patterns:
        hybrid_ngram_tags = hybrid_ngram['ngram_pattern']
        
        # Use Levenshtein distance or close matches to find similar POS tag sequences
        similarity = compare_pos_sequences(input_pos_tags, hybrid_ngram_tags)
        
        if similarity < 2:  # Threshold for closeness (you can adjust it)
            closest_matches.append((hybrid_ngram['pattern_id'], hybrid_ngram_tags, similarity))
    
    # Generate substitution or insertion suggestions based on closest matches
    if closest_matches:
        for match in closest_matches:
            pattern_id, ngram_tags, distance = match
            suggestion = generate_substitution_suggestion(input_pos_tags, ngram_tags)
            suggestions.append(f"Pattern ID {pattern_id}: Suggest {suggestion}")
    
    return suggestions

# Example Execution of Error Detection for Pantasa
if __name__ == "__main__":
    jar_path = r'C:\Projects\Pantasa\rules\Libraries\FSPOST\stanford-postagger.jar'
    model_path = r'C:\Projects\Pantasa\rules\Libraries\FSPOST\filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'

    # Input sentence for detection
    input_sentence = "ang pagwawasto ng mga texto"
    
    # Run error detection
    has_error, message = detect_errors_with_pantasa(input_sentence, jar_path, model_path, hybrid_ngram_patterns)
    print(message)
