from detection import handle_errors
from predefined_rules.rule_main import apply_predefined_rules
from ngram_matching import ngram_matching
from utils import load_hybrid_ngram_patterns

def generate_suggestions(errors, rule_corrected):
    """
    Generates suggestions based on the detected errors and rule corrections.
    
    Args:
    - errors: List of detected errors from the error detection module.
    - rule_corrected: Corrected text after applying predefined rules.
    
    Returns:
    - List of suggestions.
    """
    suggestions = []
    unique_suggestions = set()  # Use a set to track unique suggestions
    
    for error in errors:
        suggestion = f"Suggested correction: {rule_corrected}"
        
        if suggestion not in unique_suggestions:
            suggestions.append(suggestion)
            unique_suggestions.add(suggestion)  # Track the suggestion to avoid duplicates
    
    return suggestions

def process_sentence(input_sentence, jar_path, model_path, hybrid_ngram_patterns):
    
    # Step 1: N-Gram Matching
    ngram_matching_result, ngram_collection, preprocessed_output = ngram_matching(input_sentence, jar_path, model_path, hybrid_ngram_patterns)
    
    # Step 2: Error Detection
    detected_errors = handle_errors(ngram_matching_result, ngram_collections=ngram_collection)
    
    # Step 3: Rule Checking
    if detected_errors:
        tokens, lemmas, pos_tags, checked_sentence = preprocessed_output[0]
        rule_corrected_text = apply_predefined_rules(checked_sentence)
        print(f"{rule_corrected_text}")
    else:
        rule_corrected_text = input_sentence
    
    # Step 4: Suggestion Generation
    suggestions = generate_suggestions(detected_errors, rule_corrected_text)
    
    return suggestions

# Example usage
if __name__ == "__main__":
    input_sentence = "aya ko na maki pag usap"
    jar_path = 'rules/Libraries/FSPOST/stanford-postagger.jar'
    model_path = 'rules/Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    hybrid_ngram_patterns = load_hybrid_ngram_patterns('data/processed/hngrams.csv')
    suggestions = process_sentence(input_sentence, jar_path, model_path, hybrid_ngram_patterns)
    print(suggestions)