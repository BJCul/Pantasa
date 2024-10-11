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
    
    for error in errors:
        suggestions.append(f"Suggested correction: {rule_corrected}")
    
    return suggestions

def process_sentence(input_sentence, jar_path, model_path, hybrid_ngram_patterns):
    
    # Step 1: N-Gram Matching
    ngram_matching_result = ngram_matching(input_sentence, jar_path, model_path, hybrid_ngram_patterns)

    ngram_collection = {}
    # Step 2: Error Detection
    detected_errors = handle_errors(ngram_matching_result, ngram_collections=ngram_collection)
    
    # Step 3: Rule Checking
    if detected_errors:
        rule_corrected_text = apply_predefined_rules(input_sentence)
    else:
        rule_corrected_text = input_sentence
    
    # Step 4: Suggestion Generation
    suggestions = generate_suggestions(detected_errors, rule_corrected_text)
    print (f"SDASDAS {rule_corrected_text}")
    
    return suggestions

# Example usage
if __name__ == "__main__":
    input_sentence = "kumain ang mga bata ng mansana"
    jar_path = 'rules/Libraries/FSPOST/stanford-postagger.jar'
    model_path = 'rules/Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    hybrid_ngram_patterns = load_hybrid_ngram_patterns('data/processed/hngrams.csv')
    suggestions = process_sentence(input_sentence, jar_path, model_path, hybrid_ngram_patterns)
    print("Suggestions:", suggestions)
