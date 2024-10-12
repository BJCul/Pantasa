from app.detection import handle_errors
from app.predefined_rules.rule_main import apply_predefined_rules
from app.ngram_matching import ngram_matching
from app.suggestion import generate_suggestions

def pantasa(input_sentence, jar_path, model_path, hybrid_ngram_patterns):
    
    # Step 1: N-Gram Matching
    ngram_matching_result, ngram_collection, preprocessed_output = ngram_matching(input_sentence, jar_path, model_path, hybrid_ngram_patterns)
    
    # Step 2: Error Detection
    detected_errors = handle_errors(ngram_matching_result, ngram_collections=ngram_collection)
    
    # Step 3: Rule Checking
    if detected_errors:
        tokens, lemmas, pos_tags, checked_sentence, misspelled_words = preprocessed_output[0]

        rule_corrected_text = apply_predefined_rules(checked_sentence)
        print(f"{rule_corrected_text}")
    else:
        rule_corrected_text = input_sentence
    
    # Step 4: Suggestion Generation
    suggestions, misspelled_words, rule_corrected_text = generate_suggestions(detected_errors, rule_corrected_text, misspelled_words)
    print("---------------------------------------------------------")
    print(suggestions)
    print(misspelled_words)
    print(rule_corrected_text)
    return suggestions, misspelled_words, rule_corrected_text
