from app.detection import handle_errors
from app.predefined_rules.rule_main import apply_predefined_rules
from app.ngram_matching import ngram_matching
from app.utils import load_hybrid_ngram_patterns

def generate_suggestions(errors, rule_corrected, misspelled_words):
    """
    Generates separate suggestions for grammar errors and misspelled words.
    
    Args:
    - errors: List of detected grammar errors.
    - rule_corrected: Corrected text after applying predefined rules.
    - misspelled_words: List of tuples (misspelled_word, suggestion) from the spell checker.
    
    Returns:
    - grammar_suggestions: List of grammar error correction suggestions.
    - misspelled_suggestions: List of suggestions for misspelled words.
    """
    grammar_suggestions = []
    misspelled_suggestions = []
    unique_suggestions = set()  # To track unique suggestions for grammar
    
    # Handling grammar or predefined rule errors
    for error in errors:
        suggestion = f"Suggested correction for grammar: {rule_corrected}"
        
        if suggestion not in unique_suggestions:
            grammar_suggestions.append(suggestion)
            unique_suggestions.add(suggestion)  # Track the suggestion to avoid duplicates
    
    # Handling misspelled word corrections
    for misspelled_word, suggestion in misspelled_words:
        if suggestion:
            suggestion_text = f"Suggested correction for '{misspelled_word}': {suggestion}"
        else:
            suggestion_text = f"No suggestion found for misspelled word: '{misspelled_word}'"
        
        misspelled_suggestions.append(suggestion_text)
    
    return grammar_suggestions, misspelled_words, rule_corrected
