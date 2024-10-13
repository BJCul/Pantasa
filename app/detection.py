import logging
from app.utils import log_message

logger = logging.getLogger(__name__)

def detect_errors_from_ngrams(matched_ngrams, ngram_collections):
    """
    Detects errors by comparing the matched n-grams to all generated n-grams from the input sentence.
    
    Args:
    - matched_ngrams: List of matched n-grams and their corresponding pattern IDs.
    - ngram_collections: All dynamically generated n-grams from the input sentence.
    
    Returns:
    - error_detected: Boolean flag if an error is detected.
    - unmatched_ngrams: List of unmatched n-grams (if any).
    """
  
    # Flatten the list of n-grams generated from the input sentence
    all_ngrams = []
    for ngram_size, ngrams in ngram_collections.items():
        all_ngrams.extend(ngrams)

    # Extract the n-grams that matched
    matched_ngram_list = [match['ngram'] for match in matched_ngrams]

    # Find unmatched n-grams by comparing all n-grams to the matched n-grams
    unmatched_ngrams = [ngram for ngram in all_ngrams if ngram not in matched_ngram_list]
    
    # If there are any unmatched n-grams, we detect an error
    error_detected = len(unmatched_ngrams) > 0
    log_message('info', f"Unmatched n-grams found: {unmatched_ngrams}" if error_detected else "No errors found.")
    
    return error_detected, unmatched_ngrams

# Example usage within n-gram processing
def handle_errors(matched_ngrams, ngram_collections):
    """
    Handles the error detection process based on the matched n-grams.
    
    Args:
    - matched_ngrams: List of matched n-grams and their corresponding pattern IDs.
    - ngram_collections: All dynamically generated n-grams from the input sentence.
    
    Returns:
    - detected_errors: List of unmatched n-grams (errors).
    """
    error_detected, unmatched_ngrams = detect_errors_from_ngrams(matched_ngrams, ngram_collections)
    if error_detected:
        logger.info(f"Errors detected: {unmatched_ngrams}. Proceeding to rule checking.")
    else:
        logger.info("No errors detected.")
    
    return unmatched_ngrams
