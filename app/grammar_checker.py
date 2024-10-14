""# Modified pantasa checker function

import re
from app.utils import log_message


def pantasa_checker(input_sentence, jar_path, model_path, rule_bank, pos_tag_dict):
    # Step 1: Preprocess the input text
    log_message("info", "Starting preprocessing")
    tokens = tokenize_sentence(input_sentence)
    pos_tags = pos_tagging(tokens, jar_path=jar_path, model_path=model_path)
    if not pos_tags:
        log_message("error", "POS tagging failed during preprocessing")
        return [], [], []

    # Step 2: Check if words exist in the dictionary and tag those that don't
    log_message("info", "Checking words against the dictionary")
    words = [word for word, _ in pos_tags]
    incorrect_words = check_words_in_dictionary(words)
    log_message("info", f"Incorrect words: {incorrect_words}")

    # Step 3: Apply pre-defined rules before any modification
    log_message("info", "Applying pre-defined rules (pre)")
    pre_rules_corrected_text = apply_predefined_rules_pre(input_sentence)
    log_message("info", f"Text after pre-defined rules (pre): {pre_rules_corrected_text}")

    # Step 4: Check the dictionary again for any remaining incorrect words
    log_message("info", "Re-checking words against the dictionary after pre-defined rules (pre)")
    tokens = tokenize_sentence(pre_rules_corrected_text)
    pos_tags = pos_tagging(tokens, jar_path=jar_path, model_path=model_path)
    words = [word for word, _ in pos_tags]
    incorrect_words = check_words_in_dictionary(words)
    log_message("info", f"Incorrect words after pre-defined rules (pre): {incorrect_words}")

    # Step 5: Spell check the words tagged as incorrect
    log_message("info", "Spell checking incorrect words")
    spell_checked_text = spell_check_incorrect_words(pre_rules_corrected_text, incorrect_words)
    log_message("info", f"Text after spell checking: {spell_checked_text}")

    # Step 6: Apply pre-defined rules after modifications
    log_message("info", "Applying pre-defined rules (post)")
    post_rules_corrected_text = apply_predefined_rules_post(spell_checked_text)
    log_message("info", f"Text after pre-defined rules (post): {post_rules_corrected_text}")

    # Step 7: Retokenize and re-tag the text after modifications
    log_message("info", "Retokenizing and re-tagging after modifications")
    tokens = tokenize_sentence(post_rules_corrected_text)
    pos_tags = pos_tagging(tokens, jar_path=jar_path, model_path=model_path)
    if not pos_tags:
        log_message("error", "POS tagging failed after modifications")
        return [], [], []
    words = [word for word, _ in pos_tags]

    # Step 8: Generate suggestions using n-gram matching
    log_message("info", "Generating suggestions")
    token_suggestions = generate_suggestions(pos_tags)

    # Step 9: Apply POS corrections
    log_message("info", "Applying POS corrections")
    corrected_sentence, word_suggestions = apply_pos_corrections(token_suggestions, pos_tags, pos_tag_dict)
    log_message("info", f"Final corrected sentence: {corrected_sentence}")

    # Return the corrected sentence and any suggestions
    return corrected_sentence, incorrect_words, word_suggestions


def check_words_in_dictionary(words):
    """
    Check if words exist in the dictionary.
    Args:
    - words: List of words to check.
    Returns:
    - List of incorrect words.
    """
    incorrect_words = []
    # Load your dictionary here; for example purposes, we'll use a simple set
    dictionary = load_dictionary()
    for word in words:
        if word.lower() not in dictionary:
            incorrect_words.append(word)
    return incorrect_words

def spell_check_incorrect_words(text, incorrect_words):
    """
    Spell check only the words tagged as incorrect.
    Args:
    - text: Original text.
    - incorrect_words: List of incorrect words.
    Returns:
    - Text after spell checking.
    """
    # Implement spell checking logic here
    # For example, replace incorrect words in text with corrected versions
    corrected_text = text
    for word in incorrect_words:
        # Get suggestions from your spell checker
        suggestions = get_spell_checker_suggestions(word)
        if suggestions:
            # Replace the word with the first suggestion
            corrected_word = suggestions[0]
            corrected_text = corrected_text.replace(word, corrected_word)
            log_message("info", f"Replaced '{word}' with '{corrected_word}'")
        else:
            log_message("warning", f"No suggestions found for '{word}'")
    return corrected_text

def apply_predefined_rules_pre(text):
    """
    Apply predefined rules that should be applied before any modifications.
    Args:
    - text: Input text.
    Returns:
    - Text after applying pre-defined rules.
    """
    # Implement your pre-defined rules here
    # Example: Unmerge words like 'masmaganda' to 'mas maganda'
    corrected_text = text
    # Example rule: Unmerge 'masmaganda' to 'mas maganda'
    corrected_text = re.sub(r'\bmasmaganda\b', 'mas maganda', corrected_text)
    # Add more rules as needed
    return corrected_text

def apply_predefined_rules_post(text):
    """
    Apply predefined rules that should be applied after modifications.
    Args:
    - text: Input text.
    Returns:
    - Text after applying post-defined rules.
    """
    # Implement your post-defined rules here
    # Example: Correct specific grammatical structures
    corrected_text = text
    # Add your rules here
    return corrected_text

def load_dictionary():
    """
    Load the dictionary words into a set for efficient lookup.
    Returns:
    - A set containing all dictionary words.
    """
    # Load your dictionary file here
    # For example purposes, we'll use a hardcoded set
    dictionary = {'mas', 'maganda', 'ka', 'kumain', 'ng', 'mansanas', 'magandang'}
    return dictionary


def get_spell_checker_suggestions(word):
    """
    Get spell checker suggestions for a word.
    Args:
    - word: The word to get suggestions for.
    Returns:
    - A list of suggested corrections.
    """
    # Implement your spell checker logic here
    # For example, return a list of words
    suggestions = []
    # Assuming you have a function spell_check_word(word) that returns suggestions
    suggestions = spell_check_word(word)
    return suggestions

def spell_check_word (word):
    """
    Load the spell checker
    """

    pass
