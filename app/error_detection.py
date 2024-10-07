# Rule-based grammar error detection

def error_detection(tokens, rule_bank):
    # Compare tokens with rule_bank
    corrections = []
    for i, token in enumerate(tokens):
        word, pos_tag = token
        if (word, pos_tag) not in rule_bank:
            corrections.append((i, word))  # Record the index and word of the error
    return corrections

