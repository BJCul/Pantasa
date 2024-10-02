# Rule-based grammar error detection

def detect_errors(tokens):
    errors = []
    for token in tokens:
        # Add grammar error detection logic
        if token == "example_error":
            errors.append((token, "Grammar error found"))
    return errors
