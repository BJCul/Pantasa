# Preprocessing functions: Tokenization, POS tagging, Lemmatization

def tokenize(text):
    # Tokenization logic
    return text.split()

def pos_tagging(tokens):
    # POS tagging logic (use a pre-trained model or rules)
    return [(token, "POS_TAG") for token in tokens]

def lemmatize(token):
    # Lemmatization logic (reduce to root form)
    return token.lower()
