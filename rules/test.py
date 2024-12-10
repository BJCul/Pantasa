import re

# Updated regex to strictly match abbreviations
ABBREVIATION_REGEX = re.compile(r'\b(?:[A-Z][a-z]*\.|[A-Z]{2,}\.?|[A-Z]\.)\b')

test_row = [
    "Kingpin nga,.' raw ang naging reputasyon ni Mr. Digong."
]

tokens = []

# Tokenize the sentence properly
TOKENIZE_REGEX = re.compile(r"\w+|[^\w\s]")
for sentence in test_row:
    words = TOKENIZE_REGEX.findall(sentence)
    for word in words:
        if ABBREVIATION_REGEX.match(word):
            print(f"{word} is an abbreviation")
            tokens.append(word)
        else:
            print(f"{word} is not an abbreviation")