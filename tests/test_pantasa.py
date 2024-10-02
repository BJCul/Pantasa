import nltk
import re
import numpy as np

# Ensure the necessary NLTK data packages are downloaded
nltk.download('punkt')

# Placeholder functions for Tagalog POS tagging and lemmatization
def pos_tag(tokens):
    # In practice, replace this with FSPOST or an appropriate Tagalog POS tagger
    return [(token, 'POS') for token in tokens]

def lemmatize(token, pos_tag):
    # In practice, replace this with Morphinas or an appropriate Tagalog lemmatizer
    return token

# Function to check if a word is valid (exists in the dictionary)
def is_valid_word(word):
    # Placeholder: Replace with actual dictionary lookup
    return True

# Predefined rules based on common Tagalog writing problems
def apply_predefined_rules(tokens):
    corrected_tokens = []
    i = 0
    while i < len(tokens):
        word, pos = tokens[i]
        # Rule 1: Use of 'na' / '-ng'
        if word == 'na' and i > 0:
            prev_word, prev_pos = tokens[i - 1]
            if prev_word.endswith(('a', 'e', 'i', 'o', 'u', 'n')):
                if prev_word.endswith(('a', 'e', 'i', 'o', 'u')):
                    new_word = prev_word + 'ng'
                else:
                    new_word = prev_word + 'g'
                corrected_tokens[-1] = (new_word, prev_pos)
                i += 1
                continue
        # Rule 2: Separating the prefix 'mas' from verbs
        elif word.lower().startswith('mas') and len(word) > 3:
            root_word = word[3:]
            if is_valid_word(root_word):
                corrected_tokens.append(('mas', pos))
                corrected_tokens.append((root_word, pos))
                i += 1
                continue
        # Rule 3: Combining incorrectly separated affixes
        elif word.lower() in ['pinag', 'pag', 'mag', 'nag', 'um', 'ma', 'ka', 'in', 'i'] and i + 1 < len(tokens):
            next_word, next_pos = tokens[i + 1]
            new_word = word + next_word
            if is_valid_word(new_word):
                corrected_tokens.append((new_word, pos))
                i += 2
                continue
        # Rule 4: Removing incorrect hyphenations
        elif '-' in word:
            new_word = word.replace('-', '')
            if is_valid_word(new_word):
                corrected_tokens.append((new_word, pos))
                i += 1
                continue
            else:
                corrected_tokens.append((word, pos))
                i += 1
                continue
        else:
            corrected_tokens.append((word, pos))
            i += 1
    return corrected_tokens

# Weighted Levenshtein distance function
def weighted_levenshtein_distance(s1, s2, weights):
    len1 = len(s1)
    len2 = len(s2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i  # Deletion
    for j in range(len2 + 1):
        dp[0][j] = j  # Insertion
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = get_substitution_cost(s1[i - 1][0], s2[j - 1][0], weights)
            dp[i][j] = min(
                dp[i - 1][j] + weights.get('Delete', 1.0),      # Deletion
                dp[i][j - 1] + weights.get('Insert', 1.0),      # Insertion
                dp[i - 1][j - 1] + cost                         # Substitution
            )
    return dp[len1][len2]

def get_substitution_cost(token1, token2, weights):
    if token1 == token2:
        return 0
    # Determine the type of error
    error_type = classify_error(token1, token2)
    return weights.get(error_type, 1.0)

def classify_error(word1, word2):
    # Placeholder function to classify the error type
    # Implement logic based on specific error types
    return 'Substitution'

# Error detection and correction function
def error_detection_and_correction(tokens, rule_bank, weights):
    min_distance = float('inf')
    best_match = tokens
    for pattern in rule_bank:
        distance = weighted_levenshtein_distance(tokens, pattern, weights)
        if distance < min_distance:
            min_distance = distance
            best_match = pattern
    corrections = []
    if min_distance > 0:
        # Generate correction suggestions
        for i in range(len(tokens)):
            if i < len(best_match) and tokens[i][0] != best_match[i][0]:
                corrections.append((i, best_match[i][0]))
    return corrections

# Iterative correction system
def iterative_correction_system(text, rule_bank, weights):
    corrected_text = text
    iterations = 0
    max_iterations = 10
    while iterations < max_iterations:
        iterations += 1
        sentences = nltk.sent_tokenize(corrected_text)
        total_corrections = 0
        corrected_sentences = []
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            tagged_tokens = pos_tag(tokens)
            lemmatized_tokens = [(lemmatize(word, pos), pos) for word, pos in tagged_tokens]
            # Apply predefined rules
            lemmatized_tokens = apply_predefined_rules(lemmatized_tokens)
            # Error detection and correction
            corrections = error_detection_and_correction(lemmatized_tokens, rule_bank, weights)
            total_corrections += len(corrections)
            # Apply corrections
            for idx, new_word in corrections:
                lemmatized_tokens[idx] = (new_word, lemmatized_tokens[idx][1])
            corrected_sentence = ' '.join([word for word, pos in lemmatized_tokens])
            corrected_sentences.append(corrected_sentence)
        new_corrected_text = ' '.join(corrected_sentences)
        if total_corrections == 0 or new_corrected_text == corrected_text:
            break
        corrected_text = new_corrected_text
    return corrected_text

# Main function
def main():
    # Sample rule bank (patterns of correct sentences)
    # In practice, this should be generated by the hybrid n-gram learning module
    rule_bank = [
        [('Ang', 'DET'), ('bata', 'NN'), ('ay', 'VBZ'), ('kumain', 'VB'), ('ng', 'IN'), ('mangga', 'NN')],
        [('Siya', 'PRP'), ('ay', 'VBZ'), ('tumakbo', 'VB'), ('sa', 'IN'), ('park', 'NN')],
        # Add more patterns as needed
    ]

    # Weights for the weighted Levenshtein edit distance algorithm
    weights = {
        'Wrong use of ‘nang’ vs. ‘ng’': 0.8,
        'Wrong use of enclitics': 0.8,
        'Wrong use of ‘ang’ and ‘ng’ pronouns': 0.8,
        'Wrong Words Different  POS': 0.95,
        'Wrong Use of Space (remove)': 0.7,
        'Wrong Use of Space (split)': 0.7,
        'Morphological Errors': 0.6,
        'Improper Word Casing': 0.65,
        'Wrong Use of Punctuation Marks': 0.8,
        'Missing Words': 1.0,
        'Duplicate/ Unnecessary Words': 1.0,
        'Insert': 1.0,
        'Delete': 1.0,
        'Substitution': 1.0
    }

    # Input text
    text = "Masaya na bata ang kumain na mangga."

    # Run the iterative correction system
    corrected_text = iterative_correction_system(text, rule_bank, weights)

    print("Original Text:", text)
    print("Corrected Text:", corrected_text)

if __name__ == "__main__":
    main()
