import pandas as pd
import numpy as np

# Load dictionary from CSV
def load_dictionary(csv_path):
    df = pd.read_csv(csv_path)
    # Assuming the words are in the first column and filtering out non-string values
    words = df.iloc[:, 0].dropna().astype(str).tolist()
    return words

# Function to calculate Levenshtein distance
def levenshtein_distance(word1, word2):
    len1, len2 = len(word1), len(word2)
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,  # Deletion
                               dp[i][j - 1] + 1,  # Insertion
                               dp[i - 1][j - 1] + 1)  # Substitution

    return dp[len1][len2]

# Load the dictionary from the CSV file
dictionary_file = load_dictionary("data/raw/dictionary.csv")

# Spell Checker function to check each word in a sentence
def spell_check_sentence(sentence, dictionary=dictionary_file, max_distance=2):
    words = sentence.split()  # Split sentence into words
    corrected_sentence = []
    
    for word in words:
        # Check if the word is already in the dictionary
        if word in dictionary:
            corrected_sentence.append(word)  # The word is correct
        else:
            # If no exact match, compute Levenshtein distance
            suggestions = []
            for dict_word in dictionary:
                distance = levenshtein_distance(word, dict_word)
                if distance <= max_distance:
                    suggestions.append((dict_word, distance))
            
            suggestions.sort(key=lambda x: x[1])
            
            # If suggestions exist, mark the word as misspelled
            if suggestions:
                corrected_sentence.append(f"<<{word}>>")
            else:
                corrected_sentence.append(f"<<{word}>>")  # No suggestions, mark as misspelled anyway

    return ' '.join(corrected_sentence)

