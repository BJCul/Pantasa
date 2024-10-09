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

# Spell Checker function
def spell_checker(input_word, dictionary, max_distance=2):
    suggestions = []
    for word in dictionary:
        distance = levenshtein_distance(input_word, word)
        if distance <= max_distance:
            suggestions.append((word, distance))

    suggestions.sort(key=lambda x: x[1])
    return suggestions

# Load the dictionary from the CSV file
dictionary = load_dictionary("data/raw/dictionary.csv")

# Input word to check
input_word = "byan"

# Get corrections
corrections = spell_checker(input_word, dictionary)

# Display suggestions
if corrections:
    print("Did you mean:")
    for suggestion, dist in corrections:
        print(f"{suggestion} (distance: {dist})")
else:
    print("No suggestions found")
