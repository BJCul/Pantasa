import pandas as pd
import numpy as np

# Step 1: Load dictionary and create index based on first letter
def load_dictionary_with_index(csv_path):
    df = pd.read_csv(csv_path)
    # Assuming the words are in the first column
    words = df.iloc[:, 0].dropna().astype(str).tolist()

    # Create a dictionary where the key is the first letter, and the value is a set of words
    indexed_dict = {}
    for word in words:
        first_letter = word[0].lower()
        if first_letter not in indexed_dict:
            indexed_dict[first_letter] = set()
        indexed_dict[first_letter].add(word.lower())
    
    return indexed_dict

# Step 2: Function to calculate Levenshtein distance
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

# Step 3: Spell Checker function to check each word in a sentence
def spell_check_sentence(sentence, dictionary, max_distance=2):
    words = sentence.split()  # Split sentence into words
    corrected_sentence = []
    
    for word in words:
        word = word.lower()  # Normalize the case
        first_letter = word[0] if word else ''
        
        # Step 4: Check if the word's first letter is in the dictionary
        if first_letter in dictionary:
            # Search within the set of words that start with the same letter
            if word in dictionary[first_letter]:
                corrected_sentence.append(word)  # The word is correct
            else:
                # Step 5: If no exact match, compute Levenshtein distance for suggestions
                suggestions = []
                for dict_word in dictionary[first_letter]:
                    distance = levenshtein_distance(word, dict_word)
                    if distance <= max_distance:
                        suggestions.append((dict_word, distance))
                
                suggestions.sort(key=lambda x: x[1])
                
                # If suggestions exist, mark the word as misspelled
                if suggestions:
                    corrected_sentence.append(f"{suggestions[0][0]}")
                else:
                    corrected_sentence.append(f"<<{word}>>")  # No suggestions, mark as misspelled anyway
        else:
            # If the first letter is not in the dictionary, the word is unknown
            corrected_sentence.append(f"<<{word}>>")
    
    return ' '.join(corrected_sentence)

# Step 6: Main execution to load the dictionary and test the spell checker
if __name__ == "__main__":
    # Load the dictionary with the indexed structure
    dictionary = load_dictionary_with_index("data/raw/dictionary.csv")

    # Test the spell checker with sample sentences
    test_sentence_1 = "ang mga bata ay masaya"
    test_sentence_2 = "kumakai ang mga bata ng mansana sa halamana"

    print("Original:", test_sentence_1)
    print("Corrected:", spell_check_sentence(test_sentence_1, dictionary))

    print("Original:", test_sentence_2)
    print("Corrected:", spell_check_sentence(test_sentence_2, dictionary))
