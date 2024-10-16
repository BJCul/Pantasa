import pandas as pd
import re

def combined_spell_checker(text, incorrect_words, csv_path="data/raw/dictionary.csv"):
    """
    Spell checks the text, correcting any misspelled words from the incorrect_words list.
    """
    # Load the dictionary
    def load_dictionary(csv_path):
        df = pd.read_csv(csv_path)
        words = df.iloc[:, 0].dropna().astype(str).tolist()
        return words
    
    # Spell check a word
    def spell_check_word(word, dictionary):
        word_lower = word.lower()
        if word_lower in dictionary:
            return word, None
        else:
            suggestions = get_closest_words(word_lower, dictionary)
            if suggestions:
                return word, suggestions[0][0]
            else:
                return word, None
    
    # Find closest words based on Levenshtein distance
    def get_closest_words(word, dictionary, num_suggestions=1):
        word_distances = []
        for dict_word in dictionary:
            distance = weighted_levenshtein_word(word, dict_word)
            word_distances.append((dict_word, distance))
        word_distances.sort(key=lambda x: x[1])
        return word_distances[:num_suggestions]
    
    # Levenshtein distance calculation with weights
    def weighted_levenshtein_word(word1, word2):
        len_word1 = len(word1)
        len_word2 = len(word2)
        distance_matrix = [[0] * (len_word2 + 1) for _ in range(len_word1 + 1)]
        for i in range(len_word1 + 1):
            distance_matrix[i][0] = i
        for j in range(len_word2 + 1):
            distance_matrix[0][j] = j
        
        substitution_weight = 0.8
        insertion_weight = 1.0
        deletion_weight = 1.0

        for i in range(1, len_word1 + 1):
            for j in range(1, len_word2 + 1):
                cost = substitution_weight if word1[i-1] != word2[j-1] else 0
                distance_matrix[i][j] = min(
                    distance_matrix[i-1][j] + deletion_weight,
                    distance_matrix[i][j-1] + insertion_weight,
                    distance_matrix[i-1][j-1] + cost
                )
        return distance_matrix[len_word1][len_word2]
    
    # Load dictionary once
    dictionary = load_dictionary(csv_path)

    # Iterate through each incorrect word in the text
    corrected_text = text
    for word in incorrect_words:
        misspelled_word, corrected_word = spell_check_word(word, dictionary)
        if corrected_word:
            corrected_text = re.sub(r'\b{}\b'.format(re.escape(word)), corrected_word, corrected_text)
            print(f"Replaced '{word}' with '{corrected_word}'")
        else:
            print(f"No suggestions found for '{word}'")
    
    return corrected_text

# Example usage:
text = "This is a sampple text with some missplelled words."
incorrect_words = ["sampple", "missplelled"]
corrected_text = combined_spell_checker(text, incorrect_words)
print(corrected_text)
