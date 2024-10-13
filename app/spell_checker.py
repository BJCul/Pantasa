import pandas as pd
import numpy as np
from app.utils import weighted_levenshtein

# Load dictionary from CSV
def load_dictionary(csv_path):
    df = pd.read_csv(csv_path)
    # Assuming the words are in the first column and filtering out non-string values
    words = df.iloc[:, 0].dropna().astype(str).tolist()
    return words

# Load the dictionary from the CSV file
dictionary_file = load_dictionary("data/raw/dictionary.csv")

# Spell Checker function to check each word in a sentence
def spell_check_sentence(sentence, dictionary=dictionary_file, max_distance=2):
    words = sentence.split()  # Split sentence into words
    misspelled_words = []  # List to store misspelled words and suggestions
    corrected_sentence = []  # List to store the corrected sentence
    
    for word in words:
        # Check if the word is already in the dictionary
        if word in dictionary:
            corrected_sentence.append(word)  # The word is correct, add it to the corrected sentence
        else:
            # If no exact match, compute Levenshtein distance
            suggestions = []
            for dict_word in dictionary:
                distance = weighted_levenshtein(word, dict_word)
                if distance <= max_distance:
                    suggestions.append((dict_word, distance))
            
            suggestions.sort(key=lambda x: x[1])
            
            # If suggestions exist, store the misspelled word and suggested correction
            if suggestions:
                corrected_word = suggestions[0][0]
                corrected_sentence.append(corrected_word)
                misspelled_words.append((word, corrected_word))  # Store the misspelled word and the best suggestion
            else:
                corrected_sentence.append(word)  # No suggestions, keep the word as is
                misspelled_words.append((word, None))  # Store the misspelled word with no suggestion
    
    # Join the corrected sentence list into a string
    corrected_sentence_str = ' '.join(corrected_sentence)
    
    return misspelled_words, corrected_sentence_str

if __name__ == "__main__":
    # Load the dictionary
    dictionary = load_dictionary("data/raw/dictionary.csv")

    # Test the spell checker with a sample sentence
    test_sentence_1 = "ang mga bata ay mastaya"
    test_sentence_2 = "kumakain ang mga bata ng mansana"

    print("Original:", test_sentence_1)
    print("Corrected:", spell_check_sentence(test_sentence_1, dictionary))

    print("Original:", test_sentence_2)
    print("Corrected:", spell_check_sentence(test_sentence_2, dictionary))

    