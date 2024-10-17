import pandas as pd
import numpy as np
from rapidfuzz.distance import Levenshtein  # Use RapidFuzz's Levenshtein
from app.utils import weighted_levenshtein  # Assuming you have custom utilities here

# Load dictionary from CSV
def load_dictionary(csv_path):
    df = pd.read_csv(csv_path)
    # Assuming the words are in the first column and filtering out non-string values
    words = df.iloc[:, 0].dropna().astype(str).tolist()
    return words

# Load the dictionary from the CSV file
dictionary_file = load_dictionary("data/raw/dictionary.csv")

# Define the weighted Levenshtein function using RapidFuzz with 1-1-2 weights
def weighted_levenshtein(word1, word2):
    return Levenshtein.distance(word1, word2, weights=(1, 1, 3))

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

def spell_check_word(word):
    """
    Check if the word is spelled correctly and provide a correction if not.
    Args:
    - word: The word to check.
    Returns:
    - A tuple (misspelled_word, corrected_word). If the word is correct, corrected_word will be None.
    """
    dictionary = load_dictionary("data/raw/dictionary.csv")
    word_lower = word.lower()
    
    if word_lower in dictionary:
        # Word is spelled correctly
        return word, None
    else:
        # Word is misspelled; find the closest match
        suggestions = get_closest_words(word_lower, dictionary)
        if suggestions:
            corrected_word = suggestions[0][0]  # Get the best suggestion
            return word, corrected_word
        else:
            # No suggestions found
            return word, None

def get_closest_words(word, dictionary, num_suggestions=1):
    """
    Find the closest words in the dictionary to the input word using Levenshtein distance.
    Args:
    - word: The misspelled word.
    - dictionary: A set of correct words.
    - num_suggestions: The number of suggestions to return.
    Returns:
    - A list of tuples (word, distance).
    """
    word_distances = []
    for dict_word in dictionary:
        distance = weighted_levenshtein(word, dict_word)
        word_distances.append((dict_word, distance))
    
    # Sort the words by distance
    word_distances.sort(key=lambda x: x[1])
    
    # Return the top suggestions
    return word_distances[:num_suggestions]

if __name__ == "__main__":
    # Load the dictionary
    dictionary = load_dictionary("data/raw/dictionary.csv")

    # Test the spell checker with a sample sentence
    test_sentence_1 = "ang mga bat"
    test_sentence_2 = "kumakain ang mga bata ng mansana"

    print("Original:", test_sentence_1)
    print("Corrected:", spell_check_word(test_sentence_1))

    print("Original:", test_sentence_2)
    print("Corrected:", spell_check_sentence(test_sentence_2, dictionary))
