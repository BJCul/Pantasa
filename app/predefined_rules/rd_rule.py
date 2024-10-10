#main function is rd_interchange(text)
import re

affix_list = {
    "common_prefixes": ["pinang", "nag", "na", "mag", "ma", "i", "i-", "ika-", "isa-", "ipag", "ipang", "ipa", "pag", "pa", "um", "in", "ka", "ni", "pinaka", "pinag", "pina"],
    "prefix_assimilation": ["pang", "pam", "pan", "mang", "mam", "man", "sang", "sam", "san", "sing", "sim", "sin"],
    "common_infixes": ["um", "in"],
    "common_suffixes": ["nan", "han", "hin", "an", "in", "ng"],
    "long_prefixes": ["napag", "mapag", "nakipag", "nakikipag", "makipag", "makikipag", "nakiki", "makiki", "naka", "nakaka"],
    "compound_prefix": ["pinag", "pinagpa", "ipang", "ipinag", "ipinagpa", "nakiki", "makiki", "nakikipag", "napag", "mapag", "nakipag", "makipag", "naka", "maka", "nagpa", "nakaka", "makaka", "nagka", "nagkaka", "napaki", "napakiki", "mapaki", "mapakiki", "paki", "pagka", "pakiki", "pakikipag", "pagki", "pagkiki", "pagkikipag", "ika", "ikapag", "ikapagna", "ikima", "ikapang", "ipa", "ipaki", "ipag", "ipagka", "ipagpa", "ipapang", "makapag", "magkanda", "magkang", "makasing", "maging", "maging", "nakapag", "napaka"]
}

# Combine all prefixes for easy access
prefixes = affix_list["common_prefixes"] + affix_list["long_prefixes"] + affix_list["prefix_assimilation"] + affix_list["compound_prefix"]


# Check if a character is a consonant
def is_consonant(char):
    return char.lower() in "bcdfghjklmnpqrstvwxyz"

# Check if a character is a vowel
def is_vowel(char):
    return char.lower() in "aeiou"

# Check if a word ends with an exception syllable
def ends_with_exception_syllable(word):
    return any(word.endswith(suffix) for suffix in ["ri", "ra", "raw", "ray"])

# Correct din/rin and daw/raw usage based on rules
def correct_din_rin_daw_raw(text):
    words = text.split()
    corrected_words = []
    
    for i, word in enumerate(words):
        if word.lower() in ['din', 'rin', 'daw', 'raw']:
            # Get the previous word if it exists
            if i > 0:
                prev_word = words[i - 1].lower()
                last_char = prev_word[-1]

                # Check if the previous word ends with an exception syllable
                if ends_with_exception_syllable(prev_word):
                    print(f" - Previous word '{prev_word}' ends with exception syllable. Using 'din' or 'daw'.")
                    if word.lower() == 'rin':
                        corrected_words.append('din')
                    elif word.lower() == 'raw':
                        corrected_words.append('daw')
                    else:
                        corrected_words.append(word)
                # Otherwise, apply the vowel/consonant rule
                elif is_vowel(last_char) or last_char in ['w', 'y']:
                    print(f" - Previous word '{prev_word}' ends with a vowel or vowel sound. Using 'rin' or 'raw'.")
                    if word.lower() == 'din':
                        corrected_words.append('rin')
                    elif word.lower() == 'daw':
                        corrected_words.append('raw')
                    else:
                        corrected_words.append(word)
                else:
                    print(f" - Previous word '{prev_word}' ends with a consonant. Using 'din' or 'daw'.")
                    if word.lower() == 'rin':
                        corrected_words.append('din')
                    elif word.lower() == 'raw':
                        corrected_words.append('daw')
                    else:
                        corrected_words.append(word)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)

def find_prefix(word, prefixes):
    """ Find the prefix in the word and return the prefix and the remaining word, keeping original case. """
    
    lower_word = word.lower()  # Convert the word to lowercase for case-insensitive comparison
    
    for prefix in sorted(prefixes, key=len, reverse=True):  # Sort prefixes by length (descending order)
        lower_prefix = prefix.lower()  # Convert prefix to lowercase for comparison
        
        if lower_word.startswith(lower_prefix):  # Compare lowercase word and prefix
            # Find the length of the prefix to return the original case from the word
            original_prefix = word[:len(prefix)]  # Extract the prefix part in the original case
            remaining_word = word[len(prefix):]  # Extract the remaining part of the word in its original case
            return original_prefix, remaining_word  # Return original case for both parts
    
    print(f"No match found for {word}")
    return None, word  # Return None if no match is


# New function to handle transformation of "d" to "r" based on prefix ending
def correct_d_to_r_prefix(text):
    words = text.split()
    corrected_words = []
    
    for word in words:
        # Check if the word has a valid prefix and the remaining part starts with "d"
        prefix, remaining_word = find_prefix(word, prefixes)
        
        if prefix and remaining_word.startswith("d"):
            last_char = prefix[-1]  # Get the last character of the prefix
            
            # Apply the transformation if the prefix ends in a vowel
            if is_vowel(last_char):
                corrected_word = f"{prefix}r{remaining_word[1:]}"  # Replace "d" with "r" after the prefix
                print(f" - Word '{word}' starts with 'd' after a vowel-ending prefix '{prefix}', changing to '{corrected_word}'.")
                corrected_words.append(corrected_word)
            else:
                print(f" - Word '{word}' starts with 'd' after a consonant-ending prefix '{prefix}', keeping as '{word}'.")
                corrected_words.append(word)
        else:
            # No prefix transformation or valid prefix found, keep word as is
            corrected_words.append(word)
    
    return ' '.join(corrected_words)


# Combined function to handle all the rules
def rd_interchange(text):
    # First, apply the din/rin, daw/raw correction
    text = correct_din_rin_daw_raw(text)
    # Then, apply the d-to-r prefix rule
    text = correct_d_to_r_prefix(text)
    return text


