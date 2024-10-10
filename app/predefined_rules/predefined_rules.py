import re

def handle_nang_ng(text, pos_tags):
    vowels = 'aeiou'  # Define vowels
    words = text.split()
    corrected_words = []
    
    for i, word in enumerate(words):
        if word == "nang":
            # Handle "nang" as a synonym for "noong" (RBW)
            if pos_tags[i] == 'RBW':  
                corrected_words.append("noong")
            
            # Handle "nang" as a conjunction (CCB, CCT)
            elif pos_tags[i] in ['CCB', 'CCT']:
                corrected_words.append(word)  # Keep "nang" as conjunction
            
            # Handle "nang" as a ligature (CCP)
            elif pos_tags[i] == 'CCP' and i > 0:
                prev_pos = pos_tags[i - 1]
                if prev_pos in ['RB', 'VB', 'JJ']:  # Connecting adverb of manner/intensity to verb/adjective
                    corrected_words.append(word)
                else:
                    corrected_words.append("nang")
        
        elif word == "ng":
            # Handle "ng" as a ligature (CCP or CCB)
            if pos_tags[i] in ['CCP', 'CCB']:
                corrected_words.append(word)  # Keep "ng"
            else:
                corrected_words.append(word)
        
        elif word == "na" and i > 0:  # If the current word is "na" and it's not the first word
            prev_word = words[i - 1]
            if prev_word[-1].lower() in vowels:  # Check if the previous word ends with a vowel
                corrected_word = prev_word + "ng"
                corrected_words[-1] = corrected_word  # Update the last word in corrected_words
            elif prev_word[-1].lower() == 'n':  # Check if the previous word ends with 'n'
                corrected_word = prev_word + "g"
                corrected_words[-1] = corrected_word  # Update the last word in corrected_words
        else:
            corrected_words.append(word)  # Append the word if no correction is made
    
    return ' '.join(corrected_words)

# Load the dictionary from the CSV file
dictionary_file = load_dictionary("data/raw/dictionary.csv")

def separate_mas(text, dictionary=dictionary_file):
    words = text.split()
    for i, word in enumerate(words):
        if word.lower().startswith("mas"):
            remaining = word[3:]
            if remaining.lower() in dictionary:
                words[i] = "mas " + remaining
    return ' '.join(words)

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


def merge_affixes(text, dictionary=dictionary_file):
    words = text.split()
    corrected_words = []
    i = 0
    while i < len(words):
        word = words[i]
        merged = False

        # Check if the word is an affix
        if word.lower() in (prefixes):
            # Ensure there is a next word to merge with
            if i + 1 < len(words):
                next_word = words[i + 1]
                # Merge affix with the next word
                combined_word = word + next_word
                # Check if combined word exists in the dictionary
                if combined_word.lower() in dictionary:
                    corrected_words.append(combined_word)
                    i += 2  # Skip the next word as it's already merged
                    merged = True
        if not merged:
            corrected_words.append(word)
            i += 1
    return ' '.join(corrected_words)

