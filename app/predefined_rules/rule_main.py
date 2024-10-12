import pandas as pd
import sys
import os
from predefined_rules.hyphen_rule import correct_hyphenation
from predefined_rules.rd_rule import rd_interchange
import logging
from preprocess import pos_tagging

logger = logging.getLogger(__name__)

# Add the 'app' directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the necessary modules

from rules.Modules.POSDTagger import pos_tag

def load_dictionary(csv_path):
    df = pd.read_csv(csv_path)
    # Assuming the words are in the first column and filtering out non-string values
    words = df.iloc[:, 0].dropna().astype(str).tolist()
    return words

def handle_nang_ng(text, pos_tags):
    vowels = 'aeiou'  # Define vowels
    words = text.split()
    corrected_words = []
    
    for i, word in enumerate(words):
        # Use the POS tag and lemma to guide the correction
        if word == "nang":
            if pos_tags[i] == 'RBW':  # "nang" as a synonym for "noong" (adverb)
                corrected_words.append("noong")
            elif pos_tags[i] in ['CCB', 'CCT']:  # Conjunction usage of "nang"
                corrected_words.append(word)
            elif pos_tags[i] == 'CCP' and i > 0:  # Ligature usage (connecting adverbs)
                prev_pos = pos_tags[i - 1]
                if prev_pos in ['RB', 'VB', 'JJ']:  # Correct context for ligature
                    corrected_words.append(word)
                else:
                    corrected_words.append("nang")
        elif word == "ng":
            if pos_tags[i] in ['CCP', 'CCB']:
                corrected_words.append(word)
            else:
                corrected_words.append(word)
        elif word == "na" and i > 0:  # Handle "na" as a ligature
            prev_word = words[i - 1]
            if prev_word[-1].lower() in vowels:
                corrected_word = prev_word + "ng"
                corrected_words[-1] = corrected_word  # Update the previous word
            elif prev_word[-1].lower() == 'n':
                corrected_word = prev_word + "g"
                corrected_words[-1] = corrected_word
        else:
            corrected_words.append(word)  # No correction needed

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

def apply_predefined_rules(text):
    pos = pos_tag(text)
    
    logger.info(f"Applying predefined rules to text: {text}")

    rd_correction = rd_interchange(text)
    mas_correction = separate_mas(rd_correction)
    prefix_merged = merge_affixes(mas_correction)
    nang_handled = handle_nang_ng(prefix_merged,pos)
    rule_corrected = correct_hyphenation(nang_handled)

    logger.info(f"Final rule-corrected sentence: {rule_corrected}")


    return rule_corrected


if __name__ == "__main__":
    text = "pinang tiklop maski pagusp"
    corrected_sentence = apply_predefined_rules(text)

    print(f"ff{corrected_sentence}")