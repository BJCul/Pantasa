import pandas as pd
import sys
import os
from app.predefined_rules.hyphen_rule import correct_hyphenation
from app.predefined_rules.rd_rule import rd_interchange
import logging
from app.preprocess import pos_tagging

logger = logging.getLogger(__name__)

# Add the 'app' directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the necessary 
from rules.Modules.POSDTagger import pos_tag

# Hierarchical POS Tag Dictionary
hierarchical_pos_tags = {
    "NN.*": ["NNC", "NNP", "NNPA", "NNCA"],
    "PR.*": ["PRS", "PRP", "PRSP", "PRO", "PRQ", "PRQP", "PRL", "PRC", "PRF", "PRI"],
    "DT.*": ["DTC", "DTCP", "DTP", "DTPP"],
    "CC.*": ["CCT", "CCR", "CCB", "CCA", "CCP", "CCU"],
    "LM": [],
    "TS": [],
    "VB.*": ["VBW", "VBS", "VBH", "VBN", "VBTS", "VBTR", "VBTF", "VBTP", "VBAF", "VBOF", "VBOB", "VBOL", "VBOI", "VBRF"],
    "JJ.*": ["JJD", "JJC", "JJCC", "JJCS", "JJCN", "JJN"],
    "RB.*": ["RBD", "RBN", "RBK", "RBP", "RBB", "RBR", "RBQ", "RBT", "RBF", "RBW", "RBM", "RBL", "RBI", "RBJ", "RBS"],
    "CD.*": ["CDB"],
    "FW": [],
    "PM.*": ["PMP", "PME", "PMQ", "PMC", "PMSC", "PMS"]
}

def load_dictionary(csv_path):
    df = pd.read_csv(csv_path)
    # Assuming the words are in the first column and filtering out non-string values
    words = df.iloc[:, 0].dropna().astype(str).tolist()
    return words

def handle_nang_ng(text, pos_tags):
    vowels = 'aeiou'  # Define vowels
    words = text.split()
    corrected_words = []
    
    # pos_tags is expected to be a list of tuples: [(word, pos), (word, pos), ...]
    # We'll extract the POS for each token using pos_tags[i][1].
    
    for i, word in enumerate(words):
        # Safely get previous and next indices
        prev_i = i - 1
        next_i = i + 1
        
        # Extract current pos if available
        current_pos = pos_tags[i][1] if i < len(pos_tags) else ''
        
        # Extract previous and next pos if they exist
        prev_pos = pos_tags[prev_i][1] if prev_i >= 0 and prev_i < len(pos_tags) else ''
        next_pos = pos_tags[next_i][1] if next_i < len(pos_tags) else ''
        
        if word == "ng":
            # Handle "ng" as a ligature connecting verbs to adverbs
            if prev_i >= 0 and next_i < len(pos_tags) and prev_pos.startswith('VB') and next_pos.startswith('RB'):
                corrected_words.append("ng")
                print("'ng' kept as ligature (connecting verb and adverb)")
            
            # Handle "ng" as other use cases, such as between nouns and adjectives
            elif next_i < len(pos_tags) and next_pos.startswith('JJ'):
                corrected_words.append("ng")
                print("'ng' kept as ligature (connecting noun and adjective)")
            
            else:
                corrected_words.append(word)

        elif word == "nang":
            # Case 1: "nang" as a coordinating conjunction
            if current_pos == 'CCB' and prev_i >= 0:
                corrected_words.append("ng")
                print("'nang' corrected to 'ng' (coordinating conjunction CCB)")
            
            # Case 2: Ligature use (connecting an adjective or a verb to a noun/adjective/adverb)
            elif prev_i >= 0 and prev_pos in ['JJ', 'VB']:
                corrected_words.append("ng")
                print("'nang' corrected to 'ng' (ligature for adjective/verb)")
            
            # Case 3: Check if "nang" is followed by an adjective (often requires ligature)
            elif next_i < len(pos_tags) and next_pos.startswith('JJ'):
                corrected_words.append("ng")
                print("'nang' corrected to 'ng' (followed by adjective JJ)")
            
            # Case 4: Check if "nang" is tagged as adverb (RBW) but acts as a ligature
            elif current_pos == 'RBW' and prev_i >= 0 and prev_pos in ['NNC', 'NNP']:
                corrected_words.append("ng")
                print("'nang' corrected to 'ng' (acting as ligature with noun)")
            
            # Case 5: Other valid use cases (e.g., adverbial phrases)
            else:
                corrected_words.append(word)
                print("'nang' kept unchanged (other use cases)")
            
        else:
            corrected_words.append(word)  # Append the word if no correction is made
            print(f"No correction needed for '{word}'")
    
    corrected_text = ' '.join(corrected_words)
    print(f"\nFinal corrected text: {corrected_text}")
    return corrected_text

# Load the dictionary from the CSV file
dictionary_file = load_dictionary("data/raw/dictionary.csv")

def separate_mas(text, dictionary=dictionary_file):
    words = text.split()
    for i, word in enumerate(words):
        if word.lower().startswith("mas"):
            remaining = word[3:]
            # Only unmerge if the word is not in the dictionary and the remaining part is a valid word
            if word not in dictionary and remaining in dictionary:
                words[i] = "mas " + remaining
    return ' '.join(words)

affix_list = {
    "common_prefixes": ["pinaka","pinang", "nag", "na", "mag", "ma", "i", "i-", "ika-", "isa-", "ipag", "ipang", "ipa", "pag", "pa", "um", "in", "ka", "ni", "pinaka", "pinag", "pina"],
    "prefix_assimilation": ["pang", "pam", "pan", "mang", "mam", "man", "sang", "sam", "san", "sing", "sim", "sin"],
    "common_infixes": ["um", "in"],
    "common_suffixes": ["nan", "han", "hin", "an", "in", "ng"],
    "long_prefixes": ["napag", "mapag", "nakipag", "nakikipag", "makipag", "makikipag", "nakiki", "makiki", "naka", "nakaka"],
    "compound_prefix": ["pinag", "pinagpa", "ipang", "ipinag", "ipinagpa", "nakiki", "makiki", "nakikipag", "napag", "mapag", "nakipag", "makipag", "naka", "maka", "nagpa", "nakaka", "makaka", "nagka", "nagkaka", "napaki", "napakiki", "mapaki", "mapakiki", "paki", "pagka", "pakiki", "pakikipag", "pagki", "pagkiki", "pagkikipag", "ika", "ikapag", "ikapagna", "ikima", "ikapang", "ipa", "ipaki", "ipag", "ipagka", "ipagpa", "ipapang", "makapag", "magkanda", "magkang", "makasing", "maging", "maging", "nakapag", "napaka"]
}

# Combine all prefixes for easy access
prefixes = affix_list["common_prefixes"] + affix_list["long_prefixes"] + affix_list["prefix_assimilation"] + affix_list["compound_prefix"]

merge_affix_pos = hierarchical_pos_tags['VB.*'] + hierarchical_pos_tags['NN.*'] + hierarchical_pos_tags['JJ.*']

def merge_affixes(text, pos_tags, dictionary=dictionary_file):
    words = text.split()
    corrected_words = []
    i = 0

    while i < len(words):
        word = words[i]
        merged = False
        
        # Check if the word is an affix
        if word.lower() in prefixes:
            # Ensure there is a next word to merge with
            if i + 1 < len(words):
                next_word = words[i + 1]
                next_word_pos = pos_tags[i + 1]  # Get the POS tag of the next word
                
                # Only merge if the next word's POS tag is in mergee_affix_pos list
                if next_word_pos in merge_affix_pos:
                    # Merge affix with the next word
                    combined_word = word + next_word
                    # Check if the combined word exists in the dictionary
                    if combined_word.lower() in dictionary:
                        corrected_words.append(combined_word)
                        i += 2  # Skip the next word as it's already merged
                        merged = True
        
        if not merged:
            corrected_words.append(word)
            i += 1
    
    return ' '.join(corrected_words)

def apply_predefined_rules_pre(text):
    pos = pos_tagging(text)
    rd_correction = rd_interchange(text)
    mas_correction = separate_mas(rd_correction)
    rule_corrected = handle_nang_ng(mas_correction,pos)

    return rule_corrected

def apply_predefined_rules_post(text):
    pos = pos_tagging(text)

    rd_correction = rd_interchange(text)
    prefix_merged = merge_affixes(rd_correction, pos)
    nang_ng_correction = handle_nang_ng(prefix_merged,pos)
    rule_corrected = correct_hyphenation(nang_ng_correction)

    return rule_corrected

def apply_predefined_rules(text):
    pos = pos_tagging(text)

    mas_correction = separate_mas(text, pos)
    prefix_merged = merge_affixes(mas_correction, pos)
    nang_handled = handle_nang_ng(prefix_merged, pos)
    rule_corrected = correct_hyphenation(nang_handled)

    logger.info(f"Final rule-corrected sentence: {rule_corrected}")

    return rule_corrected


if __name__ == "__main__":
    text = "pinang tiklop maski pagusp"
    corrected_sentence = apply_predefined_rules_pre(text)

    print(f"Corrected sentence: {corrected_sentence}")

    # Test case
    test = "tumakbo ng mabilis"
    pos = pos_tagging(test).split()  # Split the string into a list
    print(test)
    print(pos)
    print(handle_nang_ng(test, pos))
