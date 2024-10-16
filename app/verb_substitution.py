import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Add the 'app' directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary modules
from rules.Modules.POSDTagger import pos_tag

# Define the mapping for the verb affixes based on the POS tag table you provided
affix_table = {
    "VBW": {  # Neutral/Infinitive
        "prefix": "mag-", "suffix": "-in"
    },
    "VBTS": {  # Time Past (Perfective)
        "VBAF": "nag-", "VBOF": "in-", "VBOB": "i-", "VBOL": "pag-", "VBOI": "ipina-", "VBOB": "in-", "VBRF": "pina-"
    },
    "VBTR": {  # Time Present (Imperfective)
        "VBAF": "nag-", "VBOF": "in-", "VBOB": "i-", "VBOL": "in-", "VBOI": "ipina-", "VBOB": "in-", "VBRF": "pina-"
    },
    "VBTF": {  # Time Future (Contemplative)
        "VBAF": "mag-", "VBOF": "-in", "VBOB": "i-", "VBOL": "pag-", "VBOI": "ipa-", "VBRF": "pina-"
    },
    # Affixes when only a trigger tag is present
    "VBAF": {  # Actor Focus
        "default": "um-",  # Use "um-" for general cases
    },
    "VBOF": {  # Object Focus
        "default": "-in",  # Use "-in" for general cases
    },
    "VBOL": { #Locative Focus
        "default": "-an",  # Use "-an" for general cases
    },
    "VBOB": {  # Benefactive Focus
        "default": "i-",  # Use "i-" for general cases
    },
    "VBOI": {  # Instrumental Focus
        "default": "ipa-",  # Use "ipang-" for general cases
    },
    "VBRF": {  # Referential/Measurement Focus
        "default": "pina-",  # Use "pinag-" for general cases
    }
}

long_vowels = "iua"
vowels = "eo" + long_vowels

# Function to handle recently completed (katatapos) tense (VBTP)
def apply_recently_completed_affix(verb_root):
    """
    Handles the application of the recently completed (katatapos) affix.
    - The 'ka-' prefix is added, and the first syllable or first two letters of the verb root are repeated.
    """
   
    # If the first two characters are consonant-vowel, repeat the first syllable
    if len(verb_root) > 1 and verb_root[0] not in vowels:
        repeated_part = verb_root[:2]  # First consonant and vowel
    else:
        repeated_part = verb_root[0]  # Repeat only the first consonant
    
    return f"ka{repeated_part}{verb_root}"


# Function to handle locative focus affixes based on verb ending
def apply_suffix(verb_root):
    print(f"DEBUG: Starting apply_suffix with verb_root: {verb_root}")

    # Open the file and search for the keyword
    row = None  # Initialize row
    print(f"DEBUG: Searching in f7xdict_core-1.0.txt for {verb_root}")
    with open('data/raw/phoneme/F7Xdict-main/f7xdict_core-1.0.txt', 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, start=1):
            if verb_root in line:
                row = line.strip()  # Remove newline characters
                break  # Stop once we've found the row containing the keyword
    
    if row is None:
        print(f"DEBUG: Searching in f7xdict_noncore-1.0.txt for {verb_root}")
        with open('data/raw/phoneme/F7Xdict-main/f7xdict_noncore-1.0.txt', 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, start=1):
                if verb_root in line:
                    row = line.strip()  # Remove newline characters
                    break  # Stop once we've found the row containing the keyword
    
    if row is None:
        return verb_root  # Return the original verb_root if not found

    phoneme_dict = row.split('\t')
    phonemes = phoneme_dict[-1].split()

    verb = verb_root

    # Check phonemes and apply transformations
    if len(phonemes[-1]) == 1:  # Last letter is NOT a vowel
        phoneme = phonemes[-1]
        stress = phoneme[-1]
        if stress == "1":
            verb = verb_root[-2:] + verb_root[:-1]  # Removes the last vowel
        else:
            if len(phonemes[-4]) != 2:
                if verb[-5] in long_vowels:
                    if verb[-5] == "e":
                        verb = verb_root[-2:] + "i" + verb_root[:-1]
                    if verb[-5] == "o":
                        verb = verb_root[-2:] + "u" + verb_root[:-1]
    elif len(phonemes[-1]) == 2:  # Phoneme is a vowel with stress
        phoneme = phonemes[-1]
        stress = phoneme[-1]
        if stress == "1":
            verb = verb_root[:-1]  # Removes the last vowel
        else:
            if len(phonemes[-3]) != 2:
                if verb[-4] in long_vowels:
                    if verb[-4] == "e":
                        verb = verb_root[-1:] + "i"
                    if verb[-4] == "o":
                        verb = verb_root[-1:] + "u"
    elif len(phonemes[-1]) == 3:  # When a verb ends in "y"
        phoneme = phonemes[-1]
        stress = phoneme[-1]
        if stress == "1":
            verb = verb_root[:-2] + verb_root[-1:] # Removes the vowel before the 'y'

    # Apply additional transformations
    if verb in vowels and phonemes[-1] != "Q":
        verb = verb + "h"


    if verb[-1] == "d":
        verb = verb[:-1] + "r"

    return verb

    


# Function to handle the CCP (ligature) rule
def apply_ligature_rule(verb):
    """
    Applies the ligature rule based on the verb's ending:
    - 'N-ng' for vowel-ending words
    - '-g' for words ending in 'n'
    - 'na' for words ending in other consonants
    """
    
    # Check the last character of the verb
    last_char = verb[-1].lower()
    
    if last_char in vowels:
        return verb + "ng"
    elif last_char == 'n':
        return verb + "g"
    else:
        return verb + " na"

# Function to handle "um-" and "in-" affix exceptions
def apply_um_in_affix(affix, verb_root):
    """
    Handles the placement of the "um-" and "in-" affixes.
    - If the verb starts with a consonant, place the affix after the first letter.
    - If the verb starts with a vowel, place the affix before the word.
    """
    # If the first letter is a vowel, place the affix before the root

    if verb_root[0] in vowels:
        return affix + verb_root
    elif verb_root[1] not in vowels:
        # Place the affix after the second letter if the first consonant is a pair of consonant
        return verb_root[:2] + affix + verb_root[2:]
    else:
        # Place the affix after the first letter
        return verb_root[0] + affix + verb_root[1:]

# Function to conjugate verbs based on POS tags (including single trigger tags)
def conjugate_tagalog_verb(verb_root, pos_tag):
    """
    Conjugates a Tagalog verb based on its POS tag, handling both conjoined and single trigger tags.
    
    :param verb_root: The root form of the verb (e.g., 'kain').
    :param pos_tag: The conjoined or single POS tag (e.g., 'VBTR_VBOF' or 'VBAF').
    :return: The conjugated verb.
    """
    
    # Early check for NoneType pos_tag
    if pos_tag is None:
        logging.error(f"POS tag is None for verb '{verb_root}'")
        return verb_root  # Return the original verb_root if no POS tag is given

    # Split the conjoined POS tag by the underscore, if present
    pos_tags = pos_tag.split("_")
    
    if not pos_tags:
        logging.error(f"POS tag could not be split for verb '{verb_root}'")
        return verb_root  # Return original verb_root if splitting fails
    

    has_ligature = "CCP" in pos_tags

    # Apply the ligature rule if CCP is present
    if "CCP" in pos_tags:
        has_ligature = "CCP" in pos_tags
        pos_tags.remove("CCP")  # Properly remove CCP without affecting pos_tags
    
    # Handle Special Circumstances 
    if "VBTP" in pos_tags:
        return apply_recently_completed_affix(verb_root)

    # For repeating the first syllable (future and present tenses)
    if ("VBTF" in pos_tags) or ("VBTR" in pos_tags):
        if verb_root[0] in vowels:
            verb_root = verb_root[0] + verb_root
        elif verb_root[1] not in vowels:  # Double consonant handling
            verb_root = verb_root[:3] + verb_root
        else:
            verb_root = verb_root[:2] + verb_root

    # Handle nasal consonant (Instrumental/Referential focus)
    if "VBOI" in pos_tags or "VBRF" in pos_tags:
        if verb_root[0] in ["d", "l", "t"]:
            verb_root = "n" + verb_root
        elif verb_root[0] in ["b", "p"]:
            verb_root = "m" + verb_root
        elif verb_root[0] in vowels:
            verb_root = "ng-" + verb_root
        else:
            verb_root = "ng" + verb_root
    
    conjugated_verb = verb_root  # Initialize with root form

    # Handle single trigger tag cases
    if len(pos_tags) == 1:
        trigger_tag = pos_tags[0]  # This is the trigger tag
        if trigger_tag in affix_table and "default" in affix_table[trigger_tag]:
            affix = affix_table[trigger_tag]["default"]

            # Apply the affix and handle the "um-" or "in-" exception
            if affix in ["um-", "in-"]:
                conjugated_verb = apply_um_in_affix(affix.rstrip('-'), verb_root)

            elif affix.startswith("-"):  # If it's a suffix
                if affix == "-in":
                    if verb_root[-1] == "i" or verb_root[-2] == "i":
                        affix = "an"
                        verb_root = apply_suffix(verb_root)
                        conjugated_verb = verb_root.lstrip('-') + affix
                        
                else:        
                    verb_root = apply_suffix(verb_root)
                    conjugated_verb = verb_root.lstrip('-') + affix

            else:  # If it's a prefix
                if verb_root[0] in vowels:
                    conjugated_verb = affix.rstrip('-') + "-" + verb_root
                else:
                    conjugated_verb = affix.rstrip('-') + verb_root

    # Handle two tags (tense + trigger)
    elif len(pos_tags) == 2:
        tense_tag, focus_tag = pos_tags
        if tense_tag in affix_table and focus_tag in affix_table[tense_tag]:
            affix = affix_table[tense_tag][focus_tag]

            if focus_tag ==  "VBOL":
                loc_affix = affix_table[focus_tag]["default"]
                verb_root = apply_suffix(verb_root) + loc_affix.lstrip('-')  


            if affix in ["um-", "in-"]:
                conjugated_verb = apply_um_in_affix(affix.rstrip('-'), verb_root)

            elif affix.startswith("-"):  # If it's a suffix
                verb_root = apply_suffix(verb_root)
                conjugated_verb = verb_root + affix.lstrip('-')
                
            else:  # If it's a prefix
                if verb_root[0] in vowels:
                    conjugated_verb = affix.rstrip('-') + "-" + verb_root
                else:
                    conjugated_verb = affix.rstrip('-') + verb_root
       
    
    # Apply the ligature rule if CCP was present
    if has_ligature:
        conjugated_verb = apply_ligature_rule(conjugated_verb)
    
    # Return the final conjugated verb
    return conjugated_verb

