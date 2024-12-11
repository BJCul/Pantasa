import subprocess
import re

punctuation = '.!?,;:—"\'()\[\]{}'
brackets = {
    "-LRB-":"(", "-RRB-":")",
    "-LSB-":"[","-RSB-":"]",
    "-LCB-":"{","-RCB-":"}"
    }
terminating_tags = {"PMP", "PME", "PMQ", "PMC", "PMSC"}
tereminator = '.!?,;'

def characteer_return(word,tag):
    if word in brackets:
        word = brackets.get(word, "X")
        tag = "PMS"
    if word == "!":
        tag = "PME"
    if word == "''":
        word = '"'
    if word == "`":
        word = "'"
        tag = "PMS"
    if word == "--":
        word= "—" #em dash
    return word, tag

# Mapping of detailed POS tags to rough POS tags
tag_mapping = {
    "NNC": "NN.*", "NNP": "NN.*", "NNPA": "NN.*", "NNCA": "NN.*",
    "PRS": "PR.*", "PRP": "PR.*", "PRSP": "PR.*", "PRO": "PR.*",
    "PRQ": "PR.*", "PRQP": "PR.*", "PRL": "PR.*", "PRC": "PR.*",
    "PRF": "PR.*", "PRI": "PR.*", "DTC": "DT.*", "DTCP": "DT.*",
    "DTP": "DT.*", "DTPP": "DT.*", "CCT": "CC.*", "CCR": "CC.*",
    "CCB": "CC.*", "CCA": "CC.*", "CCP": "CC.*", "CCU": "CC.*",
    "VBW": "VB.*", "VBS": "VB.*", "VBH": "VB.*", "VBN": "VB.*",
    "VBTS": "VB.*", "VBTR": "VB.*", "VBTF": "VB.*", "VBTP": "VB.*",
    "VBAF": "VB.*", "VBOF": "VB.*", "VBOB": "VB.*", "VBOL": "VB.*",
    "VBOI": "VB.*", "VBRF": "VB.*", "JJD": "JJ.*", "JJC": "JJ.*",
    "JJCC": "JJ.*", "JJCS": "JJ.*", "JJCN": "JJ.*", "JJN": "JJ.*",
    "RBD": "RB.*", "RBN": "RB.*", "RBK": "RB.*", "RBP": "RB.*",
    "RBB": "RB.*", "RBR": "RB.*", "RBQ": "RB.*", "RBT": "RB.*",
    "RBF": "RB.*", "RBW": "RB.*", "RBM": "RB.*", "RBL": "RB.*",
    "RBI": "RB.*", "RBJ": "RB.*", "RBS": "RB.*", "CDB": "CD.*",
    "PMP": "PM.*", "PME": "PM.*", "PMQ": "PM.*", "PMC": "PM.*",
    "PMSC": "PM.*", "PMS": "PM.*", "LM": "LM", "TS": "TS",
    "FW":"FW"
}

def map_tag(tag):
    # Check for combined tags
    if "_" in tag:
        parts = tag.split("_")
        print(f"Split parts: {parts}")  # Debugging: Show split parts of the tag

        if "PMS" in parts[0] and "PMS" in parts[-1]:
            parts = parts[1:-1]  # Remove the first and last partsg
        
        if any(part in parts for part in terminating_tags):
            print(parts)
            mapped_tag=[]
            for part in parts:
                mapping = tag_mapping.get(part, "X")
                mapped_tag.append(mapping)
                print(mapped_tag)
            mapped_tags = " ".join(mapped_tag)
            print(mapped_tags)
            return mapped_tags
        else:
            mapped_tags = tag_mapping.get(parts[0], "X")
            return "".join(mapped_tags)

    return tag_mapping.get(tag, "X")

def semantic_cleaning(sentence):
    # Replace all hyphens in words with nothing
    dehyphened = re.sub(r'(\b\w+(-\w+)+\b)', lambda match: '^ ' + match.group(0).replace('-', ''), sentence)
    print(fr"from {sentence} to {dehyphened}")
    apo_at_handling = re.sub(r"\b\w*'[a-z]\b", lambda match: match.group(0).replace("'", "* a"), dehyphened)
    number_comma_handling = re.sub(r"^(?:\d{1,3}(,\d{3})*|\d+)$", lambda match: match.group(0).replace(",", ""), apo_at_handling)

    return number_comma_handling


def punctuation_separation(sentence):
    i = 0  # Index to iterate through the sentence

    while i < len(sentence):
        char = sentence[i]
        # If the character is a punctuation mark
        if char in tereminator:
            # Check the preceding character
            if i > 0 and sentence[i - 1] != ' ':
                sentence = sentence[:i] + ' ' + sentence[i:]
                i += 1  # Adjust index after insertion
            
            # Check the succeeding character
            if i < len(sentence) - 1 and sentence[i + 1] != ' ':
                sentence = sentence[:i + 1] + ' ' + sentence[i + 1:]
                i += 1  # Adjust index after insertion
        i += 1  # Move to the next 
    
    return sentence.strip()

def pos_tag(sentence):
    # Set the path to the Stanford POS Tagger directory
    stanford_pos_tagger_dir = "rules/Libraries/FSPOST"

    # Set the paths to the model and jar files
    model = stanford_pos_tagger_dir + '/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    jar = stanford_pos_tagger_dir + '/stanford-postagger.jar'

    # Preprocess the sentence to handle punctuation before and after words
    dehyphened = semantic_cleaning(sentence)
    preprocessed_sentence = punctuation_separation(dehyphened)
    print(preprocessed_sentence)

    # Command to run the POS tagger
    command = [
        'java', '-mx2g', '-cp', jar, 'edu.stanford.nlp.tagger.maxent.MaxentTagger',
        '-model', model, '-outputFormat', 'tsv'
    ]

    # Run the command using subprocess
    result = subprocess.run(command, input=preprocessed_sentence, text=True, capture_output=True, encoding='utf-8')
    to_print = fr"{result}"
    print(to_print)

    if result.returncode != 0:
        print("Error:", result.stderr)
        return None

    # Process the output
    tagged_sentence = result.stdout.strip().split('\n')
    pos_tags = []
    contraction_word = ""

    after_skip = 0
    index_holder = 0
    


    for idx, word_tag in enumerate(tagged_sentence):
        try:
            print(word_tag)
            print(idx)
            print(len(tagged_sentence))
            print(index_holder)
            word, tag = word_tag.split('\t')

            word, tag = characteer_return(word,tag)
            tagged_sentence[idx + index_holder] = word + "\t" + tag

            if idx < len(tagged_sentence):
                next_word, _= tagged_sentence[idx + 1 + index_holder].split('\t')
            else:
                next_word = ""

            if next_word != "*":
                not_contraction = True

            if word == "*":
                print("A contraction was used")
                prev_word, prev_tag = tagged_sentence[idx - 1 + index_holder].split('\t')
                next_word, next_tag = tagged_sentence[idx + 1 + index_holder].split('\t')
                contraction_word = prev_word + "'" + next_word[1:]
                contraction_pos = prev_tag + "_" + next_tag
                pos_tags[-1]= contraction_pos
                print(contraction_word)
                print(contraction_pos)
                tagged_sentence[idx - 1 + index_holder] = contraction_word + "\t" + contraction_pos
                del tagged_sentence[idx + index_holder]
                index_holder -= 1
                continue

            
            if word == "^":
                next_word, next_tag = tagged_sentence[idx + 1 + index_holder].split('\t')
                print(fr"{next_word} is originally hyphinated")
                for i in range(1, len(next_word)):
                    hyphenated_word = next_word[:i] + '-' + next_word[i:]  # Insert hyphen at position i
                    # Check if the hyphenated word exists in the sentence
                    if hyphenated_word in sentence:
                        # Step 3: Replace the original word with the correct hyphenation
                        tagged_sentence[idx + index_holder] = hyphenated_word + "\t" +  next_tag
                        word = hyphenated_word
                        tag = next_tag
                        print(fr"rehyphend {next_word} to {hyphenated_word}")
                        del tagged_sentence[idx+1 + index_holder]
                        break  # Exit the loop once the correct hyphenation is found
                        
            
            print(f"Processing: Word = '{word}', Tag = '{tag}'")

            if after_skip == 1:
                after_skip -= 1
                prev_tag = pos_tags[-1].split('_')
                print(f"Word '{word}' attacched to previous word as {prev_tag}, skipping.  {after_skip}")
                if tag in terminating_tags:
                    continue
                else:
                    prev_tag = pos_tags[-1].split('_') 
                    prev_tag[-1] = tag
                    prev_tag= "_".join(prev_tag)
                    pos_tags[-1] = prev_tag

                continue

            elif after_skip > 1:
                after_skip -= 2
                print(f"Word '{word}' was part of a contraction {contraction_word or hyphenated_word}, skipping. {after_skip}")
                contraction_word =''
                hyphenated_word = ''
                continue


            # Locate the punctuation in the original sentence
            prev_word,prev_tag = tagged_sentence[idx - 1 + index_holder].split('\t')
            prev_word, _ = characteer_return(prev_word, prev_tag)
            search_word1= fr"{prev_word} {word}"
            search_word2= fr"{prev_word}{word}"

            skip = 1
            search_index = sentence.find(search_word1)
            print(fr"index search key: {search_word1} and search_index: {search_index}")
            if search_index == -1:
                search_index = sentence.find(search_word2)
                skip -= 1
                if search_index == -1:
                    print(f"Word '{search_word2}' not found in original sentence, skipping. search_index: {search_index}")
                    pos_tags.append(tag)
                    continue
                skip = skip + len(prev_word)
                start_index = skip + search_index
            else: # Locate the word in the original sentence
                skip = skip + len(prev_word)
                start_index = skip + search_index

            if start_index == -1:
                print(f"Word '{word}' not found in original sentence, skipping.")
                pos_tags.append(tag)
                continue

            end_index = start_index + len(word)

            # Characters before and after the word
            before_char = sentence[start_index - 1] if start_index > 0 else None
            after_char = sentence[end_index] if end_index < len(sentence) else None

            print(f"Character before: '{before_char}', Character after: '{after_char}'")

            # Check punctuation positions
            attached_before = re.match(r'[.!?,;:—"\'()\[\]{}]', before_char) if before_char else False
            attached_after = re.match(r'[.!?,;:—"\'()\[\]{}]', after_char) if after_char and not_contraction else False

            spacer  = "_"

            
            if attached_before and attached_after:
                # Wrapped by punctuation
                if idx > 0 and idx + 1 < len(tagged_sentence):
                    prev_tag = pos_tags[-1]
                    next_word, next_tag = tagged_sentence[idx + 1].split('\t')
                    next_word, next_tag = characteer_return(next_word, next_tag)
                    if next_word in tereminator:
                        spacer = " "
                    # Update POS tag and skip the succeeding token
                    combined_tag = f"{prev_tag}{spacer}{tag}{spacer}{next_tag}"
                    pos_tags[-1] = combined_tag
                    
                    print(f"Word '{word}' is WRAPPED. Combined tag: {combined_tag}")
                    after_skip += 1
                    continue  # Skip the succeeding token in the iteration
            elif attached_before:
                # Punctuation is BEFORE the word
                if pos_tags:
                    prev_tag = pos_tags[-1]
                    if tag in terminating_tags:
                        spacer = " "
                    combined_tag = f"{prev_tag}{spacer}{tag}"
                    pos_tags[-1] = combined_tag  # Update the previous token's tag
                    print(f"Word '{word}' has punctuation BEFORE. Combined tag: {combined_tag}")
            elif attached_after:
                if word in tereminator:
                    spacer = " "
                # Punctuation is AFTER the word
                if idx + 1 < len(tagged_sentence):
                    next_word, next_tag = tagged_sentence[idx + 1].split('\t')
                    next_word, next_tag = characteer_return(next_word, next_tag)
                    if next_word in tereminator:
                        spacer = " "
                    else:
                        spacer = "_"
                    combined_tag = f"{tag}{spacer}{next_tag}"
                    pos_tags.append(combined_tag)  # Append the combined tag
                    print(f"start index: {start_index} end index:{end_index}")
                    print(f"Word '{word}' has punctuation AFTER. Combined tag: {combined_tag}")
                    after_skip += 1
                    continue  # Skip regular appending for this token
            else:
                # Regular word, no punctuation attachment
                pos_tags.append(tag)
                print(f"Processed word: '{word}' with tag: '{tag}'")
        except ValueError:
            print(f"Skipping invalid entry: {word_tag}")
            del tagged_sentence[idx + index_holder]
            index_holder -= 1
            
    detailed_pos= " ".join(pos_tags)

    rough_map = []
    
    to_maps = detailed_pos.split(" ")
    for to_map in to_maps:
        mapped = map_tag(to_map)
        rough_map.append(mapped)
        
    general_pos = " ".join(rough_map)    

    return detailed_pos, general_pos

# Test the function
# text = "Kumusta ka, Juan? 'di ba't sinabi mo'y ika-8 'yon: 'aalis ako nang maaga—mga alas-siyete ng umaga, siguro...' Pero, bakit nandito ka pa rin?!"
# pos, rpos = pos_tag(text)
# print(pos)
# print(rpos)


    #####pag nag  invalid value error, mag aadd ng 1 sa isang bagong variable, para sa next makiktia na iba pagkukuhanan ng prev word, tapos ii-state dun sa bawat continue/break states yung magiging value nia 