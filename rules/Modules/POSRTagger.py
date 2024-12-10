import subprocess
import re

punctuation = '.!?,;:—"\'()\[\]{}'
brackets = {
    "-LRB-":"(", "-RRB-":")",
    "-LSB-":"[","-RSB-":"]",
    "-LCB-":"{","-RCB-":"}"
    }
terminating_tags = {"PMP", "PME", "PMQ", "PMC", "PMSC"}
tereminator = '.!?,;:'

def characteer_return(word,tag):
    if word in brackets:
        word = brackets.get(word, "X")
        tag = "PMS"
    if word == "!":
        tag = "PME"
    if word == "''": #front double qoute
        word = '”'
    if word == "``": #back double qoute
        word = '“'
        tag="PMS"
    if word == "`": #back tick for single qoutation symbol
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
    if "_" in tag:
        parts = tag.split("_")
        if "PMS" in parts[0] and "PMS" in parts[-1] and len(parts) > 2:
            parts = parts[1:-1]
        if any(part in parts for part in terminating_tags):
            mapped_tag = [tag_mapping.get(part, "X") for part in parts]
            mapped_tags = " ".join(mapped_tag)
            return mapped_tags
        else:
            mapped_tag = tag_mapping.get(parts[0], "X")
            return mapped_tag
    else:
        mapped_tag = tag_mapping.get(tag, "X")
        return mapped_tag

def semantic_cleaning(sentence):
    #print(f"semantic_cleaning - input: {sentence}")
    dehyphened = re.sub(r'(\b\w+(-\w+)+\b)', lambda match: '^ ' + match.group(0).replace('-', ''), sentence)
    #print(dehyphened)
    apo_handling = re.sub(r"\b\w*'[a-z]\b", lambda match: match.group(0).replace("'", "* a"), dehyphened)
    #print(apo_handling)
    number_comma_handling = re.sub(r"\b\d{1,3}(?:,\d{3})*\b", lambda match: match.group(0).replace(",", ""), apo_handling)
    #print(f"semantic_cleaning - output: {number_comma_handling}")
    return number_comma_handling

def punctuation_separation(sentence):
    #print(f"punctuation_separation - input: {sentence}")
    i = 0
    while i < len(sentence):
        char = sentence[i]
        if char in tereminator:
            if i > 0 and sentence[i - 1] != ' ':
                sentence = sentence[:i] + ' ' + sentence[i:]
                i += 1
            if i < len(sentence) - 1 and sentence[i + 1] != ' ':
                sentence = sentence[:i + 1] + ' ' + sentence[i + 1:]
                i += 1
        i += 1
    #print(f"punctuation_separation - output: {sentence.strip()}")
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

    # Command to run the POS tagger
    command = [
        'java', '-mx2g', '-cp', jar, 'edu.stanford.nlp.tagger.maxent.MaxentTagger',
        '-model', model, '-outputFormat', 'tsv'
    ]

    # Run the command using subprocess
    result = subprocess.run(command, input=preprocessed_sentence, text=True, capture_output=True, encoding='utf-8')
    #print(result.stdout)

    if result.returncode != 0:        
        return None

    # Process the output
    tagged_sentence = result.stdout.strip().split('\n')
    tagged_sentence = list(filter(None, tagged_sentence))
    pos_tags = []
    contraction_word = ""
    abbreviation = ""
    hyphenated_word = ""
    after_skip = 0
    index_holder = 0
    used_hyphen = 0

    for idx, word_tag in enumerate(tagged_sentence):
        #print(word_tag)
        try:          
            word, tag = word_tag.split('\t')

            word, tag = characteer_return(word,tag)
            tagged_sentence[idx + index_holder] = word + "\t" + tag


            if idx < len(tagged_sentence)-1:
                next_word, _= tagged_sentence[idx + 1].split('\t')
            else:
                next_word = ""

            if next_word != "*":
                not_contraction = True

            if word == "*":
                
                prev_word, prev_tag = tagged_sentence[idx - 1 + index_holder].split('\t')
                next_word, next_tag = tagged_sentence[idx + 1 + index_holder].split('\t')
                contraction_word = prev_word + "'" + next_word[1:]
                contraction_pos = prev_tag + "_" + next_tag
                pos_tags[-1]= contraction_pos
                tagged_sentence[idx - 1 + index_holder] = contraction_word + "\t" + contraction_pos
                del tagged_sentence[idx + index_holder]
                index_holder -= 1
                #print(f"Word '{prev_word}' and '{next_word}' was part of a contraction {contraction_word}")
                continue

            
            if word == "^":
                word_tag = word + "\t" + tag
                next_word, next_tag = tagged_sentence[idx + 1].split('\t')
                start_index = sentence.replace("-","").find(next_word)
                #print(start_index)
                end_index = start_index + len(next_word)
                #print(len(next_word))
                substring = sentence[start_index:end_index+1]
                #print(substring.count("-"))
                hyphen_count = substring.count("-")
                substring = sentence[start_index+used_hyphen:end_index+hyphen_count+used_hyphen]
                used_hyphen += hyphen_count
                word = substring
                tag = next_tag
                tagged_sentence [idx] = word + "\t" + tag
                del tagged_sentence[idx + 1]  # Remove the original unhyphenated word


            if after_skip == 1:
                after_skip -= 1
                prev_tag = pos_tags[-1].split('_')
                
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
                #print(f"Word '{word}' was part of a contraction {contraction_word or hyphenated_word or abbreviation}, skipping. {after_skip}")
                contraction_word =''
                hyphenated_word = ''
                continue

            #print(fr"Prcesssing wordd:{word} tag:{tag}")


            # Locate the punctuation in the original sentence
            prev_word,prev_tag = tagged_sentence[idx - 1].split('\t')
            prev_word, _ = characteer_return(prev_word, prev_tag)
            search_word1= fr"{prev_word} {word}"
            search_word2= fr"{prev_word}{word}"

            skip = 1
            search_index = sentence.find(search_word1)
            
            if search_index == -1:
                #print(fr"index search key: {search_word1} and search_index: {search_index}")
                search_index = sentence.find(search_word2)
                skip -= 1
                if search_index == -1:
                    #print(f"Word '{search_word2}' not found in original sentence, skipping. search_index: {search_index}")
                    pos_tags.append(tag)
                    continue
                skip = skip + len(prev_word)
                start_index = skip + search_index
            else: # Locate the word in the original sentence
                skip = skip + len(prev_word)
                start_index = skip + search_index

            if start_index == -1:
                #print(f"Word '{word}' not found in original sentence, skipping.")
                pos_tags.append(tag)
                continue

            end_index = start_index + len(word)

            # Characters before and after the word
            before_char = sentence[start_index - 1] if start_index > 0 else None
            after_char = sentence[end_index] if end_index < len(sentence) else None

            #print(f"Character before: '{before_char}', Character after: '{after_char}'")

            if tag == "NNPA" and after_char==".":
                abbreviation = word
                word = word +"."
                (f"Processed word: '{word}' with tag: '{tag}' as abbreviation")
                after_skip += 2
                pos_tags.append(tag)
                continue

            # Check punctuation positions
            attached_before = re.match(r'[.!?,;:—"“”`\'()\[\]{}]', before_char) if before_char else False
            attached_after = re.match(r'[.!?,;:—"“”`\'()\[\]{}]', after_char) if after_char and not_contraction else False

            spacer  = "_"

            
            if attached_before and attached_after:
                # Wrapped by punctuation
                if idx > 0 and idx + 1 < len(tagged_sentence):
                    prev_tag = pos_tags[-1]
                    next_word, next_tag = tagged_sentence[idx + 1].split('\t')
                    next_word, next_tag = characteer_return(next_word, next_tag)
                    if next_word in tereminator or tag in terminating_tags or prev_tag in terminating_tags:
                        spacer = " "
                    # Update POS tag and skip the succeeding token
                    combined_tag = f"{prev_tag}{spacer}{tag}{spacer}{next_tag}"
                    pos_tags[-1] = combined_tag
                    #print(f"Word '{word}' is WRAPPED. Combined tag: {combined_tag}")
                    after_skip += 1
                    continue  # Skip the succeeding token in the iteration
            elif attached_before:
                # Punctuation is BEFORE the word
                    prev_tag = pos_tags[-1]
                    if tag in terminating_tags or prev_tag in terminating_tags:
                        spacer = " "
                    
                    combined_tag = f"{prev_tag}{spacer}{tag}"
                    pos_tags[-1] = combined_tag  # Update the previous token's tag
                    #print(f"Word '{word}' has punctuation BEFORE. Combined tag: {combined_tag}")
            elif attached_after:
                if word in tereminator or tag in terminating_tags:
                    spacer = " "
                # Punctuation is AFTER the word
                if idx + 1 < len(tagged_sentence):
                    next_word, next_tag = tagged_sentence[idx + 1].split('\t')
                    next_word, next_tag = characteer_return(next_word, next_tag)
                    if next_word in tereminator:
                        spacer = " "
                    combined_tag = f"{tag}{spacer}{next_tag}"
                    pos_tags.append(combined_tag)  # Append the combined 
                    #print(f"start index: {start_index} end index:{end_index}")
                    #print(f"Word '{word}' has punctuation AFTER. Combined tag: {combined_tag}")        
                    after_skip += 1
                    continue  # Skip regular appending for this token
            else:
                # Regular word, no punctuation attachment
                pos_tags.append(tag)
                #print(f"Processed word: '{word}' with tag: '{tag}'")
        except ValueError:
            #print(f"Skipping invalid entry: {word_tag}")
            index_holder -= 1
            
    detailed_pos= " ".join(pos_tags)
    
    rough_map = []
    
    to_maps = detailed_pos.split(" ")
    for to_map in to_maps:
        mapped = map_tag(to_map)
        rough_map.append(mapped)
    general_pos = " ".join(rough_map)   
    
    #print(sentence)
    #print(f"{detailed_pos}")
    #print(f"{general_pos}")
    return detailed_pos, general_pos




