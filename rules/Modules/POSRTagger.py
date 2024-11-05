import subprocess
import re

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
    "FW": "FW"
}

def map_tag(tag):
    """
    Maps detailed POS tags to general POS tags.
    Handles combined tags (tags with underscores) and unknown tags.
    """
    if "_" in tag:
        parts = tag.split("_")
        mapped_tags = [tag_mapping.get(part, "X") for part in parts]
        return "_".join(mapped_tags)
    return tag_mapping.get(tag, "X")

def preprocess_sentence(sentence):
    """
    Preprocesses the sentence by handling punctuation before and after words.
    The regex separates punctuation from words but keeps them in their places.
    """
    return re.sub(r'([.!?,;:—]*["\'\s]*)(\w+)', r'\1 \2', sentence)

def pos_tag(sentence):
    """
    Runs the Stanford POS Tagger on a given sentence and returns both original and general POS tags.
    """
    # Set the path to the Stanford POS Tagger directory
    stanford_pos_tagger_dir = "rules/Libraries/FSPOST"

    # Set the paths to the model and jar files
    model = stanford_pos_tagger_dir + '/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    jar = stanford_pos_tagger_dir + '/stanford-postagger.jar'

    # Preprocess the sentence to handle punctuation before and after words
    preprocessed_sentence = preprocess_sentence(sentence)

    # Command to run the POS tagger
    command = [
        'java', '-Xmx2g', '-cp', jar, 'edu.stanford.nlp.tagger.maxent.MaxentTagger',
        '-model', model, '-outputFormat', 'tsv'
    ]

    # Run the command using subprocess
    result = subprocess.run(command, input=preprocessed_sentence, text=True, capture_output=True, encoding='utf-8')

    if result.returncode != 0:
        print("Error:", result.stderr)
        return None

    # Process the output and map detailed tags to general tags
    tagged_sentence = result.stdout.strip().split('\n')
    original_pos_tags = []
    general_pos_tags = []
    previous_tag = ""

    for word_tag in tagged_sentence:
        try:
            word, original_tag = word_tag.split('\t')
            general_tag = map_tag(original_tag)

            # Check for punctuation attached before or after the word
            if re.search(r'^[.!?,;:—"\'\s]+', word):
                if original_pos_tags:
                    # Combine the previous tag with punctuation before the word
                    original_pos_tags[-1] += f" {original_tag}"
                    general_pos_tags[-1] += f" {general_tag}"
                else:
                    # Handle cases where the first tag has punctuation before the word
                    original_pos_tags.append(f"{original_tag} ")
                    general_pos_tags.append(f"{general_tag} ")
                previous_tag = general_tag
                continue

            if re.search(r'[.!?,;:—"\'\s]+$', word):
                if original_pos_tags:
                    # Combine the previous tag with punctuation after the word
                    original_pos_tags[-1] += f" {original_tag}"
                    general_pos_tags[-1] += f" {general_tag}"
                else:
                    # Handle cases where the first tag has punctuation after the word
                    original_pos_tags.append(f"{original_tag} ")
                    general_pos_tags.append(f"{general_tag} ")
                previous_tag = general_tag
            else:
                if previous_tag:
                    # Handle the case where punctuation follows a word
                    original_pos_tags.append(f"{original_tag}")
                    general_pos_tags.append(f"{general_tag}")
                    previous_tag = ""
                else:
                    original_pos_tags.append(original_tag)
                    general_pos_tags.append(general_tag)
        except ValueError:
            # Skip lines that do not have exactly two parts
            continue

    # Join original POS tags and general POS tags into space-separated strings
    original_pos_tag_sequence = " ".join(original_pos_tags)
    general_pos_tag_sequence = " ".join(general_pos_tags)
    
    return original_pos_tag_sequence, general_pos_tag_sequence
