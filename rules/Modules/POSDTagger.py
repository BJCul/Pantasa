import subprocess
import re

def preprocess_sentence(sentence):
    # Use regex to handle punctuation before and after words
    # The regex will also separate punctuation from words but keep them in their places
    return re.sub(r'([.!?,;:—]*["\'\s]*)(\w+)', r'\1 \2', sentence)

def pos_tag(sentence):
    # Set the path to the Stanford POS Tagger directory
    stanford_pos_tagger_dir = "rules/Libraries/FSPOST"

    # Set the paths to the model and jar files
    model = stanford_pos_tagger_dir + '/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    jar = stanford_pos_tagger_dir + '/stanford-postagger.jar'

    # Preprocess the sentence to handle punctuation before and after words
    preprocessed_sentence = preprocess_sentence(sentence)

    # Command to run the POS tagger
    command = [
        'java', '-mx2g', '-cp', jar, 'edu.stanford.nlp.tagger.maxent.MaxentTagger',
        '-model', model, '-outputFormat', 'tsv'
    ]

    # Run the command using subprocess
    result = subprocess.run(command, input=preprocessed_sentence, text=True, capture_output=True, encoding='utf-8')

    if result.returncode != 0:
        print("Error:", result.stderr)
        return None

    # Process the output
    tagged_sentence = result.stdout.strip().split('\n')
    pos_tags = []
    previous_word = ""
    tag_count = 0
    
    for word_tag in tagged_sentence:
        try:
            word, tag = word_tag.split('\t')
            
            # Check for punctuation attached before or after the word
            if re.search(r'^[.!?,;:—"\'\s]+', word):
                punctuation = re.search(r'^[.!?,;:—"\'\s]+', word).group()
                base_word = re.sub(r'^[.!?,;:—"\'\s]+', '', word)
                
                if pos_tags:
                    # Combine the previous tag with punctuation before the word
                    pos_tags[-1] += f" {tag}"
                else:
                    # Handle cases where the first tag has punctuation before the word
                    pos_tags.append(f"{tag} ")
                previous_word = base_word
                continue

            if re.search(r'[.!?,;:—"\'\s]+$', word):
                base_word = re.sub(r'[.!?,;:—"\'\s]+$', '', word)
                punctuation = re.search(r'[.!?,;:—"\'\s]+$', word).group()
                
                if pos_tags:
                    # Combine the previous tag with punctuation after the word
                    pos_tags[-1] += f" {tag}"
                else:
                    # Handle cases where the first tag has punctuation after the word
                    pos_tags.append(f"{tag} ")
                previous_word = base_word
            else:
                if previous_word:
                    # Handle the case where punctuation follows a word
                    pos_tags.append(f"{tag}")
                    previous_word = ""
                else:
                    pos_tags.append(tag)
        except ValueError:
            # Skip lines that do not have exactly two parts
            continue

    # Join POS tags into a single string separated by spaces
    
    tag_count += 1
    
    if tag_count >1:
        if pos_tag_sequence[-1] == "_":
            pos_tag_sequence = "".join(pos_tags)
        else:
            pos_tag_sequence = " ".join(pos_tags)
    else:
        pos_tag_sequence = " ".join(pos_tags)
    return pos_tag_sequence

