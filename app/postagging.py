import pandas as pd
import subprocess
import tempfile

# Example: POS tags you're interested in
pos_tags = [
    "NN.*", "NNC", "NNP", "NNPA", "NNCA", "PR.*", "PRS", "PRP", "PRSP", "PRO",
    "PRQ", "PRQP", "PRL", "PRC", "PRF", "PRI", "DT.*", "DTC", "DTCP", "DTP",
    "DTPP", "CC.*", "CCT", "CCR", "CCB", "CCA", "CCP", "CCU", "LM", "VB.*",
    "VBW", "VBS", "VBH", "VBN", "VBTS", "VBTR", "VBTF", "VBTP", "VBAF", "VBOF",
    "VBOB", "VBOL", "VBOI", "VBRF", "JJ.*", "JJD", "JJC", "JJCC", "JJCS",
    "JJCN", "JJN", "RB.*", "RBD", "RBN", "RBK", "RBP", "RBB", "RBR", "RBQ", 
    "RBT", "RBF", "RBW", "RBM", "RBL", "RBI", "RBJ", "RBS", "CD.*", "CDB",
    "TS", "FW", "PM.*", "PMP", "PME", "PMQ", "PMC", "PMSC", "PMS"
]

# Define your FSPOSTagger class (as provided)
class FSPOSTagger:
    def __init__(self, jar_path, model_path):
        self.jar_path = jar_path
        self.model_path = model_path

    def tag(self, tokens):
        # Prepare the input sentence
        sentence = ' '.join(tokens)

        # Use a temporary file to simulate the command-line behavior
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(sentence)
            temp_file.flush()  # Ensure the content is written to the file
            
            temp_file_path = temp_file.name

        # Command to run the Stanford POS Tagger (FSPOST)
        command = [
            'java', '-mx300m', '-cp', self.jar_path,
            'edu.stanford.nlp.tagger.maxent.MaxentTagger',
            '-model', self.model_path,
            '-textFile', temp_file_path  # Pass the temp file as input
        ]

        # Execute the command and capture the output
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

        # Process the raw output by splitting each word|tag pair
        tagged_output = output.decode('utf-8').strip().split()
        tagged_tokens = [tuple(tag.split('|')) for tag in tagged_output if '|' in tag]  # Correctly split by '|'
        
        return tagged_tokens


# Initialize your tagger (update jar_path and model_path to actual paths)
jar_path = 'C:/Users/Jarlson/Downloads/3rd AY dwnld/2nd sem/Thesis/Pantasa/Libraries/FSPOST/stanford-postagger.jar'  # Adjust the path to the JAR file
model_path = 'C:/Users/Jarlson/Downloads/3rd AY dwnld/2nd sem/Thesis/Pantasa/Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'  # Adjust the path to the model file
pos_tagger = FSPOSTagger(jar_path, model_path)

# Load your dictionary CSV
df = pd.read_csv('filtered_taglish_dictionary.csv', header=None, names=['Word'])

# Create an empty dictionary to hold POS-tagged words
tagged_words = {tag: [] for tag in pos_tags}

# Tag each word and group them by POS tag
for word in df['Word']:
    # Tag the word using your FSPOSTagger
    tagged_token = pos_tagger.tag([word])
    
    # Extract the POS tag from the result
    if tagged_token:
        word, tag = tagged_token[0]  # Assuming one token at a time

        # Check if the tag matches any of your listed POS tags
        for pos in pos_tags:
            if tag.startswith(pos.replace(".*", "")):  # Handle patterns like NN.* or VB.*
                tagged_words[pos].append(word)
                break  # Stop after matching the first relevant POS tag

# Save each POS group into a separate CSV
for pos, words in tagged_words.items():
    if words:  # Only save if there are words for that POS tag
        pd.DataFrame(words, columns=['Word']).to_csv(f'{pos}_words.csv', index=False, header=False)

print("Words have been grouped and saved into CSV files by POS tag.")
