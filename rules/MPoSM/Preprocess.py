from .Modules.Tokenizer import tokenize
from .Modules.POSDTagger import pos_tag as pos_dtag
from .Modules.POSRTagger import pos_tag as pos_rtag

def load_dataset(file_path):
    """
    Generator function to load the dataset line by line.
    Each line is split into an identifier and the actual sentence.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:
                # Split the line into identifier and sentence based on the tab character
                identifier, sentence = stripped_line.split('\t', 1)
                yield identifier, sentence  # Yield both parts

def preprocess_text(input_file, output_file, batch_size=5000):
    """
    Process the input file by tokenizing and performing POS tagging, 
    writing the output in batches to avoid memory overload.
    """
    with open(output_file, 'a', encoding='utf-8') as output:
        tokenized_sentences = []
        sentence_identifiers = []
        batch_counter = 0
        
        # Stream the dataset using the generator function
        for identifier, sentence in load_dataset(input_file):
            # Tokenize the sentence
            sentences = tokenize(sentence)
            tokenized_sentences.extend(sentences)
            # Keep track of the identifier for each tokenized sentence
            sentence_identifiers.extend([identifier] * len(sentences))
            
            # Process in batches to avoid memory overload
            if len(tokenized_sentences) >= batch_size:
                process_batch(sentence_identifiers, tokenized_sentences, output)
                tokenized_sentences = []  # Clear the list after processing
                sentence_identifiers = []  # Clear the identifier list after processing

        # If there are any remaining sentences in the last incomplete batch
        if tokenized_sentences:
            process_batch(sentence_identifiers, tokenized_sentences, output)

    print(f"Preprocessed data saved to {output_file}")

def process_batch(sentence_identifiers, tokenized_sentences, output_file):
    """
    Helper function to process a batch of tokenized sentences and write to output file.
    Each sentence is written along with its identifier.
    """
    general_pos_tagged_batch = []
    detailed_pos_tagged_batch = []
    
    # Perform POS tagging for each sentence
    for sentence in tokenized_sentences:
        if sentence:
            general_pos_tagged_batch.append(pos_rtag(sentence))  # Rough POS tagging
            detailed_pos_tagged_batch.append(pos_dtag(sentence))  # Detailed POS tagging
        else:
            general_pos_tagged_batch.append('')
            detailed_pos_tagged_batch.append('')

    # Write tokenized sentences and their POS tags to the output file
    for identifier, tok_sentence, gen_pos, det_pos in zip(sentence_identifiers, tokenized_sentences, general_pos_tagged_batch, detailed_pos_tagged_batch):
        # Add quotes around sentences that contain commas to handle CSV format properly
        if ',' in tok_sentence:
            tok_sentence = f'"{tok_sentence}"'
        output_file.write(f"{identifier},{tok_sentence},{gen_pos},{det_pos}\n")

    # Clear lists after processing to save memory
    general_pos_tagged_batch.clear()
    detailed_pos_tagged_batch.clear()

# Main function to execute the preprocessing
def main():
    input_file = 'C:/Users/Carlo Agas/Documents/GitHub/Pantasa/dataset/ALT-Parallel-Corpus-20191206/data_fil.txt'   # Input file with raw text
    output_file = 'C:/Users/Carlo Agas/Documents/GitHub/Pantasa/MPoSM/preprocessed_output.csv'  # Output file to save processed data

    preprocess_text(input_file, output_file)  # Apply preprocessing and POS tagging

if __name__ == "__main__":
    main()
