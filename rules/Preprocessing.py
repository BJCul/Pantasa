import pandas as pd
from Modules.Tokenizer import tokenize
from Modules.POSDTagger import pos_tag as pos_dtag
from Modules.POSRTagger import pos_tag as pos_rtag
from Modules.Lemmatizer import lemmatize_sentence 
import os
import sys

# Set the JVM options to increase the heap size
os.environ['JVM_OPTS'] = '-Xmx2g'

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df['text'].tolist()

def save_text_file(text_data, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        for line in text_data:
            f.write(line + "\n")

def preprocess_text(input_file, tokenized_file, output_file, batch_size=700):
    dataset = load_dataset(input_file)
    tokenized_sentences = []
    general_pos_tagged_sentences = []
    detailed_pos_tagged_sentences = []
    lemmatized_sentences = []

    with open(tokenized_file, 'w', encoding='utf-8') as token_file:
        for text in dataset:
            sentences = tokenize(text)
            token_file.write("\n".join(sentences) + "\n")
            tokenized_sentences.extend(sentences)
        print(f"Sentences tokenized to {tokenized_file}")
    
    with open(input_file, "r+", encoding='utf-8') as f:
        d = f.readlines()
        f.seek(0)
        for i in d:
            if i != "\n":
                f.write(i)
        f.truncate()

    seen_lines = set()
    with open(tokenized_file, 'r+', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line not in seen_lines:  # If the line hasn't been seen before
                file.write(line)
                seen_lines.add(line)

    with open(output_file, 'a', encoding='utf-8') as output:
        for i in range(0, len(tokenized_sentences), batch_size):
            batch = tokenized_sentences[i:i + batch_size]

            general_pos_tagged_batch = []
            detailed_pos_tagged_batch = []
            lemmatized_batch = []

            for sentence in batch:
                if sentence:
                    general_pos_tagged_batch.append(pos_rtag(sentence))
                    detailed_pos_tagged_batch.append(pos_dtag(sentence))
                    lemmatized_batch.append(lemmatize_sentence(sentence))
                else:
                    general_pos_tagged_batch.append('')
                    detailed_pos_tagged_batch.append('')
                    lemmatized_batch.append('')

            # Append batch results to output file immediately
            for tok_sentence, gen_pos, det_pos, lemma in zip(batch, general_pos_tagged_batch, detailed_pos_tagged_batch, lemmatized_batch):
                # Add single quotes around sentences ending with a comma
                if ',' in tok_sentence:
                    tok_sentence = f'"{tok_sentence}"'
                output.write(f"{tok_sentence},{gen_pos},{det_pos},{lemma},\n")

            # Clear lists after each batch to avoid memory issues
            tokenized_sentences.clear()
            general_pos_tagged_sentences.clear()
            detailed_pos_tagged_sentences.clear()
            lemmatized_sentences.clear()

    print(f"Preprocessed data saved to {output_file}")

def main(input_csv, tokenized_txt, output_csv):
    preprocess_text(input_csv, tokenized_txt, output_csv)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python Preprocessing.py <input_csv> <tokenized_txt> <output_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    tokenized_txt = sys.argv[2]
    output_csv = sys.argv[3]

    main(input_csv, tokenized_txt, output_csv)


#enter this in cmd to run the program "python Preprocessing.py database\dataset.csv database\tokenized.txt database\preprocessed.csv"