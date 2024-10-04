from Modules.preprocessing.Tokenizer import tokenize
from Modules.preprocessing.POSDTagger import pos_tag
from Modules.preprocessing.Lemmatizer import lemmatize_sentence

def preprocess_text(text, batch_size=700):
    # Tokenize the paragraph into individual sentences
    tokenized_sentences = tokenize(text)
    
    pos_tagged_sentences = []
    lemmatized_sentences = []

    results = []

    for i in range(0, len(tokenized_sentences), batch_size):
        batch = tokenized_sentences[i:i + batch_size]

        pos_tagged_batch = []
        lemmatized_batch = []

        for sentence in batch:
            if sentence:
                # Perform POS tagging and lemmatization
                pos_tagged_batch.append(pos_tag(sentence))  # Use the correct function name
                lemmatized_batch.append(lemmatize_sentence(sentence))
            else:
                pos_tagged_batch.append('')
                lemmatized_batch.append('')

        # Append batch results to the final lists
        pos_tagged_sentences.extend(pos_tagged_batch)
        lemmatized_sentences.extend(lemmatized_batch)

    # Prepare the dictionary result
    for tok_sentence, pos, lemma in zip(tokenized_sentences, pos_tagged_sentences, lemmatized_sentences):
        results.append({
            'tokenized': tok_sentence,
            'pos': pos,  # Use the correct variable name
            'lemmatized': lemma
        })

    return results

def main(paragraph_text):
    return preprocess_text(paragraph_text)

# Example usage
if __name__ == "__main__":
    input_paragraph = """
    This is a sample paragraph with multiple sentences. It should be tokenized properly.
    Also, the POS tagging and lemmatization should work on each sentence.
    """
    result = main(input_paragraph)
    print(result)
