import re

def tokenize(text):
    # Regular expression to match sentences with strict punctuation delimiters, including em dash
    regex = r"[^.!?,;:—\s][^.!?,;:—]*[.!?,;:—]?['\"]?(?=\s|$)"
    pattern = re.compile(regex)

    # Initialize a list to store the tokenized sentences
    result = []

    # Split the text by lines since the dataset contains sentence IDs and sentences separated by a tab
    for line in text.splitlines():
        if "\t" in line:
            # Extract only the sentence text (ignore the sentence ID part)
            _, sentence_text = line.split("\t", 1)

            # Find all sentences in the text
            sentences = pattern.findall(sentence_text)

            # Extend the result list with the tokenized sentences
            result.extend(sentences)
        else:
            sentences = pattern.findall(text)

            # Extend the result list with the tokenized sentences
            result.extend(sentences)

    return result

