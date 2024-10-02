import re

def tokenize(text):
    # Regular expression to match sentences with strict punctuation delimiters, including em dash
    regex = r"[^.!?,;:—\s][^.!?,;:—]*[.!?,;:—]?['\"]?(?=\s|$)"
    pattern = re.compile(regex)
    
    # Find all sentences in the text
    sentences = pattern.findall(text)
    
    tagalog_sentences = tagalog_sentences.append(sentences)

    return tagalog_sentences


