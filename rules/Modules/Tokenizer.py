import re

def tokenize(text):
    """
    Tokenizes text into sentences using punctuation delimiters.
    """
    # Regular expression to split sentences by punctuation, considering Tagalog-specific cases
    regex = r"(?<=[.!?])\s+"
    sentences = re.split(regex, text.strip())
    return sentences

def split_sentences(text):
    """
    Further splits and filters sentences for usability.
    """
    # Step 1: Tokenize into sentences
    raw_sentences = tokenize(text)

    # Step 2: Clean and process sentences
    cleaned_sentences = []
    for sentence in raw_sentences:
        # Remove leading/trailing whitespace
        sentence = sentence.strip()

        # Ignore fragments or very short sentences
        if len(sentence.split()) > 2:
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences

# Example usage
text = """
Natalo ng Italya ang Portugal sa puntos na 31-5 sa Grupong C noong 2007 sa Pandaigdigang laro ng Ragbi sa Parc des Princes, Paris, France, Manila. Natalo ng Italya ang Portugal, patric. Gayumpaman, hindi natin makakaila ang pagbabago
"""
usable_sentences = split_sentences(text)
print(usable_sentences)