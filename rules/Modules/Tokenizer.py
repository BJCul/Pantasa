import re

def tokenize(text):
    """
    Tokenizes text into sentences using punctuation delimiters,
    with structural abbreviation, proper sentence splitting, and tab-separated line handling.
    """
    # Regex pattern to identify abbreviations or initials
    abbreviation_pattern = r"\b[A-Z][a-z]{1,2}\."
    initials_pattern = r"\b[A-Z]\."

    # Placeholder list for abbreviations/initials
    placeholders = []

    def replace_abbreviation(match):
        """Replaces abbreviation or initial with a placeholder."""
        placeholders.append(match.group())
        return f"__PLACEHOLDER_{len(placeholders) - 1}__"

    def restore_placeholders(sentence):
        """Restores placeholders for abbreviations and initials in a sentence."""
        for i, placeholder in enumerate(placeholders):
            sentence = sentence.replace(f"__PLACEHOLDER_{i}__", placeholder)
        return sentence

    # Initialize a list to store the tokenized sentences
    result = []


    # Split the text by lines since the dataset may contain sentence IDs and sentences separated by a tab
    for line in text.splitlines():
        if "\t" in line:
            # Extract only the sentence text (ignore the sentence ID part)
            _, sentence_text = line.split("\t", 1)

            # Temporarily replace abbreviations and initials
            sentence_text = re.sub(abbreviation_pattern, replace_abbreviation, sentence_text)
            sentence_text = re.sub(initials_pattern, replace_abbreviation, sentence_text)

            # Split sentences based on punctuation
            regex = r"(?<=\.|\!|\?)\s+"
            sentences = re.split(regex, sentence_text.strip())

            # Restore placeholders
            sentences = [restore_placeholders(sentence) for sentence in sentences]

            # Extend the result list with tokenized sentences
            result.extend(sentences)
        else:
            # Handle lines without tabs as plain text
            text = re.sub(abbreviation_pattern, replace_abbreviation, line.strip())
            text = re.sub(initials_pattern, replace_abbreviation, text)
            sentences = re.split(r"(?<=\.|\!|\?)\s+", text.strip())
            sentences = [restore_placeholders(sentence) for sentence in sentences]
            result.extend(sentences)

    return result


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
SNT.001\tDr. Santos is an expert in linguistics. He spoke at the conference yesterday. 
SNT.002\tJust like France. Usually it goes on. Ms. Reyes, on the other hand, presented her research findings.
"""
usable_sentences = split_sentences(text)
print(usable_sentences)
