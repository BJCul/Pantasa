import nlu

def lemmatize_sentence(sentence):
    """
    Calls the nlu lemmatizer to lemmatize a sentence and returns the lemmatized string.
    """
    try:
        # Check if the sentence is enclosed in single quotation marks with a comma before the closing mark
        if sentence.startswith('"') and sentence.endswith('"') and sentence.__contains__(','):
            sentence = sentence[0:-1]  # Remove the opening and closing quotation marks and the comma

        # Load the lemmatizer model
        lemmatizer = nlu.load("tl.lemma")

        # Use the model to predict lemmatized output
        result = lemmatizer.predict(sentence)

        # Extract the lemmatized text from the result
        lemmatized_string = result['lemma'].values[0]

        # Add back the quotation marks if they were removed
        if ',' in sentence:
            lemmatized_string = '"' + lemmatized_string + '"'

        return lemmatized_string
        
    except Exception as e:
        print(f"Exception occurred during lemmatization: {e}")
        return sentence


