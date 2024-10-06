from py4j.java_gateway import JavaGateway

# Initialize the connection to the Java Gateway Server
def initialize_stemmer():
    gateway = JavaGateway()  # Connect to the running Java Gateway Server
    lemmatizer = gateway.entry_point  # Access the Lemmatizer object from Java
    return gateway, lemmatizer

# Function to lemmatize a single word
def lemmatize_single_word(word, lemmatizer):
    return lemmatizer.lemmatizeSingle(word)

# Function to lemmatize multiple words
def lemmatize_multiple_words(words, gateway, lemmatizer):
    # Convert Python list to Java array
    java_words_array = gateway.new_array(gateway.jvm.String, len(words))
    for i, word in enumerate(words):
        java_words_array[i] = word

    # Call the lemmatizeMultiple method with the Java array
    lemmas_java_array = lemmatizer.lemmatizeMultiple(java_words_array)

    # Convert the Java array back to a Python list
    lemmas = [lemmas_java_array[i] for i in range(len(lemmas_java_array))]
    return lemmas

# Call initialize_stemmer at the start
gateway, lemmatizer = initialize_stemmer()

# Test: Lemmatize a single word
lemma = lemmatize_single_word("kabataan", lemmatizer)
print(f"Lemmatized word: {lemma}")

# Test: Lemmatize multiple words
words = ["kumain", "nagulumihanan", "hihikbian", "umiinom"]
lemmas = lemmatize_multiple_words(words, gateway, lemmatizer)
print(f"Lemmatized words: {lemmas}")

