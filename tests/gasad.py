import os
from collections import defaultdict, LinkedHashSet
from threading import Thread
import re

# Utility function to convert array to string
def array_to_string(arr):
    return ' '.join(arr)

# Utility function to load file contents
def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip().split() for line in file.readlines()]

# WordLemmaPOSMap class to map a word to its lemma, POS tag, and POS ID.
class WordLemmaPOSMap:
    def __init__(self, word, posTag, posID=None, lemma=None):
        self.word = word
        self.posTag = posTag
        self.posID = posID
        self.lemma = lemma

    def get_word(self):
        return self.word

    def get_posTag(self):
        return self.posTag

    def get_posID(self):
        return self.posID

    def get_lemma(self):
        return self.lemma

# HybridNGram class representing hybrid n-grams.
class HybridNGram:
    def __init__(self, posTagsAsString, posIDs, isHybridAsString, nonHybridWordsAsString, baseNGramFrequency):
        self.posTags = posTagsAsString.split(" ")
        self.posIDs = posIDs
        self.nonHybridWords = nonHybridWordsAsString.split(" ")
        self.isHybrid = [True if x == 'True' else False for x in isHybridAsString.split(" ")]
        self.baseNGramFrequency = baseNGramFrequency

    def get_posTags(self):
        return self.posTags

    def get_isHybrid(self):
        return self.isHybrid

    def get_baseNGramFrequency(self):
        return self.baseNGramFrequency

# Suggestion class that provides correction suggestions for n-grams.
class Suggestion:
    def __init__(self, suggType, tokenSuggestions, isHybrid, posSuggestion, affectedIndex, affectedIndexNoOffset, editDistance, frequency):
        self.suggType = suggType
        self.tokenSuggestions = tokenSuggestions
        self.isHybrid = isHybrid
        self.posSuggestion = posSuggestion
        self.affectedIndex = affectedIndex
        self.affectedIndexNoOffset = affectedIndexNoOffset
        self.editDistance = editDistance
        self.frequency = frequency

# Simulated MDAG class to act as a word dictionary.
class MDAG:
    def __init__(self):
        self.words = set()

    def add(self, word):
        self.words.add(word)

    def contains(self, word):
        return word in self.words

# Simulated Stemmer class to check if a word is a prefix.
class Stemmer:
    def __init__(self, prefixes):
        self.prefixes = prefixes

    def isPrefix(self, word):
        return word in self.prefixes

# MergeCorrection class (simulates merging of words based on prefix and suffix rules).
class MergeCorrection:
    def __init__(self, stemmer):
        self.stemmer = stemmer

    def setConsecutiveWords(self, word1, word2):
        mergeWord = ""
        if not word2 == "*#" and self.stemmer.isPrefix(word1):
            if word1.endswith("ag") and re.match(r'[aeiou].*', word2):
                mergeWord = word1 + "-" + word2
            elif word1.endswith("ag") and re.match(r'[a-z&&[^aeiou]].*', word2):
                mergeWord = word1 + word2
        return mergeWord

# SplitCorrection class (splits a misspelled word into two valid words).
class SplitCorrection:
    def __init__(self, filiDict):
        self.filiDict = filiDict

    def splitSuggestion(self, misspelled):
        suggestionList = set()
        for counter in range(2, len(misspelled) - 1):
            first = misspelled[:counter]
            second = misspelled[counter:]
            if self.filiDict.contains(first) and self.filiDict.contains(second):
                suggestionList.add(first + " " + second)
        return suggestionList

# Simulating NGramDao and POS_NGram_Indexer using dictionaries.
class NGramDao:
    def __init__(self, ngram_size):
        self.ngram_size = ngram_size
        self.ngrams = []
        self.ngram_id_counter = 1

    def add(self, words, lemmas, pos):
        ngram_entry = {
            'id': self.ngram_id_counter,
            'words': words,
            'lemmas': lemmas,
            'pos': pos
        }
        self.ngrams.append(ngram_entry)
        self.ngram_id_counter += 1
        return ngram_entry['id']

    def get_similar_ngrams(self, similarity_threshold):
        return self.ngrams

    def delete(self, ngram_id):
        self.ngrams = [ngram for ngram in self.ngrams if ngram['id'] != ngram_id]

class POS_NGram_Indexer:
    def __init__(self):
        self.index = defaultdict(list)

    def add(self, pos, ngram_id):
        self.index[array_to_string(pos)].append(ngram_id)

# NGramPopulatorThread now uses WordLemmaPOSMap for mapping word, lemma, and POS
class NGramPopulatorThread(Thread):
    def __init__(self, word_files, lemma_files, pos_files, ngram_size):
        Thread.__init__(self)
        self.word_files = word_files
        self.lemma_files = lemma_files
        self.pos_files = pos_files
        self.ngram_size = ngram_size

    def run(self):
        ngram_dao = NGramDao(self.ngram_size)
        indexer = POS_NGram_Indexer()

        for word_file, lemma_file, pos_file in zip(self.word_files, self.lemma_files, self.pos_files):
            words = load_file(word_file)
            lemmas = load_file(lemma_file)
            pos_tags = load_file(pos_file)

            for sentence_words, sentence_lemmas, sentence_pos in zip(words, lemmas, pos_tags):
                for i in range(len(sentence_words) - self.ngram_size + 1):
                    ngram_words = sentence_words[i:i+self.ngram_size]
                    ngram_lemmas = sentence_lemmas[i:i+self.ngram_size]
                    ngram_pos = sentence_pos[i:i+self.ngram_size]

                    word_lemma_pos_maps = [WordLemmaPOSMap(w, p, None, l) for w, l, p in zip(ngram_words, ngram_lemmas, ngram_pos)]

                    # Add to database (simulated)
                    ngram_id = ngram_dao.add(ngram_words, ngram_lemmas, ngram_pos)
                    indexer.add(ngram_pos, ngram_id)

class RulesGeneralizationService:
    def __init__(self):
        self.rules = set()

    def generalize(self, ngram_size, ngram_dao):
        ngrams = ngram_dao.get_similar_ngrams(similarity_threshold=2)
        for ngram in ngrams:
            words = ngram['words']
            pos_tags = ngram['pos']

            is_pos_generalized = [False] * ngram_size
            generalization_map = {}

            for i in range(ngram_size):
                word_set = set(ngram_dao.ngrams[j]['words'][i].lower() for j in range(len(ngram_dao.ngrams)))
                if len(word_set) > 1:
                    is_pos_generalized[i] = True

            for ngram in ngram_dao.ngrams:
                hybrid_ngram = HybridNGram(array_to_string(pos_tags), None, array_to_string(is_pos_generalized), array_to_string(words), 1)
                generalized_rule = []
                for i in range(ngram_size):
                    if is_pos_generalized[i]:
                        generalized_rule.append(pos_tags[i])
                    else:
                        generalized_rule.append(words[i])
                
                self.rules.add(array_to_string(generalized_rule))
        
        for rule in self.rules:
            print(rule)
        print(f"Generated {len(self.rules)} rules.")

# Simulating the split and merge operations during n-gram processing.
def simulate_ngram_processing(word_files, lemma_files, pos_files):
    ngram_sizes = [2, 3]
    threads = []

    for ngram_size in ngram_sizes:
        thread = NGramPopulatorThread(word_files, lemma_files, pos_files, ngram_size)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    generalization_service = RulesGeneralizationService()
    for ngram_size in ngram_sizes:
        ngram_dao = NGramDao(ngram_size)
        generalization_service.generalize(ngram_size, ngram_dao)

# Example usage
if __name__ == "__main__":
    # Simulate file paths (replace these with actual file paths for real usage)
    word_files = ["words.txt"]
    lemma_files = ["lemmas.txt"]
    pos_files = ["pos.txt"]

    # Simulated MDAG dictionary
    filiDict = MDAG()
    filiDict.add("ang")
    filiDict.add("bata")
    filiDict.add("ng")
    filiDict.add("maganda")

    # Simulated Stemmer with prefixes
    stemmer = Stemmer(["pag", "mag"])

    # Creating the correction objects
    merge_correction = MergeCorrection(stemmer)
    split_correction = SplitCorrection(filiDict)

    # Test the merge function
    merged_word = merge_correction.setConsecutiveWords("pag", "ibig")
    print(f"Merged word: {merged_word}")

    # Test the split function
    split_suggestions = split_correction.splitSuggestion("angbata")
    print(f"Split suggestions: {split_suggestions}")

    # Simulate n-gram processing
    simulate_ngram_processing(word_files, lemma_files, pos_files)
