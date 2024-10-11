from preprocess import preprocess_text
from ngram_matching import compare_with_hybrid_ngrams, weighted_levenshtein
from services.substitution_service import SubstitutionService
from services.insertion_unmerging_service import InsertionAndUnmergingService
from services.deletion_merging_service import DeletionAndMergingService
from utils import load_hybrid_ngram_patterns, process_sentence_with_dynamic_ngrams

class pantasa:
    def __init__(self, is_verbose=False, is_generate_text_file=False, ngram_size_to_get=5):
        self.is_verbose = is_verbose
        self.is_generate_text_file = is_generate_text_file
        self.ngram_size_to_get = ngram_size_to_get
        self.hybrid_ngram_patterns = load_hybrid_ngram_patterns('data/processed/hngrams.csv')

    def get_grammar_suggestions(self, sentence):
        """
        Returns grammar suggestions for the given input sentence by performing grammar checks.
        """
        preprocessed_output = preprocess_text(sentence)
        if not preprocessed_output:
            print("Error during preprocessing.")
            return []

        tokens, lemmas, pos_tags = preprocessed_output[0]
        input_data = Input(tokens, lemmas, pos_tags)

        suggestions = self.check_grammar_with_input(input_data)
        return [sugg.get_suggestions() for sugg in suggestions]

    def check_grammar_with_input(self, input_data):
        """
        Processes grammar checking based on preprocessed input data (tokens, lemmas, POS tags).
        """
        top_suggestions = []

        # Iterate over n-gram sizes in reverse order (from largest n-gram to smallest)
        for ngram_size in range(self.ngram_size_to_get, 1, -1):
            suggestions = self.get_ngram_suggestions(ngram_size, input_data)
            top_suggestions.extend(suggestions)
            
            if len(suggestions) > 0:
                break

        return top_suggestions

    def get_ngram_suggestions(self, ngram_size, input_data):
        """
        Generate grammar suggestions based on n-grams.
        """
        ngram_suggestions = []
        ngram_collections = process_sentence_with_dynamic_ngrams(input_data.words)

        # Process n-grams of the given size
        for ngram in ngram_collections.get(f'{ngram_size}-gram', []):
            w_arr = ngram
            p_arr = input_data.pos[:ngram_size]
            l_arr = input_data.lemmas[:ngram_size]

            # Substitution Service
            sub_service = SubstitutionService()
            sub_service.set_input_values(w_arr, l_arr, p_arr, ngram_size)
            sub_service.run()
            suggestions = sub_service.get_suggestions()

            # Insertion and Unmerging Service
            if ngram_size < self.ngram_size_to_get:
                ins_unm_service = InsertionAndUnmergingService()
                ins_unm_service.set_input_values(w_arr, l_arr, p_arr, ngram_size)
                ins_unm_service.run()
                suggestions.extend(ins_unm_service.get_suggestions())

            # Deletion and Merging Service
            if ngram_size > 1:
                del_mer_service = DeletionAndMergingService()
                del_mer_service.set_input_values(w_arr, l_arr, p_arr, ngram_size)
                del_mer_service.run()
                suggestions.extend(del_mer_service.get_suggestions())

            ngram_suggestions.extend(suggestions)

        return ngram_suggestions


class Input:
    """
    Class for holding input data - words, lemmas, and POS tags.
    """
    def __init__(self, words, lemmas, pos):
        self.words = words
        self.lemmas = lemmas
        self.pos = pos


# Example Usage
if __name__ == "__main__":
    grammar_checker = pantasa(is_verbose=True, is_generate_text_file=False)
    sentence = "kumain ang mga bata ng mansanas"
    suggestions = grammar_checker.get_grammar_suggestions(sentence)
    
    for sugg in suggestions:
        print(sugg)
