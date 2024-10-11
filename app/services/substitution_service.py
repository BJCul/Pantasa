from services.grammar_checking_thread import GrammarCheckingServiceThread
import time
from utils import load_hybrid_ngram_patterns, weighted_levenshtein
from suggestion import Suggestion, SuggestionToken, SuggestionType

class SubstitutionService(GrammarCheckingServiceThread):
    def __init__(self):
        super().__init__()
        self.hybrid_ngram_patterns = load_hybrid_ngram_patterns('data/processed/hngrams.csv')  # Load hybrid n-grams

    def perform_task(self):
        """Perform the substitution grammar-checking task using hybrid n-grams."""
        start_time = time.time()
        
        # Iterate over hybrid n-gram patterns and compare with the input n-gram
        for pattern in self.hybrid_ngram_patterns:
            hybrid_ngram = pattern['ngram_pattern']
            
            # Check for a match between the input n-gram and the hybrid n-gram
            if self.match_input_with_hybrid_ngram(hybrid_ngram):
                edit_distance = 0.0
                replacements = []

                # Compare input n-gram with the hybrid n-gram word by word
                for i in range(len(hybrid_ngram)):
                    input_word = self.input_words[i]
                    input_pos = self.input_pos[i]
                    hybrid_pos = hybrid_ngram[i]

                    # Check if the POS tags and words match
                    if hybrid_pos == input_pos and input_word != hybrid_pos:
                        edit_distance += 0.6
                        replacements.append(SuggestionToken(input_word, i, edit_distance, hybrid_pos, SuggestionType.SUBSTITUTION))
                    else:
                        edit_distance += 1.0  # General substitution with higher weight

                # Add the suggestion if edit distance is acceptable
                if edit_distance <= 1.0:
                    self.add_suggestion(replacements, edit_distance)

        end_time = time.time()
        print(f"Substitution task completed in {end_time - start_time:.2f} seconds.")

    def match_input_with_hybrid_ngram(self, hybrid_ngram):
        """Check if the input n-gram matches the hybrid n-gram pattern."""
        return len(hybrid_ngram) == len(self.input_pos) and all(
            hybrid_ngram[i] == self.input_pos[i] for i in range(len(hybrid_ngram))
        )
