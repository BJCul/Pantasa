from services.grammar_checking_thread import GrammarCheckingServiceThread
import time
from services.candidate_ngram_service import CandidateNGramService
from utils import weighted_levenshtein
from suggestion import Suggestion, SuggestionToken, SuggestionType

class SubstitutionService(GrammarCheckingServiceThread):
    def __init__(self):
        super().__init__()
        self.candidate_ngram_service = CandidateNGramService()  # Initialize the service

    def perform_task(self):
        """Perform the substitution grammar-checking task."""
        start_time = time.time()
        
        # Fetch candidate n-grams based on the input POS sequence and n-gram size
        candidate_rule_ngrams = self.candidate_ngram_service.get_candidate_ngrams(self.input_pos, len(self.input_pos))
        
        end_time = time.time()
        print(f"Candidate n-grams fetched in {end_time - start_time} seconds")
        
        # Iterate over candidate n-grams to generate suggestions
        for rule in candidate_rule_ngrams:
            rule_pos = rule.pos_tags
            rule_words = rule.words
            rule_lemmas = rule.lemmas
            rule_is_pos_generalized = rule.is_pos_generalized

            # Initialize edit distance and replacement suggestion list
            edit_distance = 0.0
            replacements = []

            # Compare input n-grams with rule n-grams word by word
            for i in range(len(rule_pos)):
                if rule_lemmas[i] == self.input_lemmas[i] and rule_words[i] != self.input_words[i] and rule_pos[i] == self.input_pos[i]:
                    # Word mismatch with same lemma and POS tag (substitution)
                    edit_distance += 0.6
                    replacements.append(SuggestionToken(rule_words[i], i, edit_distance, rule_pos[i], SuggestionType.SUBSTITUTION))
                elif rule_words[i] != self.input_words[i]:
                    if rule_lemmas[i] == self.input_lemmas[i]:
                        # Word mismatch but lemma is the same (lower weight for substitution)
                        edit_distance += 0.6
                    elif self.within_spelling_edit_distance(rule_words[i], self.input_words[i]):
                        # Words are similar based on spelling edit distance
                        edit_distance += 0.65
                    elif rule_pos[i] == self.input_pos[i]:
                        # Word mismatch with same POS tag
                        edit_distance += 0.8
                    else:
                        # General substitution with higher weight
                        edit_distance += 1.0

                    # Append the replacement suggestion
                    replacements.append(SuggestionToken(rule_words[i], i, edit_distance, rule_pos[i], SuggestionType.SUBSTITUTION))

            # If the edit distance is within the acceptable threshold, keep the suggestion
            if edit_distance <= 1.0:
                self.add_suggestion(replacements, edit_distance)

    def within_spelling_edit_distance(self, corpus_word, input_word):
        """Check if two words are within the spelling edit distance using the Levenshtein algorithm."""
        corpus_word = corpus_word.lower()
        input_word = input_word.lower()
        distance = weighted_levenshtein(corpus_word, input_word)
        
        if distance <= 1 and len(input_word) <= 4:
            return True
        elif distance <= 2 and 4 < len(input_word) <= 12:
            return True
        elif distance <= 3 and len(input_word) > 12:
            return True
        return False

    def add_suggestion(self, replacements, edit_distance):
        """Add a suggestion to the output list."""
        has_similar = False
        
        # Check if a similar suggestion already exists
        for suggestion in self.output_suggestions:
            if self.are_suggestions_similar(suggestion, replacements, edit_distance):
                suggestion.increment_frequency()
                has_similar = True
                break
        
        if not has_similar:
            # Add new suggestion to the list
            self.output_suggestions.append(Suggestion(replacements, edit_distance))

    def are_suggestions_similar(self, existing_suggestion, new_suggestions, edit_distance):
        """Check if the new suggestion is similar to an existing suggestion."""
        if existing_suggestion.get_edit_distance() == edit_distance:
            for token1, token2 in zip(existing_suggestion.get_suggestions(), new_suggestions):
                if token1.word != token2.word:
                    return False
            return True
        return False

