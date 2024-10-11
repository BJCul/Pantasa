from grammar_checking_thread import GrammarCheckingServiceThread
from services.candidate_ngram_service import CandidateNGramService
import time
from suggestion import Suggestion, SuggestionToken, SuggestionType
from services.candidate_ngram_service import CandidateNGramService
import utils

class InsertionAndUnmergingService(GrammarCheckingServiceThread):
    def __init__(self):
        super().__init__()

    def perform_task(self):
        """Perform the insertion and unmerging grammar-checking task."""
        start_time = time.time()

        # Get candidate rule n-grams that are 1 word longer than the input n-gram
        candidate_rule_ngrams = CandidateNGramService.get_candidate_ngrams(self.input_pos, len(self.input_pos) + 1)

        for rule in candidate_rule_ngrams:
            edit_distance = 0.0
            i, j = 0, 0
            rule_pos = rule.get_pos()
            rule_words = rule.get_words()
            rule_is_pos_generalized = rule.get_is_pos_generalized()
            suggestion_token = None

            while i < len(self.input_pos) and j < len(rule_pos):
                if rule_is_pos_generalized and rule_is_pos_generalized[j] and rule_pos[j] == self.input_pos[i]:
                    # If the rule allows for generalized POS and they match, move both pointers
                    i += 1
                    j += 1
                elif rule_words[j] == self.input_words[i]:
                    # If words match exactly, move both pointers
                    i += 1
                    j += 1
                elif rule_is_pos_generalized and j + 1 < len(rule_pos) and rule_is_pos_generalized[j + 1] and rule_pos[j + 1] == self.input_pos[i]:
                    # Case for Insertion: Insert a missing word (rule_word[j])
                    suggestion_token = SuggestionToken(
                        rule_words[j], i, 1.0, rule_pos[j], SuggestionType.INSERTION
                    )
                    i += 1
                    j += 2  # Skip the inserted word in the rule
                    edit_distance += 1.0
                elif j + 1 < len(rule_words) and rule_words[j + 1] == self.input_words[i]:
                    # Case for Insertion: Insert a missing word when rule_word[j + 1] matches input
                    suggestion_token = SuggestionToken(
                        rule_words[j], i, 1.0, rule_pos[j], SuggestionType.INSERTION
                    )
                    i += 1
                    j += 2  # Skip the inserted word in the rule
                    edit_distance += 1.0
                elif j + 1 < len(rule_words) and self.is_equal_to_unmerge(self.input_words[i], rule_words[j], rule_words[j + 1]):
                    # Case for Unmerging: Unmerge the word into two words (rule_words[j], rule_words[j + 1])
                    suggestion_token = SuggestionToken(
                        f"{rule_words[j]} {rule_words[j + 1]}", i, 0.7, SuggestionType.UNMERGING
                    )
                    i += 1
                    j += 2  # Skip the next word in the rule
                    edit_distance += 0.7
                else:
                    # If none of the above cases match, move both pointers (penalty)
                    i += 1
                    j += 1
                    edit_distance += 1.0

            # Add penalty for remaining unmatched words
            if i != len(self.input_pos) or j != len(rule_pos):
                edit_distance += 1.0

            # If a suggestion is found and the edit distance is within the threshold, add it to the suggestions
            if suggestion_token and edit_distance <= utils.EDIT_DISTANCE_THRESHOLD:
                has_similar = False
                for existing_suggestion in self.output_suggestions:
                    if self.are_suggestions_similar(existing_suggestion, suggestion_token, edit_distance):
                        existing_suggestion.increment_frequency()
                        has_similar = True
                        break

                if not has_similar:
                    self.output_suggestions.append(Suggestion([suggestion_token], edit_distance))

        end_time = time.time()
        print(f"Insertion and Unmerging task completed in {end_time - start_time:.2f} seconds.")

    def is_equal_to_unmerge(self, input_word, rule_left, rule_right):
        """
        Check if a word should be unmerged into two words.
        Example: "pinagsikapan" should be unmerged to "pinag" and "sikapan".
        """
        input_word = input_word.lower()
        rule_left = rule_left.lower()
        rule_right = rule_right.lower()

        combined_word = rule_left + rule_right
        hyphenated_word = f"{rule_left}-{rule_right}"

        return input_word == combined_word or input_word == hyphenated_word

    def are_suggestions_similar(self, suggestion, suggestion_token, edit_distance):
        """
        Check if a suggestion is similar to the current one to avoid duplicates.
        """
        existing_token = suggestion.get_suggestions()[0]
        if suggestion.get_edit_distance() == edit_distance and existing_token.sugg_type == suggestion_token.sugg_type:
            if existing_token.pos == suggestion_token.pos and existing_token.word == suggestion_token.word:
                return True
        return False
