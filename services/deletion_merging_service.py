from grammar_checking_thread import GrammarCheckingServiceThread
from services.candidate_ngram_service import CandidateNGramService
import time
from suggestion import Suggestion, SuggestionToken, SuggestionType
import utils

class DeletionAndMergingService(GrammarCheckingServiceThread):
    def __init__(self):
        super().__init__()

    def perform_task(self):
        """Process candidate n-grams and generate suggestions for deletions and merging."""
        start_time = time.time()

        # Get candidate rule n-grams that are 1 word shorter than the input n-gram
        candidate_rule_ngrams = CandidateNGramService.get_candidate_ngrams(self.input_pos, len(self.input_pos) - 1)

        for rule in candidate_rule_ngrams:
            edit_distance = 0.0
            i, j = 0, 0
            rule_pos = rule.get_pos()
            rule_words = rule.get_words()
            rule_is_pos_generalized = rule.get_is_pos_generalized()

            suggestion_tokens_del = []  # For Deletion suggestions
            suggestion_tokens_mer = []  # For Merging suggestions

            while i < len(self.input_pos) and j < len(rule_pos):
                if rule_is_pos_generalized and rule_is_pos_generalized[j] and rule_pos[j] == self.input_pos[i]:
                    # If generalized POS allows for match, move both pointers
                    i += 1
                    j += 1
                elif rule_words[j] == self.input_words[i]:
                    # If words match exactly, move both pointers
                    i += 1
                    j += 1
                elif i + 1 < len(self.input_pos) and self.is_equal_when_merged(self.input_words[i], self.input_words[i + 1], rule_words[j]):
                    # Case for Merging: Merge two input words to match rule_word[j]
                    suggestion_tokens_mer.append(SuggestionToken(
                        rule_words[j], i, 0.7, SuggestionType.MERGING
                    ))
                    i += 2  # Skip the next word in the input since we merged
                    j += 1
                    edit_distance += 0.7
                elif i + 1 < len(self.input_pos) and rule_is_pos_generalized and rule_pos[j] == self.input_pos[i + 1]:
                    # Case for Deletion: Remove the current input word
                    suggestion_tokens_del.append(SuggestionToken(
                        self.input_words[i], i, 1.0, self.input_pos[i], SuggestionType.DELETION
                    ))
                    i += 1  # Skip the word in input that needs to be deleted
                    edit_distance += 1.0
                else:
                    # No match, move both pointers with penalty
                    i += 1
                    j += 1
                    edit_distance += 1.0

            # Handle remaining unmatched input or rule words
            if i != len(self.input_pos) or j != len(rule_pos):
                edit_distance += 1.0

            # If we have deletion suggestions and the edit distance is within the threshold, add them to the output
            if len(suggestion_tokens_del) >= 1 and edit_distance <= utils.EDIT_DISTANCE_THRESHOLD:
                self.output_suggestions.append(Suggestion(suggestion_tokens_del, edit_distance))

            # If we have merging suggestions, add them to the output
            if len(suggestion_tokens_mer) >= 1:
                self.output_suggestions.append(Suggestion(suggestion_tokens_mer, edit_distance))

        end_time = time.time()
        print(f"Deletion and Merging task completed in {end_time - start_time:.2f} seconds.")

    def is_equal_when_merged(self, input_left, input_right, rule_word):
        """
        Check if two input words can be merged to match a rule word.
        Example: "pinag sikapan" should be merged to "pinagsikapan".
        """
        input_left = input_left.lower()
        input_right = input_right.lower()
        rule_word = rule_word.lower()

        combined_word = input_left + input_right
        hyphenated_word = input_left + '-' + input_right

        return combined_word == rule_word or hyphenated_word == rule_word

