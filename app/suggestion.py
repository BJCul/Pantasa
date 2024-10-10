class SuggestionType:
    """
    Enum-like class to define the types of grammar suggestions.
    This will help differentiate between types of corrections (e.g., substitution, deletion, insertion, merging, unmerging).
    """
    SUBSTITUTION = "SUBSTITUTION"
    INSERTION = "INSERTION"
    DELETION = "DELETION"
    MERGING = "MERGING"
    UNMERGING = "UNMERGING"


class SuggestionToken:
    """
    A token within a suggestion representing the word or POS tag that needs correction.
    Each token contains details such as the word, its index in the sentence, the associated cost, and the suggestion type.
    """
    def __init__(self, word, index, cost, pos=None, sugg_type=None):
        self.word = word
        self.index = index  # Position of the word in the sentence
        self.cost = cost  # Edit distance cost for this token
        self.pos = pos  # POS tag associated with the word (if applicable)
        self.sugg_type = sugg_type  # Type of suggestion (e.g., substitution, insertion, deletion, merging)

    def __repr__(self):
        return f"SuggestionToken(word='{self.word}', index={self.index}, cost={self.cost}, pos='{self.pos}', sugg_type='{self.sugg_type}')"


class Suggestion:
    """
    A class that represents a full grammar correction suggestion, including the tokens involved in the suggestion and the total edit distance.
    Suggestions also have a frequency counter that tracks how often a similar suggestion has been made.
    """
    def __init__(self, suggestion_tokens, edit_distance):
        self.suggestion_tokens = suggestion_tokens  # List of SuggestionTokens
        self.edit_distance = edit_distance  # Total edit distance for this suggestion
        self.frequency = 1  # Frequency counter for similar suggestions

    def increment_frequency(self):
        """Increments the frequency count for this suggestion."""
        self.frequency += 1

    def get_suggestions(self):
        """Returns the list of SuggestionTokens."""
        return self.suggestion_tokens

    def get_edit_distance(self):
        """Returns the total edit distance of the suggestion."""
        return self.edit_distance

    def get_suggestion_string(self):
        """
        Returns a string representation of the suggestion, showing the corrected words and their types.
        Example: "Replace word 'bata' with 'bato' (Substitution)"
        """
        suggestion_strs = []
        for token in self.suggestion_tokens:
            if token.sugg_type == SuggestionType.DELETION:
                suggestion_strs.append(
                    f"Delete '{token.word}' at index {token.index} (Cost: {token.cost})"
                )
            elif token.sugg_type == SuggestionType.MERGING:
                suggestion_strs.append(
                    f"Merge '{token.word}' at index {token.index} (Cost: {token.cost})"
                )
            else:
                suggestion_strs.append(
                    f"{token.sugg_type.capitalize()} '{token.word}' at index {token.index} (Cost: {token.cost})"
                )
        return ", ".join(suggestion_strs)

    def __repr__(self):
        return f"Suggestion(edit_distance={self.edit_distance}, frequency={self.frequency}, suggestions={self.suggestion_tokens})"


# Example usage
if __name__ == "__main__":
    # Example tokens for a suggestion (Substitution)
    token1 = SuggestionToken(word="bato", index=2, cost=0.6, pos="NN", sugg_type=SuggestionType.SUBSTITUTION)
    token2 = SuggestionToken(word="malaki", index=3, cost=0.8, pos="JJ", sugg_type=SuggestionType.SUBSTITUTION)

    # Example token for Deletion
    token3 = SuggestionToken(word="mga", index=1, cost=1.0, pos="DT", sugg_type=SuggestionType.DELETION)

    # Example token for Merging
    token4 = SuggestionToken(word="pinalaki", index=4, cost=0.7, sugg_type=SuggestionType.MERGING)

    # Create a suggestion with substitution, deletion, and merging
    suggestion = Suggestion([token1, token2, token3, token4], edit_distance=2.5)

    # Output the suggestion details
    print(suggestion.get_suggestion_string())
    print(suggestion)
