import time

class CandidateNGramService:
    """
    Service responsible for fetching candidate n-grams based on input POS tags and n-gram size.
    This service interacts with an indexer and n-gram DAO to retrieve candidate n-grams.
    """

    _instance = None  # Singleton instance

    def __init__(self):
        if CandidateNGramService._instance is not None:
            raise Exception("This is a singleton class. Use get_instance() to get the single instance.")
        CandidateNGramService._instance = self

    @staticmethod
    def get_instance():
        """Returns the singleton instance of CandidateNGramService."""
        if CandidateNGramService._instance is None:
            CandidateNGramService()
        return CandidateNGramService._instance

    def get_candidate_ngrams(self, input_pos, ngram_size):
        """
        Fetches candidate n-grams based on input POS tags and the size of the n-gram.
        Args:
        - input_pos (list of str): POS tags from the input sentence.
        - ngram_size (int): The size of the n-gram (e.g., 2, 3, 4, etc.).

        Returns:
        - List of NGram objects representing the candidate n-grams.
        """
        start_time = time.time()

        # Get NGramDao and POS_NGram_Indexer based on the n-gram size
        ngram_dao = DaoManager.get_ngram_dao(ngram_size)
        indexer = DaoManager.get_indexer(ngram_size)

        # Get unique POS tags from the input
        unique_pos = self.get_unique_pos(input_pos)

        # Dictionary to store instance frequencies
        instances_frequency = {}

        # Look up n-grams by POS
        for pos_tag in unique_pos:
            instances = indexer.get_instances(pos_tag)
            for instance in instances:
                if instance not in instances_frequency:
                    instances_frequency[instance] = 0
                instances_frequency[instance] += 1

        # Filter candidate n-grams
        candidate_ngrams = []
        for ngram_id, frequency in instances_frequency.items():
            if frequency >= len(unique_pos) - 2:  # Allow up to 2 mismatches
                ngram = ngram_dao.get(ngram_id)
                if ngram:
                    candidate_ngrams.append(ngram)

        end_time = time.time()
        print(f"Time taken to fetch candidate n-grams: {end_time - start_time} seconds")

        return candidate_ngrams

    def get_unique_pos(self, pos_arr):
        """
        Returns unique POS tags from the input array.
        Args:
        - pos_arr (list of str): List of POS tags from the input sentence.

        Returns:
        - List of unique POS tags.
        """
        return list(set(pos_arr))


# Simulated DaoManager for demonstration purposes
class DaoManager:
    @staticmethod
    def get_ngram_dao(ngram_size):
        return NGramDao()

    @staticmethod
    def get_indexer(ngram_size):
        return POS_NGram_Indexer()


# Simulated NGramDao and POS_NGram_Indexer classes
class NGramDao:
    def get(self, ngram_id):
        """Simulate fetching an NGram from a database by its ID."""
        # This would normally retrieve the n-gram details from a database
        return NGram(ngram_id, ["example_word1", "example_word2"], ["NN", "VB"])


class POS_NGram_Indexer:
    def get_instances(self, pos_tag):
        """Simulate returning a list of instance IDs where the POS tag appears."""
        # This simulates the indexing of n-grams by POS tags
        return [1, 2, 3]  # Example instance IDs


# Simulated NGram class for demonstration purposes
class NGram:
    def __init__(self, ngram_id, words, pos_tags):
        self.ngram_id = ngram_id
        self.words = words
        self.pos_tags = pos_tags

    def get_words(self):
        return self.words

    def get_pos(self):
        return self.pos_tags

    def get_lemmas(self):
        # Simulate lemmatization output for the words in the n-gram
        return [word.lower() for word in self.words]

    def __repr__(self):
        return f"NGram(ID: {self.ngram_id}, Words: {self.words}, POS: {self.pos_tags})"


# Example usage
if __name__ == "__main__":
    pos_input = ["NN", "VB", "DT", "NN"]
    ngram_size = 3

    # Get the singleton instance of the CandidateNGramService
    candidate_ngram_service = CandidateNGramService.get_instance()

    # Fetch candidate n-grams based on the input POS tags
    candidates = candidate_ngram_service.get_candidate_ngrams(pos_input, ngram_size)

    # Output the candidate n-grams
    for candidate in candidates:
        print(candidate)

