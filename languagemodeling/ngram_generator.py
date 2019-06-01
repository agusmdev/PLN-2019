from collections import defaultdict
import numpy as np


class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self._n = n = model._n
        # compute the probabilities
        probs = defaultdict(dict)

        # sort in descending order for efficient sampling

        sorted_probs = {}
        ngrams = [ngram for ngram in model._count.keys() if len(ngram) == n]

        for ngram in ngrams:
            prev_tokens = ngram[:n - 1]
            last_token = ngram[n - 1]
            prob_last_token = model.cond_prob(last_token, prev_tokens)
            probs[prev_tokens][last_token] = prob_last_token

        # Sort using the following order:
        # - First, sort by the probability (descending)
        # - Second, sort by lexicographical order the tokens (ascending)
        # Which is equivalent to using a key that swaps tuple elements and
        # makes all probabilities negative and sorting that list of tuples
        # using the following order:
        # - First, sort by the first component (ascending)
        # - Second, sort by the second component (ascending)
        # This is the default ordering for tuples thus we can sort each token
        # using:
        for prev_tokens in probs.keys():
            sorted_probs[prev_tokens] = sorted(probs[prev_tokens].items(),
                                               key=lambda x: (-x[1], x[0]))
        self._probs = probs
        self._sorted_probs = sorted_probs

    def generate_sent(self):
        """Randomly generate a sentence."""
        # WORK HERE!!
        n = self._n
        # Add (n-1) opening tags to start the generation
        sent = ["<s>"] * (n - 1)

        while "</s>" not in sent:

            sent += [self.generate_token(tuple(sent[-n + 1:]))]

        # Delete (n-1) opening tags and closing tag previously added
        return sent[n-1:-1]

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self._n
        if n == 1 or not prev_tokens:
            prev_tokens = tuple()

        tokens = self._sorted_probs[prev_tokens]

        elements = [element for element, probability in tokens]

        probabilities = np.array([probability for element, probability in tokens])

        choice = np.random.choice(a=elements, p=probabilities)

        return "".join(choice)
