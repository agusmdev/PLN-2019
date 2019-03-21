# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import math
import numpy as np


class LanguageModel(object):

    def tag_sentence(self, sent):
        return ["<s>"]*(self._n - 1) + sent + ["</s>"]

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        return 0.0

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        return -math.inf

    def log_prob(self, sents):
        """Log-probability of a list of sentences.

        sents -- the sentences.
        """

        return sum([self.sent_log_prob(sent) for sent in sents])

    def cross_entropy(self, sents):
        """Cross-entropy of a list of sentences.

        sents -- the sentences.
        """
        M = sum([len(i) for i in sents])
        #M = len(set((i for i in sents)))

        #return sum([self.sent_log_prob(sent) for sent in sents]) / M
        return self.log_prob(sents) / M

    def perplexity(self, sents):
        """Perplexity of a list of sentences.

        sents -- the sentences.
        """
        return 2**(-self.cross_entropy(sents))


class NGram(LanguageModel):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self._n = n

        count = defaultdict(int)

        for sent in sents:
            sent = self.tag_sentence(sent)
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i:i+n])
                count[ngram] += 1
                count[ngram[:-1]] += 1

        self._count = dict(count)

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self._count.get(tokens, 0)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        if not prev_tokens:
            prev_tokens = tuple()

        prev_tokens = tuple(prev_tokens)
        tokens = prev_tokens + (token,)

        # ngram condicional probs are based on relative counts
        W1 = self.count(tokens)
        W0 = self.count(prev_tokens)
        try:
            return float(W1 / W0)
        except ZeroDivisionError:
            return np.nan_to_num(-np.inf)

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        sent = self.tag_sentence(sent)
        n = self._n
        range_ = range(n-1, len(sent))
        cp = self.cond_prob

        return np.prod([cp(sent[i], sent[i-n+1:i]) for i in range_])

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        sent = self.tag_sentence(sent)
        n = self._n
        range_ = range(n-1, len(sent))
        cp = self.cond_prob

        try:
            return sum([math.log2(cp(sent[i], sent[i-n+1:i])) for i in range_])
        except ValueError:
            return float('-inf')


class AddOneNGram(NGram):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        # call superclass to compute counts
        super().__init__(n, sents)

        # compute vocabulary

        voc = ["</s>"]
        for i in sents:
            voc += i
        self._voc = voc = set(voc)

        self._V = len(voc)  # vocabulary size

        sents = [i + ["</s>"] for i in sents]

    def V(self):
        """Size of the vocabulary.
        """
        return self._V

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """

        if self._n == 1 or not prev_tokens:
            prev_tokens = tuple()
        tokens = tuple(prev_tokens) + (token,)
        return float(self.count(tokens) + 1) / float(self.count(tuple(prev_tokens)) + self.V())


class InterpolatedNGram(NGram):

    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        self._n = n

        if gamma is not None:
            # everything is training data
            train_sents = sents
        else:
            # 90% training, 10% held-out
            m = int(0.9 * len(sents))
            train_sents = sents[:m]
            held_out_sents = sents[m:]

        print('Computing counts...')
        # WORK HERE!!
        # COMPUTE COUNTS FOR ALL K-GRAMS WITH K <= N

        # compute vocabulary size for add-one in the last step
        self._addone = addone
        if addone:
            print('Computing vocabulary...')
            self._voc = voc = set()
            # WORK HERE!!

            self._V = len(voc)

        # compute gamma if not given
        if gamma is not None:
            self._gamma = gamma
        else:
            print('Computing gamma...')
            # WORK HERE!!
            # use grid search to choose gamma

    def count(self, tokens):
        """Count for an k-gram for k <= n.

        tokens -- the k-gram tuple.
        """
        # WORK HERE!! (JUST A RETURN STATEMENT)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # WORK HERE!!
