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
        count = defaultdict(int)

        if gamma is not None:
            # everything is training data
            train_sents = sents
        else:
            # 90% training, 10% held-out
            m = int(0.9 * len(sents))
            train_sents = sents[:m]
            held_out_sents = sents[m:]

        print('Computing counts...')
        # COMPUTE COUNTS FOR ALL K-GRAMS WITH K <= N
        for sent in train_sents:
            sent = self.tag_sentence(sent)
            # counts now holds all k-grams for 0 < k < n + 1
            for k in range(n+1):
                # move along the sent saving all its k-grams
                for i in range(n-k, len(sent) - k + 1):
                    ngram = tuple(sent[i: i + k])
                    count[ngram] += 1

        self._count = count

        # compute vocabulary size for add-one in the last step
        self._addone = addone
        if addone:
            print('Computing vocabulary...')

            voc = ["</s>"]
            for i in sents:
                voc += i
            self._voc = voc = set(voc)

            self._V = len(voc)

        # compute gamma if not given
        if gamma is not None:
            self._gamma = gamma
        else:
            print('Computing gamma...')

            gammas = dict()
            for i in range(1, 20):
                self._gamma = i**2
                perp = self.perplexity(held_out_sents)
                gammas[perp] = i**2
            #   print("Perplexity:{}\nGamma:{}\n".format(perp, i**2))

            self._gamma = gammas[min(gammas.keys())]

    def count(self, tokens):
        """Count for an k-gram for k <= n.

        tokens -- the k-gram tuple.
        """
        return self._count.get(tokens, 0)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self._n

        if not prev_tokens:
            prev_tokens = tuple()

        prev_tokens = tuple(prev_tokens)

        cond_prob = 0.0
        for i in range(1, n + 1):
            lambda_ = self.lambda_(i, prev_tokens)
            if lambda_:
                i_cond_prob = self._cond_prob_ML(i, token, prev_tokens)
                cond_prob += lambda_ * i_cond_prob
        return cond_prob

    def _cond_prob_ML(self, i, token, prev_tokens):
        """
        Conditional probability of the given order of a token.
        i -- the first token in prev_tokens to be considered
        token -- the token.
        prev_tokens -- the previous n-1 tokens
        """
        n = self._n

        prev_tokens = prev_tokens[i - 1:]
        tokens = prev_tokens + (token,)

        tokens_count = float(self.count(tokens))
        prev_tokens_count = float(self.count(prev_tokens))

        if self._addone and i == n:
            return (tokens_count + 1) / (prev_tokens_count + self._V)
        elif tokens_count == 0:
            return 0.0
        else:
            return tokens_count / prev_tokens_count

    def lambda_(self, i, tokens):
        """
        Function to find lambda_i
        i -- the first token in tokens.
        tokens -- the tokens.
        """
        n = self._n

        lambda_i = 1 - sum([self.lambda_(j, tokens) for j in range(1, i)])

        if i == n:
            return lambda_i
        else:
            count = self.count(tokens[i - 1:])
            return lambda_i * count / (count + self._gamma)


class BackOffNGram(NGram):

    def __init__(self, n, sents, beta=None, addone=True):
        """
        Back-off NGram model with discounting as described by Michael Collins.

        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        beta -- discounting hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        self._n = n
        self._beta = beta
        self._addone = addone
        self._count = counts = defaultdict(int)
        self._A_set = defaultdict(set)

        voc = ["</s>"]
        for i in sents:
            voc += i
        self._voc = voc = set(voc)
        self._V = len(voc)

        if self._beta is None:
            # 90% training, 10% held-out
            m = int(0.9 * len(sents))
            train_sents = sents[:m]
            held_out_sents = sents[m:]

        else:
            train_sents = sents

        for sent in train_sents:
            sent = self.tag_sentence(sent)
            for j in range(n+1):
                for i in range(n-j, len(sent) - j + 1):
                    ngram = tuple(sent[i: i + j])
                    counts[ngram] += 1
                    if j:
                        self._A_set[ngram[:-1]].add(ngram[-1])

        for i in range(1, n):
            counts[('<s>',)*i] += len(train_sents)
        counts[('</s>',)] = len(train_sents)

        self._count = counts

        if self._beta is None:
            self._beta = self.search_beta(held_out_sents)

    def search_beta(self, held_out_sents):
        betas = dict()
        for i in range(5, 10):
            self._beta = i*0.1
            perp = self.perplexity(held_out_sents)
            betas[perp] = i*0.1
            # print("Perplexity:{}\nBeta:{}\n".format(perp, i*0.1))

        return betas[min(betas.keys())]

    def A(self, tokens):
        """Set of words with counts > 0 for a k-gram with 0 < k < n.

        tokens -- the k-gram tuple.
        """
        if not tokens:
            tokens = ()
        return self._A_set[tuple(tokens)]

    def count_star(self, tokens):
        """
        Discounting counts for counts > 0
        """
        return self._count[tokens] - self._beta

    def alpha(self, tokens):
        """Missing probability mass for a k-gram with 0 < k < n.

        tokens -- the k-gram tuple.
        """
        if not tokens:
            tokens = tuple()

        A_set = self.A(tokens)

        return self._beta * len(A_set) / self.count(tuple(tokens)) if A_set else 1

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """

        addone = self._addone

        # unigram case
        if not prev_tokens:
            if addone:
                return (self.count((token,)) + 1) / (self.V() + self.count(()))
            else:
                return self.count((token,)) / self.count(())
        else:
            # check if we can apply discounting

            if token in self.A(prev_tokens):
                return self.count_star(tuple(prev_tokens) + (token,)) /\
                                                self.count(tuple(prev_tokens))
            else:
                # recursive call
                qD = self.cond_prob(token, prev_tokens[1:])
                denom_factor = self.denom(prev_tokens)
                try:
                    return self.alpha(prev_tokens) * qD / denom_factor
                except ZeroDivisionError:
                    return 0

    def denom(self, tokens):
        """Normalization factor for a k-gram with 0 < k < n.
        tokens -- the k-gram tuple.
        """
        return 1 - sum([self.cond_prob(i, tokens[1:]) for i in self.A(tokens)])

    def V(self):
        """Size of the vocabulary.
        """
        return self._V
