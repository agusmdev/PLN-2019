from collections import defaultdict
from collections import Counter
from math import log
import numpy as np


def log2(x):
    try:
        return log(x, 2)
    except ValueError:
        return float('-inf')


class HMM:

    def __init__(self, n, tagset, trans, out):
        """
        n -- n-gram size.
        tagset -- set of tags.
        trans -- transition probabilities dictionary.
        out -- output probabilities dictionary.
        """
        self.n = n
        self._out = out
        self._tagset = tagset
        self._trans = trans

    def tagset(self):
        """Returns the set of tags.
        """
        return self._tagset

    def trans_prob(self, tag, prev_tags):
        """Probability of a tag.
        tag -- the tag.
        prev_tags -- iterable with the previous n-1 tags
        """
        prev_tags = tuple(prev_tags)
        try:
            return self._trans[prev_tags][tag]
        except KeyError:
            return 0.0

    def log_trans_prob(self, tag, prev_tags):
        """Log probability of a tag.
        tag -- the tag.
        prev_tags -- iterable with the previous n-1 tags
        """
        return log2(self.trans_prob(tag, prev_tags))

    def out_prob(self, word, tag):
        """Probability of a word given a tag.
        word -- the word.
        tag -- the tag.
        """
        try:
            return self._out[tag][word]
        except KeyError:
            return 0.0

    def log_out_prob(self, word, tag):
        """Log probability of a word given a tag.
        word -- the word.
        tag -- the tag.
        """
        return log2(self.out_prob(word, tag))

    def add_tags(self, y):
        """Add both opening and closing tags to a list of tags or words
        y -- tagging or sentence.
        """
        return ['<s>'] * (self.n - 1) + y + ['</s>']

    def tag_prob(self, y):
        """Probability of a tagging.
        Warning: subject to underflow problems.
        y -- tagging.
        """
        n = self.n
        y = self.add_tags(y)

        prob = 1
        for i in range(n - 1, len(y)):
            tag = y[i]
            prev_tags = tuple(y[i - n + 1:i])
            prob *= self.trans_prob(tag, prev_tags)
        return prob

    def prob(self, word, y):
        """
        Joint probability of a sentence and its tagging.
        Warning: subject to underflow problems.
        word -- sentence.
        y -- tagging.
        """
        tag_prob = [self.tag_prob(y)]
        prob = np.asarray([self.out_prob(word, tag) for word, tag in zip(word, y)] + tag_prob)
        return prob.prod()

    def tag_log_prob(self, y):
        """
        Log-probability of a tagging.
        y -- tagging.
        """
        n = self.n
        y = self.add_tags(y)

        prob = 0
        for i in range(n - 1, len(y)):
            tag = y[i]
            prev_tags = tuple(y[i - n + 1:i])
            prob += self.log_trans_prob(tag, prev_tags)
        return prob

    def log_prob(self, word, y):
        """
        Joint log-probability of a sentence and its tagging.
        word -- sentence.
        y -- tagging.
        """
        tag_prob = [self.tag_log_prob(y)]
        prob = np.asarray([self.log_out_prob(word, tag) for word, tag in zip(word, y)] + tag_prob)
        return prob.sum()

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.
        sent -- the sentence.
        """
        return ViterbiTagger(self).tag(sent)


class ViterbiTagger:

    def __init__(self, hmm):
        """
        hmm -- the HMM.
        """
        self.hmm = hmm

    def _init_table(self):
        n = self.hmm.n
        start_tag = tuple(['<s>']*(n - 1))

        self._pi = defaultdict(lambda: defaultdict(tuple))
        self._pi[0][start_tag] = (log2(1.0), [])

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.
        sent -- the sentence.
        """
        hmm = self.hmm
        n = hmm.n
        inf = float('inf')
        tagset = hmm.tagset()
        self._init_table()
        pi = self._pi

        for k, word in enumerate(sent):
            for t in tagset:
                out = hmm.log_out_prob(word, t)

                # Ignore if the prob is 0

                if out == -inf:
                    continue

                for prev_tags in pi[k]:
                    trans = hmm.log_trans_prob(t, prev_tags)

                    if trans == -inf:
                        continue

                    max_prob_k, max_path_k = pi[k][prev_tags]

                    prob = out + trans + max_prob_k

                    tags = (prev_tags + (t,))[1:]

                    if tags in pi[k + 1]:
                        current_max_prob, _ = pi[k + 1][tags]
                    else:
                        current_max_prob = -inf
                    if prob > current_max_prob:
                        pi[k + 1][tags] = (prob, max_path_k + [t])

        k = len(sent)
        prob, path = max(pi[k].values())

        return path


class MLHMM(HMM):
    def __init__(self, n, tagged_sents, addone=True):
        """
        n -- order of the model.
        tagged_sents -- training sentences, each one being a list of pairs.
        addone -- whether to use addone smoothing (default: True).
        """
        # tagset -- set of tags.
        # trans -- transition probabilities dictionary.
        # out -- output probabilities dictionary.
        self.n = n
        self._addone = addone

        wordtags = [wt for sent in tagged_sents for wt in sent]

        self._tagset, self._wordset = set(), set()
        for word, tag in wordtags:
            self._tagset.add(tag)
            self._wordset.add(word)

        self._wordtag_count = dict(Counter(wordtags))

        tags = [t for s in tagged_sents for _, t in s]
        self._tag_count = dict(Counter(tags))

        self._tag_gram_count = defaultdict(float)

        # Count ngram and n-1gram
        for sent in tagged_sents:
            tags = [tag for _, tag in sent]
            tags = self.add_tags(tags)
            for i in range(len(sent) + 1):
                ngram = tuple(tags[i: i + n])
                self._tag_gram_count[ngram] += 1.0
                self._tag_gram_count[ngram[:-1]] += 1.0

        self._tag_gram_count = dict(self._tag_gram_count)
        self.tagger = ViterbiTagger(self)

    def tcount(self, tokens):
        """Count for an n-gram or (n-1)-gram of tags.
        tokens -- the n-gram or (n-1)-gram tuple of tags.
        """
        try:
            return self._tag_gram_count[tokens]
        except KeyError:
            return 0.0

    def unknown(self, word):
        """Check if a word is unknown for the model.
        word -- the word.
        """
        return not(word in self._wordset)

    def trans_prob(self, tag, prev_tags=tuple()):
        """Probability of a tag.
        tag -- the tag.
        prev_tags -- iterable with the previous n-1 tags
        """
        tcount = self.tcount
        V = len(self._tagset)

        prev_tags = tuple(prev_tags)
        tags = prev_tags + (tag,)

        if self._addone:
            return (tcount(tags) + 1.0) / (tcount(prev_tags) + V)
        return tcount(tags) / tcount(prev_tags)

    def out_prob(self, word, tag):
        """Probability of a word given a tag.
        word -- the word.
        tag -- the tag.
        """
        tag_count = self._tag_count[tag]

        if self.unknown(word) or tag_count == 0:
            return 1.0 / len(self._wordset)
        elif (word, tag) in self._wordtag_count:
            return self._wordtag_count[(word, tag)] / tag_count
        else:
            return 0.0

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.
        sent -- the sentence.
        """
        return self.tagger.tag(sent)
