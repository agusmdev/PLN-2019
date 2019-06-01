"""Print corpus statistics.

Usage:
  stats.py
  stats.py -h | --help

Options:
  -h --help     Show this screen.
"""
from docopt import docopt
from collections import defaultdict

from tagging.ancora import SimpleAncoraCorpusReader


class POSStats:
    """Several statistics for a POS tagged corpus.
    """

    def __init__(self, tagged_sents):
        """
        tagged_sents -- corpus (list/iterable/generator of tagged sentences)
        """
        # WORK HERE!!
        # COLLECT REQUIRED STATISTICS INTO DICTIONARIES.
        # Total Sentences
        self.sents_count = len(tagged_sents)
        self.word_voc = set()
        self.tag_voc = set()
        self.words_count = defaultdict(int)
        self.tags_count = defaultdict(int)
        self.words_per_tag = defaultdict(lambda: defaultdict(int))
        self.tags_per_word = defaultdict(lambda: defaultdict(int))

        for sent in tagged_sents:
            for word, tag in sent:
                self.word_voc.add(word)
                self.tag_voc.add(tag)
                self.tags_per_word[word][tag] += 1
                self.words_count[word] += 1
                self.tags_count[tag] += 1
                self.words_per_tag[tag][word] += 1

        self._tokencount = sum(map(len, tagged_sents))

    def sent_count(self):
        """Total number of sentences."""
        # WORK HERE!!
        return self.sents_count

    def token_count(self):
        """Total number of tokens."""
        # WORK HERE!!
        return self._tokencount

    def words(self):
        """Vocabulary (set of word types)."""
        # WORK HERE!!
        return self.word_voc

    def word_count(self):
        """Vocabulary size."""
        # WORK HERE!!
        # return self.words_count
        return len(self.words_count)

    def word_freq(self, w):
        """Frequency of word w."""
        # WORK HERE!!
        return self.words_count[w]

    def unambiguous_words(self):
        """List of words with only one observed POS tag."""
        # WORK HERE!!
        return set([word for word, tag in self.tags_per_word.items()
                    if len(tag) == 1])

    def ambiguous_words(self, n):
        """List of words with n different observed POS tags.
        n -- number of tags.
        """
        # WORK HERE!!
        return set([word for word, tag in self.tags_per_word.items()
                    if len(tag) == n])

    def tags(self):
        """POS Tagset."""
        # WORK HERE!!
        return self.tag_voc

    def tag_count(self):
        """POS tagset size."""
        # WORK HERE!!
        # return self.tag_voc
        return len(self.tag_voc)

    def tag_freq(self, t):
        """Frequency of tag t."""
        # WORK HERE!!
        return self.tags_count[t]

    def tag_word_dict(self, t):
        """Dictionary of words and their counts for tag t."""
        return dict(self.words_per_tag[t])


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    corpus = SimpleAncoraCorpusReader('ancora/ancora-3.0.1es/')
    sents = list(corpus.tagged_sents())

    # compute the statistics
    stats = POSStats(sents)

    print('Basic Statistics')
    print('================')
    print('sents: {}'.format(stats.sent_count()))
    token_count = stats.token_count()
    print('tokens: {}'.format(token_count))
    word_count = stats.word_count()
    print('words: {}'.format(word_count))
    print('tags: {}'.format(stats.tag_count()))
    print('')

    print('Most Frequent POS Tags')
    print('======================')
    tags = [(t, stats.tag_freq(t)) for t in stats.tags()]
    sorted_tags = sorted(tags, key=lambda t_f: -t_f[1])
    print('tag\tfreq\t%\ttop')
    for t, f in sorted_tags[:10]:
        words = stats.tag_word_dict(t).items()
        sorted_words = sorted(words, key=lambda w_f: -w_f[1])
        top = [w for w, _ in sorted_words[:5]]
        print('{0}\t{1}\t{2:2.2f}\t({3})'.format(t, f, f * 100 / token_count, ', '.join(top)))
    print('')

    print('Word Ambiguity Levels')
    print('=====================')
    print('n\twords\t%\ttop')
    for n in range(1, 10):
        words = list(stats.ambiguous_words(n))
        m = len(words)

        # most frequent words:
        sorted_words = sorted(words, key=lambda w: -stats.word_freq(w))
        top = sorted_words[:5]
        print('{0}\t{1}\t{2:2.2f}\t({3})'.format(n, m, m * 100 / word_count, ', '.join(top)))
