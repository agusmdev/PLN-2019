from collections import defaultdict


class BadBaselineTagger:

    def __init__(self, tagged_sents):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        """
        pass

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        """Tag a word.

        w -- the word.
        """
        return 'nc0s000'

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return True


class BaselineTagger:

    def __init__(self, tagged_sents, default_tag='nc0s000'):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        default_tag -- tag for unknown words.
        """
        self.default_tag = default_tag
        self.tags_per_word = defaultdict(lambda: defaultdict(int))

        for sent in tagged_sents:
            for word, tag in sent:
                self.tags_per_word[word][tag] += 1

        self.most_frequent_tag = dict()

        for word, tag_freqs in self.tags_per_word.items():

            reverse = {count: tag for tag, count in tag_freqs.items()}
            self.most_frequent_tag[word] = reverse[max(reverse.keys())]

    def tag(self, sent):
        """Tag a sentence.
        sent -- the sentence.
        """
        return [self.tag_word(word) for word in sent]

    def tag_word(self, word):
        """Tag a word.
        word -- the word.
        """
        try:
            return self.most_frequent_tag[word]
        except KeyError:
            return self.default_tag

    def unknown(self, word):
        """Check if a word is unknown for the model.
        word -- the word.
        """
        return not(word in self.most_frequent_tag)
