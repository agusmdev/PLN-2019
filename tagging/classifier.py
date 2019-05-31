from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from tagging.fasttext import FasttextDictVectorizer

classifiers = {
    'lr': LogisticRegression,
    'svm': LinearSVC,
    'mnb': MultinomialNB,
}


def make_feature_dict(base_feats, nf, sent, i):
    feat_dict = {}
    for n in range(0, nf + 1):
        for feature, fun in base_feats.items():
            prev = "p"*n
            nxt = "n"*n
            if len(sent) <= n + i:
                continue
            feat_dict[prev + feature] = fun(sent[i-n])
            feat_dict[nxt + feature] = fun(sent[i+n])

    return feat_dict


def feature_dict(sent, i, n=3):
    # n must be odd
    if n % 2 != 1:
        n -= 1

    if "<s>" not in sent:
        sent = ["<s>"] + list(sent) + ["</s>"]
        i += 1

    n_feats = int(n/2)

    base_feats = {
            "w": str.lower,
            "wu": str.isupper,
            "wt": str.istitle,
            "wd": str.isdigit,
        }

    return make_feature_dict(base_feats, n_feats, sent, i)


class ClassifierTagger:
    """Simple and fast classifier based tagger.
    """

    def __init__(self, tagged_sents, clf='lr', n_features=5):
        """
        clf -- classifying model, one of 'svm', 'lr' (default: 'lr').
        """
        self.n_features = n_features
        self.pipeline = Pipeline(
                steps=[
                    ('vect', DictVectorizer(sparse=True)),
                    ('clf', classifiers[clf]())
                 ])

        self.fit(tagged_sents)

    def fit(self, tagged_sents):
        """
        Train.

        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        self._process_sents(tagged_sents)
        self.pipeline.fit(self.X, self.y)

    def _process_sents(self, tagged_sents):
        X, y = [], []
        vocabulary = set()
        for tagged_sent in tagged_sents:
            if not tagged_sents:
                continue
            sent_words = list(dict(tagged_sent).keys())
            sent_tags = list(dict(tagged_sent).values())
            y.extend(sent_tags)
            vocabulary.update(sent_words)
            for k in range(len(sent_words)):
                X.append(feature_dict(sent_words, k, self.n_features))

        self.X, self.y, self.vocabulary = X, y, vocabulary

    def tag_sents(self, sents):
        """Tag sentences.

        sent -- the sentences.
        """
        return [self.tag(sent) for sent in sents]

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.pipeline.predict(
                    [feature_dict(sent, k, self.n_features)
                        for k in range(len(sent))])][0]

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return not(w in self.vocabulary)


class FastTextClassifier(ClassifierTagger):
    def __init__(self, tagged_sents, clf='lr', n_features=5):
        self.n_features = n_features
        self.clf = clf
        self.pipeline = Pipeline([
            ('feat_u', FeatureUnion([
                ('fast_text', FasttextDictVectorizer('cc.es.300.bin', ['w', "wu", "wt", "wd"])),
                ('vect', DictVectorizer(sparse=True)),
            ])),
            ('clf', classifiers[clf]())
        ])
        self.fit(tagged_sents)
