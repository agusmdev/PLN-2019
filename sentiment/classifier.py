from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from nltk import word_tokenize
from nltk.corpus import stopwords
import re

classifiers = {
    'maxent': LogisticRegression,
    'mnb': MultinomialNB,
    'svm': LinearSVC,
}


def preprocess_tweets(str_):
    mentions = r'(?:@[^\s]+)'
    urls = r'(?:https?\://t.co/[\w]+)'
    str_ = re.sub(mentions, '', str_)
    return re.sub(urls, '', str_)


vect = TfidfVectorizer(tokenizer=word_tokenize,
                       binary=True,
                       analyzer="char_wb",
                       ngram_range=(1, 6),
                       min_df=3,
                       max_df=0.7,
                       preprocessor=preprocess_tweets,
                       stop_words=stopwords.words("spanish"))

vect2 = TfidfVectorizer(tokenizer=word_tokenize,
                        binary=True,
                        analyzer="word",
                        ngram_range=(1, 5),
                        preprocessor=preprocess_tweets,
                        stop_words=stopwords.words("spanish"))


class SentimentClassifier(object):

    def __init__(self, clf='svm'):
        """
        clf -- classifying model, one of 'svm', 'maxent', 'mnb' (default: 'svm').
        """
        self._clf = clf
        self._pipeline = pipeline = Pipeline([
            ('feats', FeatureUnion([
                    ('vect', vect),  # can pass in either a pipeline
                    ('vect2', vect2),  # or a transformer
                ])),
            ('clf', classifiers[clf]()),
        ])

    def fit(self, X, y):
        self._pipeline.fit(X, y)

    def predict(self, X):
        return self._pipeline.predict(X)
