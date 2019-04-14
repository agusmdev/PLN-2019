from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
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


def clean_tweets(str_):
    mentions = r'(?:@[^\s]+)'
    urls = r'(?:https?\://t.co/[\w]+)'
    str_ = re.sub(mentions, '', str_)
    return re.sub(urls, '', str_)


class SentimentClassifier(object):

    def __init__(self, clf='svm'):
        """
        clf -- classifying model, one of 'svm', 'maxent', 'mnb' (default: 'svm').
        """
        self._clf = clf
        self._pipeline = pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=word_tokenize,
                                     binary=True,
                                     ngram_range=(1, 3),
                                     preprocessor=clean_tweets,
                                     stop_words=stopwords.words("spanish"))),
            ('clf', classifiers[clf]()),
        ])

    def fit(self, X, y):
        self._pipeline.fit(X, y)

    def predict(self, X):
        return self._pipeline.predict(X)

