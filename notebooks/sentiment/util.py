from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn import metrics
import pickle
import re


def load_datasets():
    dataset = load_files('review_polarity/txt_sentoken', shuffle=False)
    docs_traindev, docs_test, y_traindev, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=42)
    test = (docs_test, y_test)
    docs_train, docs_dev, y_train, y_dev = train_test_split(
        docs_traindev, y_traindev, test_size=0.2, random_state=42)
    train = docs_train, y_train
    dev = docs_dev, y_dev
    return train, dev, test


def print_eval(model, X, y_true):
    y_pred = model.predict(X)
    acc = metrics.accuracy_score(y_true, y_pred)
    print('accuracy\t{:2.2f}\n'.format(acc))
    # print(metrics.classification_report(y_true, y_pred, target_names=['neg', 'pos']))
    print(metrics.classification_report(y_true, y_pred, target_names=["n", "p", "none", "neu"]))
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)


def eval(model, X, y_true):
    y_pred = model.predict(X)
    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    return {'acc': acc, 'f1': f1}


def print_short_eval(model, X, y_true):
    res = eval(model, X, y_true)
    print('accuracy\t{acc:2.2f}\tmacro f1\t{f1:2.2f}'.format(**res))


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def preprocess_tweets(str_):
    mentions = r'(?:@[^\s]+)'
    urls = r'(?:https?\://t.co/[\w]+)'
    str_ = re.sub(mentions, '', str_)
    return re.sub(urls, '', str_)


vect = TfidfVectorizer(tokenizer=word_tokenize,
                       binary=True,
                       analyzer="char_wb",
                       ngram_range=(1, 6),
                       min_df=5,
                       max_df=0.95,
                       preprocessor=preprocess_tweets,
                       stop_words=stopwords.words("spanish"))

vect2 = TfidfVectorizer(tokenizer=word_tokenize,
                        binary=True,
                        analyzer="word",
                        ngram_range=(1, 5),
                        preprocessor=preprocess_tweets,
                        stop_words=stopwords.words("spanish"))


def build_pipeline():
    pipeline = Pipeline([
            ('feats', FeatureUnion([
                    ('vect', vect),
                    ('vect2', vect2),
                ])),
            ('clf', LogisticRegression(class_weight="balanced")),
        ])

    return pipeline
