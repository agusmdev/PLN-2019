import pickle
import difflib
from operator import itemgetter
from sentiment.tass import InterTASSReader
from copy import deepcopy


class InterTASSAugmented:

    def __init__(self, file="augmented_data.pkl", clean=True, ratio=0.9):
        self.ratio = ratio
        with open(file, "rb") as f:
            if clean:
                self.augmented_data = self.clean(pickle.load(f))
            else:
                self.augmented_data = pickle.load(f)

    def clean(self, tweets):
        corpus = "intertass-ES-train-tagged.xml"
        reader = InterTASSReader(corpus)
        original_tweets = list(reader.tweets())

        original_tweets = sorted(original_tweets, key=itemgetter('tweetid'))
        tweets = sorted(tweets, key=itemgetter('tweetid'))

        cleaned_data = []

        count = 0
        tr_tw = tweets[0]
        for or_tw in original_tweets[:-1]:
            tweets_set = set()
            while or_tw["tweetid"] == tr_tw["tweetid"]:
                if self._distance(or_tw["content"], tr_tw["content"]) > self.ratio:
                    tweets_set.add(tr_tw["content"])
                count += 1
                try:
                    tr_tw = tweets[count]
                except IndexError:
                    continue
            cleaned_data.extend(self._save_tweets(or_tw, tweets_set))

        return cleaned_data

    def _save_tweets(self, or_tw, tweets_set):
        res = []

        for content in tweets_set:
            copy = deepcopy(or_tw)
            copy["content"] = content
            res.append(copy)

        return res

    def _distance(self, original, translated):
        seq = difflib.SequenceMatcher(None, original, translated)
        return seq.ratio()

    def Xy(self):
        X, y = [], []
        for tweet_el in self.augmented_data:
            content = tweet_el["content"]
            if content not in X:
                X.append(content)
                y.append(tweet_el["sentiment"])
        return X, y
