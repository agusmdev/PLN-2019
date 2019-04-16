import pickle


class InterTASSAugmented:

    def __init__(self):
        with open("augmented_data", "rb") as f:
            self.augmented_data = pickle.load(f)

    def Xy(self):
        """Iterator over the tweet contents."""
        X = []
        y = []
        for tweet_el in self.augmented_data:
            content = tweet_el["content"]
            if content not in X:
                X.append(content)
                y.append(tweet_el["sentiment"])
        return X, y
