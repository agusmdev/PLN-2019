from sentiment.tass import InterTASSReader
import os
from collections import Counter, defaultdict as dc
from pprint import pprint

categories = ["CR", "ES", "PE"]
root_path = "InterTASS"
suffix = "train-tagged.xml"


def count_tweets(path):
    reader = InterTASSReader(path)
    dist = Counter(reader.y())
    return dict(dist), sum(dist.values())


def corpus_statistics(root_path, categories):
    statistics = dc(dict)
    for category in categories:
        path = os.path.join(root_path, category)
        for corpus in os.listdir(path):
            if corpus.endswith(suffix):
                statistics[category], statistics[category]["total"] = \
                    count_tweets(os.path.join(path, corpus))
    return dict(statistics)


if __name__ == '__main__':
    pprint(corpus_statistics(root_path, categories))
