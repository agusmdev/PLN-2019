# from googletrans import Translator
from sentiment.tass import InterTASSReader
import os
from collections import Counter, defaultdict as dc
from copy import deepcopy
from googletrans.constants import LANGUAGES
import pickle
# from google.cloud import translate
from translator import Translator
from tqdm import tqdm


def heuristic_string(tweets, lang):
    last_i = 0
    str_ = ""
    translated_tweets = []
    news = []
    for i, tweet in tqdm(enumerate(tweets)):
        translated_tweets.append(deepcopy(tweet))
        str_ += "\n---\n" + tweet["content"]
        if len(str_) >= 4800 or i == len(tweets) - 1:
            translation = client.translate_to(str_, lang).split("\n---\n")
            str_ = ""
            try:
                translation.remove("")
            except ValueError:
                pass
            for j, tw in enumerate(translated_tweets):
                try:
                    tw["content"] = translation[j]
                except IndexError:
                    pass
            news += deepcopy(translated_tweets)
            translated_tweets = []
    return news + tweets


if __name__ == '__main__':
    reader = InterTASSReader('intertass-ES-train-tagged.xml')
    tweets = list(reader.tweets())  # iterador sobre los tweets

    langs = list(LANGUAGES.keys())[:5]
    try:
        langs.remove("es")
        langs.remove('zh-cn')
        langs.remove('zh-tw')
    except ValueError:
        pass

    client = Translator(headless_browser=True, bulk=True)
    augmented_train = []
    for lang in tqdm(langs):
        try:
            augmented_train += heuristic_string(tweets, lang)
        except:
            pass
    client.quit()

    with open("augmented_data.pkl", "wb") as f:
        pickle.dump(augmented_train, f)