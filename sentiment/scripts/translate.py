from googletrans import Translator
from sentiment.tass import InterTASSReader
import os
from collections import Counter, defaultdict as dc
from copy import deepcopy
from googletrans.constants import LANGUAGES
import pickle
from google.cloud import translate

reader = InterTASSReader('train.xml')
tweets = list(reader.tweets())  # iterador sobre los tweets

lang = ["en", "fr"]
# lang = ["en", "fr", "ar", "pt"]


def augment_data(tweets, langs, client):
    data = deepcopy(tweets)

    for i, tweet in enumerate(tweets):
        data.extend(get_translations(tweet, langs, client, i))
    return data


def get_translations(tweet, langs, client, i):
    res = list()
    for lang in langs:
        try:
            to_translate = deepcopy(tweet)
            new = client.translate(to_translate["content"], target_language=lang)["translatedText"]
            to_translate["content"] = client.translate(new, target_language="es")["translatedText"]
            res.append(to_translate)
            print("tweet N°:{}".format(i))
            if i % 5:
                print(to_translate)
        except:
            print("Error")
    return res


client = translate.Client()
augmented_train = augment_data(tweets, lang, client)


with open("augmented_data", "wb") as f:
    pickle.dump(augmented_train, f)
