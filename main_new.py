import string

import pandas as pd
import numpy as np
import regex as re
import nltk
from nltk import NaiveBayesClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from from_file import get_posts_from_file
from from_file import get_traits_from_file

# nltk.download('punkt')


def get_type(position):
    if position == 0:
        return 'Introversion', 'Extroversion'
    elif position == 1:
        return 'Intuition', 'Sensing'
    elif position == 2:
        return 'Thinking', 'Feeling'
    else:
        return 'Judging', 'Perceiving'


def get_model(posts, traits, position, keys):
    type1, type2 = get_type(position)
    list_of_dictionaries = []

    for dictionary in posts:
        index = 0
        new_dict = dict()
        for word in dictionary:
            if word > 0:
                new_dict[keys[index]] = word
            index = index + 1
        list_of_dictionaries.append(new_dict)

    features = []
    for j in range(len(traits)):
        if traits[j][position] == '0':
            features.append((list_of_dictionaries[j], type1))
        elif traits[j][position] == '1':
            features.append((list_of_dictionaries[j], type2))
    return features



traits = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP",
          "ISFP", "ESTP", "ESFP"]

list_posts, dict_vocabulary = get_posts_from_file()
list_traits = get_traits_from_file()
posts_train, posts_test, traits_train, traits_test = train_test_split(list_posts, list_traits, test_size=0.2, random_state=0)

keys = list(dict_vocabulary)


# =========== Introversion - Extroversion model ==========
features = get_model(posts_train, traits_train, 0, keys)
introversion_extroversion_model = NaiveBayesClassifier.train(features)
print(nltk.classify.util.accuracy(introversion_extroversion_model, features)*100)

features_test = get_model(posts_test, traits_test, 0, keys)
print(nltk.classify.util.accuracy(introversion_extroversion_model, features_test)*100)


# =========== Intuition - Sensing model ==========
features = get_model(posts_train, traits_train, 1, keys)
intuition_sensing_model = NaiveBayesClassifier.train(features)
print(nltk.classify.util.accuracy(intuition_sensing_model, features)*100)

features_test = get_model(posts_test, traits_test, 1, keys)
print(nltk.classify.util.accuracy(intuition_sensing_model, features_test)*100)


# =========== Thinking - Feeling model ==========
features = get_model(posts_train, traits_train, 2, keys)
thinking_feeling_model = NaiveBayesClassifier.train(features)
print(nltk.classify.util.accuracy(thinking_feeling_model, features)*100)

features_test = get_model(posts_test, traits_test, 2, keys)
print(nltk.classify.util.accuracy(thinking_feeling_model, features_test)*100)


# =========== Judging - Perceiving model ==========
features = get_model(posts_train, traits_train, 3, keys)
judging_perceiving_model = NaiveBayesClassifier.train(features)
print(nltk.classify.util.accuracy(judging_perceiving_model, features)*100)

features_test = get_model(posts_test, traits_test, 3, keys)
print(nltk.classify.util.accuracy(judging_perceiving_model, features_test)*100)