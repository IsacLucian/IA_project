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

traits = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP",
          "ISFP", "ESTP", "ESFP"]

list_posts, dict_vocabulary = get_posts_from_file()
list_traits = get_traits_from_file()
posts_train, posts_test, traits_train, traits_test = train_test_split(list_posts, list_traits, test_size=0.2,
                                                                         random_state=0)

keys = list(dict_vocabulary)
list_of_dictionaries = []

for dictionary in list_posts:
    index = 0
    new_dict = dict()
    for word in dictionary:
        if word > 0:
            new_dict[keys[index]] = word
        index = index + 1
    list_of_dictionaries.append(new_dict)


features = []
for j in range(len(traits_train)):
    if traits_train[j][0] == '0':
        features.append((list_of_dictionaries[j], 'introvert'))
    elif traits_train[j][0] == '1':
        features.append((list_of_dictionaries[j], 'extrovert'))

# print(features)

IntroExtro = NaiveBayesClassifier.train(features)

print(nltk.classify.util.accuracy(IntroExtro, features)*100)





# useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
# def build_bag_of_words_features_filtered(words):
#     words = nltk.word_tokenize(words)
#     return {
#         word:1 for word in words \
#         if not word in useless_words}
#
#
#
# data_set = pd.read_csv("mbti_1.csv")
# all_posts= pd.DataFrame()
# for j in traits:
#     temp1 = data_set[data_set['type']==j]['posts']
#     temp2 = []
#     for i in temp1:
#         temp2+=i.split('|||')
#     temp3 = pd.Series(temp2)
#     all_posts[j] = temp3
#
# # Features for the bag of words model
# features=[]
# for j in traits:
#     temp1 = all_posts[j]
#     temp1 = temp1.dropna() #not all the personality types have same number of files
#     if('I' in j):
#         features += [[(build_bag_of_words_features_filtered(i), 'introvert') \
#         for i in temp1]]
#     if('E' in j):
#         features += [[(build_bag_of_words_features_filtered(i), 'extrovert') \
#         for i in temp1]]
#
# split=[]
# for i in range(16):
#     split += [len(features[i]) * 0.8]
# split = np.array(split,dtype = int)
#
# train=[]
# for i in range(16):
#     train += features[i][:split[i]]
#
# print(train)

# IntroExtro = NaiveBayesClassifier.train(train)

# print(nltk.classify.util.accuracy(IntroExtro, train)*100)