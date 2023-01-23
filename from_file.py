from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def get_posts_from_file():
    f = open("list_posts.txt", "r")
    list_posts = []
    list_posts_original = []
    lines = f.readlines()
    for line in lines:
        list_posts.append(line[0:-2])
        list_posts_original.append(line[0:-2])
    f.close()
    vect = CountVectorizer(max_features=1000, max_df=0.7, min_df=0.1)
    vect.fit(list_posts)
    dict_vocabulary = vect.vocabulary_
    list_posts = vect.transform(list_posts).toarray()
    return list_posts, dict_vocabulary


def get_traits_from_file():
    f = open("list_traits.txt", "r")
    list_traits = []
    lines = f.readlines()
    for line in lines:
        list_traits.append(line.split(" ")[0:-1])
    f.close()
    return list_traits