import pandas as pd
import regex as re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Uncomment for first time run
# nltk.download('stopwords')

csv_path = "mbti_1.csv"
useless_words = stopwords.words('english')
traits = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP",
          "ISFP", "ESTP", "ESFP"]

translate_traits = {
    "I": 0,
    "E": 1,
    "N": 0,
    "S": 1,
    "F": 0,
    "T": 1,
    "J": 0,
    "P": 1
}


def pre_process_csv(path):
    data = pd.read_csv(path)
    list_posts = []
    list_traits = []
    for ind in range(0, data.shape[0]):
        remove_url = re.sub(r'http\S+', ' ', data['posts'][ind])
        remove_special_characters = re.sub(r'[^a-zA-Z]', ' ', remove_url)
        repeat_pattern = re.compile(r'(\w)\1*')
        remove_multiple_letters = repeat_pattern.sub(r'\1', remove_special_characters)
        remove_short_words = re.sub(r'\b\w{1,2}\b', ' ', remove_multiple_letters)
        remove_long_words = re.sub(r'\b\w{20,}\b', ' ', remove_short_words)
        remove_spaces = re.sub(r' +', ' ', remove_long_words).lower()
        remove_spaces = remove_spaces.strip()
        remove_traits = " ".join([word for word in remove_spaces.split() if word.upper() not in traits])

        # Remove useless words and reduce them
        reduce_words = nltk.stem.WordNetLemmatizer()
        reduced = " ".join([reduce_words.lemmatize(word) for word in remove_traits.split()
                            if word not in useless_words]).strip()

        if len(reduced) > 0:
            list_posts.append(reduced + " ")
            # Translate trait
            list_traits.append([translate_traits[trait] for trait in data['type'][ind]])

    vect = CountVectorizer(max_features=1000, max_df=0.7, min_df=0.1)
    vect.fit(list_posts)
    list_posts = vect.transform(list_posts).toarray()
    posts_train, posts_test, traits_train, traits_test = train_test_split(list_posts, list_traits, test_size=0.2,
                                                                          random_state=0)
    return posts_train, posts_test, traits_train, traits_test


def main():
    posts_train, posts_test, traits_train, traits_test = pre_process_csv(csv_path)
    print(posts_train.shape, posts_test.shape, len(traits_train), len(traits_test))


if __name__ == '__main__':
    main()
