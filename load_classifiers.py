import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
import regex as re
from from_file import get_posts_from_file
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, RandomForestClassifier

useless_words = stopwords.words('english')

list_posts, dict_vocabulary = get_posts_from_file()


def get_model(text, dict_vocabulary):
    new_dict = dict()
    for word in dict_vocabulary:
        new_dict[word] = 0
    for word in text.split(" "):
        if word in dict_vocabulary:
            new_dict[word] += 1
    return new_dict


def pre_process_input(text):
    remove_url = re.sub(r'http\S+', ' ', text)
    remove_special_characters = re.sub(r'[^a-zA-Z]', ' ', remove_url)
    repeat_pattern = re.compile(r'(\w)\1*')
    remove_multiple_letters = repeat_pattern.sub(r'\1', remove_special_characters)
    remove_short_words = re.sub(r'\b\w{1,2}\b', ' ', remove_multiple_letters)
    remove_long_words = re.sub(r'\b\w{20,}\b', ' ', remove_short_words)
    remove_spaces = re.sub(r' +', ' ', remove_long_words).lower()
    remove_spaces = remove_spaces.strip()
    return remove_spaces


def obtain_data(features):
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(features)
    return X_train


def modify_data(text):
    features = get_model(pre_process_input(text), dict_vocabulary)
    return obtain_data(features)


def grid_search_for_model(model_name, data):
    personality = ''
    if model_name=="Logistic Regression":
        classifier_f = open("logistic_regression0.pickle", "rb")
        gs_rf = pickle.load(classifier_f)
        personality += gs_rf.predict(data)[0][0]
        classifier_f.close()
        classifier_f = open("logistic_regression1.pickle", "rb")
        gs_rf = pickle.load(classifier_f)
        if gs_rf.predict(data)[0]=='Intuition':
            personality += 'N'
        else:
            personality += gs_rf.predict(data)[0][0]
        classifier_f.close()
        classifier_f = open("logistic_regression2.pickle", "rb")
        gs_rf = pickle.load(classifier_f)
        personality += gs_rf.predict(data)[0][0]
        classifier_f.close()
        classifier_f = open("logistic_regression3.pickle", "rb")
        gs_rf = pickle.load(classifier_f)
        personality += gs_rf.predict(data)[0][0]
        classifier_f.close()
        return personality
    elif model_name=="Random Forest":
        classifier_f = open("random_forest0.pickle", "rb")
        gs_rf = pickle.load(classifier_f)
        personality += gs_rf.predict(data)[0][0]
        classifier_f.close()
        classifier_f = open("random_forest1.pickle", "rb")
        gs_rf = pickle.load(classifier_f)
        if gs_rf.predict(data)[0]=='Intuition':
            personality += 'N'
        else:
            personality += gs_rf.predict(data)[0][0]
        classifier_f.close()
        classifier_f = open("random_forest2.pickle", "rb")
        gs_rf = pickle.load(classifier_f)
        personality += gs_rf.predict(data)[0][0]
        classifier_f.close()
        classifier_f = open("random_forest3.pickle", "rb")
        gs_rf = pickle.load(classifier_f)
        personality += gs_rf.predict(data)[0][0]
        classifier_f.close()
        return personality
    elif model_name=="SVM":
        classifier_f = open("svm0.pickle", "rb")
        gs_svm = pickle.load(classifier_f)
        personality += gs_svm.predict(data)[0][0]
        classifier_f.close()
        classifier_f = open("svm1.pickle", "rb")
        gs_svm = pickle.load(classifier_f)
        if gs_svm.predict(data)[0]=='Intuition':
            personality += 'N'
        else:
            personality += gs_svm.predict(data)[0][0]
        classifier_f.close()
        classifier_f = open("svm2.pickle", "rb")
        gs_svm = pickle.load(classifier_f)
        personality += gs_svm.predict(data)[0][0]
        classifier_f.close()
        classifier_f = open("svm3.pickle", "rb")
        gs_svm = pickle.load(classifier_f)
        personality += gs_svm.predict(data)[0][0]
        classifier_f.close()
        return personality
    elif model_name=="Decision Tree":
        classifier_f = open("decision_tree0.pickle", "rb")
        gs_dt = pickle.load(classifier_f)
        personality += gs_dt.predict(data)[0][0]
        classifier_f.close()
        classifier_f = open("decision_tree1.pickle", "rb")
        gs_dt = pickle.load(classifier_f)
        if gs_dt.predict(data)[0]=='Intuition':
            personality += 'N'
        else:
            personality += gs_dt.predict(data)[0][0]
        classifier_f.close()
        classifier_f = open("decision_tree2.pickle", "rb")
        gs_dt = pickle.load(classifier_f)
        personality += gs_dt.predict(data)[0][0]
        classifier_f.close()
        classifier_f = open("decision_tree3.pickle", "rb")
        gs_dt = pickle.load(classifier_f)
        personality += gs_dt.predict(data)[0][0]
        classifier_f.close()
        return personality
    elif model_name=="KNN":
        classifier_f = open("knn0.pickle", "rb")
        gs_dt = pickle.load(classifier_f)
        personality += gs_dt.predict(data)[0][0]
        classifier_f.close()
        classifier_f = open("knn1.pickle", "rb")
        gs_dt = pickle.load(classifier_f)
        if gs_dt.predict(data)[0]=='Intuition':
            personality += 'N'
        else:
            personality += gs_dt.predict(data)[0][0]
        classifier_f.close()
        classifier_f = open("knn2.pickle", "rb")
        gs_dt = pickle.load(classifier_f)
        personality += gs_dt.predict(data)[0][0]
        classifier_f.close()
        classifier_f = open("knn3.pickle", "rb")
        gs_dt = pickle.load(classifier_f)
        personality += gs_dt.predict(data)[0][0]
        classifier_f.close()
        return personality



text = "story working felt upon sense hold older physical huge friendship go house wek"
print(grid_search_for_model("Logistic Regression", modify_data(text)))
print(grid_search_for_model("Random Forest", modify_data(text)))
print(grid_search_for_model("SVM", modify_data(text)))
print(grid_search_for_model("Decision Tree", modify_data(text)))