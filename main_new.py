from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from from_file import get_posts_from_file
from from_file import get_traits_from_file
from sklearn.feature_extraction import DictVectorizer
import numpy as np

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


def obtain_data(features, features_test):
    vectorizer = DictVectorizer()
    X_train, y_train = list(zip(*features))
    X_train = vectorizer.fit_transform(X_train)
    X_test, y_test = list(zip(*features_test))
    X_test = vectorizer.transform(X_test)
    return X_train, y_train, X_test, y_test


traits = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP",
          "ISFP", "ESTP", "ESFP"]

list_posts, dict_vocabulary = get_posts_from_file()
list_traits = get_traits_from_file()
posts_train, posts_test, traits_train, traits_test = train_test_split(list_posts, list_traits, test_size=0.2,
                                                                      random_state=0)

keys = list(dict_vocabulary)


def grid_search_for_model(grid_parameters,model_name,position):
    if model_name=="Logistic Regression":
        features = get_model(posts_train, traits_train, position, keys)
        features_test = get_model(posts_test, traits_test, position, keys)
        X_train, y_train, X_test, y_test = obtain_data(features, features_test)
        gs_log_reg = GridSearchCV(estimator=LogisticRegression(),
                                param_grid=grid_parameters,
                                cv=5,
                                verbose=True)
        gs_log_reg.fit(X_train, y_train)
        print(gs_log_reg.best_params_)
        if position==0:
            print("Introversion Extroversion Logistic Regression Model Train accuracy percent:",
                  gs_log_reg.score(X_train, y_train) * 100)
            print("Introversion Extroversion Logistic Regression Model Test accuracy percent:",
                  gs_log_reg.score(X_test, y_test) * 100)
        elif position==1:
            print("Intuition Sensing Logistic Regression Model Train accuracy percent:",
                  gs_log_reg.score(X_train, y_train) * 100)
            print("Intuition Sensing Logistic Regression Model Test accuracy percent:",
                  gs_log_reg.score(X_test, y_test) * 100)
        if position==2:
            print("Thinking Feeling Logistic Regression Model Train accuracy percent:",
                  gs_log_reg.score(X_train, y_train) * 100)
            print("Thinking Feeling Logistic Regression Model Test accuracy percent:",
                  gs_log_reg.score(X_test, y_test) * 100)
        elif position==3:
            print("Judging Perceiving Logistic Regression Model Train accuracy percent:",
                  gs_log_reg.score(X_train, y_train) * 100)
            print("Judging Perceiving Logistic Regression Model Test accuracy percent:",
                  gs_log_reg.score(X_test, y_test) * 100)
    elif model_name=="Random Forest":
        features = get_model(posts_train, traits_train, position, keys)
        features_test = get_model(posts_test, traits_test, position, keys)
        X_train, y_train, X_test, y_test = obtain_data(features, features_test)
        gs_rf = GridSearchCV(estimator=RandomForestClassifier(),
                                param_grid=grid_parameters,
                                cv=3,
                                verbose=True)
        gs_rf.fit(X_train, y_train)
        print(gs_rf.best_params_)
        if position==0:
            print("Introversion Extroversion Random Forest Model Train accuracy percent:",
                  gs_rf.score(X_train, y_train) * 100)
            print("Introversion Extroversion Random Forest Model Test accuracy percent:",
                  gs_rf.score(X_test, y_test) * 100)
        elif position==1:
            print("Intuition Sensing Random Forest Model Train accuracy percent:",
                  gs_rf.score(X_train, y_train) * 100)
            print("Intuition Sensing Random Forest Model Test accuracy percent:",
                  gs_rf.score(X_test, y_test) * 100)
        if position==2:
            print("Thinking Feeling Random Forest Model Train accuracy percent:",
                  gs_rf.score(X_train, y_train) * 100)
            print("Thinking Feeling Random Forest Model Test accuracy percent:",
                  gs_rf.score(X_test, y_test) * 100)
        elif position==3:
            print("Judging Perceiving Random Forest Model Train accuracy percent:",
                  gs_rf.score(X_train, y_train) * 100)
            print("Judging Perceiving Random Forest Model Test accuracy percent:",
                  gs_rf.score(X_test, y_test) * 100)
    elif model_name=="SVM":
        features = get_model(posts_train, traits_train, position, keys)
        features_test = get_model(posts_test, traits_test, position, keys)
        X_train, y_train, X_test, y_test = obtain_data(features, features_test)
        gs_svm = GridSearchCV(estimator=SVC(),
                             param_grid=grid_parameters,
                             refit=True,
                             verbose=2)
        gs_svm.fit(X_train, y_train)
        print(gs_svm.best_params_)
        if position == 0:
            print("Introversion Extroversion SVM Model Train accuracy percent:",
                  gs_svm.score(X_train, y_train) * 100)
            print("Introversion Extroversion SVM Model Test accuracy percent:",
                  gs_svm.score(X_test, y_test) * 100)
        elif position == 1:
            print("Intuition Sensing SVM Model Train accuracy percent:",
                  gs_svm.score(X_train, y_train) * 100)
            print("Intuition Sensing SVM Model Test accuracy percent:",
                  gs_svm.score(X_test, y_test) * 100)
        if position == 2:
            print("Thinking Feeling SVM Model Train accuracy percent:",
                  gs_svm.score(X_train, y_train) * 100)
            print("Thinking Feeling SVM Model Test accuracy percent:",
                  gs_svm.score(X_test, y_test) * 100)
        elif position == 3:
            print("Judging Perceiving SVM Model Train accuracy percent:",
                  gs_svm.score(X_train, y_train) * 100)
            print("Judging Perceiving SVM Model Test accuracy percent:",
                  gs_svm.score(X_test, y_test) * 100)
    elif model_name=="Decision Tree":
        features = get_model(posts_train, traits_train, position, keys)
        features_test = get_model(posts_test, traits_test, position, keys)
        X_train, y_train, X_test, y_test = obtain_data(features, features_test)
        gs_dt = GridSearchCV(estimator=DecisionTreeClassifier(),
                             param_grid=grid_parameters,
                              cv=10,
                              n_jobs=1,
                              verbose=2)
        gs_dt.fit(X_train, y_train)
        print(gs_dt.best_params_)
        if position == 0:
            print("Introversion Extroversion Decision Tree Model Train accuracy percent:",
                  gs_dt.score(X_train, y_train) * 100)
            print("Introversion Extroversion Decision Tree Model Test accuracy percent:",
                  gs_dt.score(X_test, y_test) * 100)
        elif position == 1:
            print("Intuition Sensing Decision Tree Model Train accuracy percent:",
                  gs_dt.score(X_train, y_train) * 100)
            print("Intuition Sensing Decision Tree Model Test accuracy percent:",
                  gs_dt.score(X_test, y_test) * 100)
        if position == 2:
            print("Thinking Feeling Decision Tree Model Train accuracy percent:",
                  gs_dt.score(X_train, y_train) * 100)
            print("Thinking Feeling Decision Tree Model Test accuracy percent:",
                  gs_dt.score(X_test, y_test) * 100)
        elif position == 3:
            print("Judging Perceiving Decision Tree Model Train accuracy percent:",
                  gs_dt.score(X_train, y_train) * 100)
            print("Judging Perceiving Decision Tree Model Test accuracy percent:",
                  gs_dt.score(X_test, y_test) * 100)



# # =========== Naive Bayes Classifier ===========
#
# # =========== Introversion - Extroversion model ==========
# features = get_model(posts_train, traits_train, 0, keys)
# introversion_extroversion_naive_bayes_model = NaiveBayesClassifier.train(features)
# print("Introversion Extroversion Naive Bayes Model Train accuracy percent:",
#       nltk.classify.util.accuracy(introversion_extroversion_naive_bayes_model, features) * 100)
#
# features_test = get_model(posts_test, traits_test, 0, keys)
# print("Introversion Extroversion Naive Bayes Model Test accuracy percent:",
#       nltk.classify.util.accuracy(introversion_extroversion_naive_bayes_model, features_test) * 100)
#
# # =========== Intuition - Sensing model ==========
# features = get_model(posts_train, traits_train, 1, keys)
# intuition_sensing_naive_bayes_model = NaiveBayesClassifier.train(features)
# print("Intuition Sensing Naive Bayes Model Train accuracy percent:",
#       nltk.classify.util.accuracy(intuition_sensing_naive_bayes_model, features) * 100)
#
# features_test = get_model(posts_test, traits_test, 1, keys)
# print("Intuition Sensing Naive Bayes Model Test accuracy percent:",
#       nltk.classify.util.accuracy(intuition_sensing_naive_bayes_model, features_test) * 100)
#
# # =========== Thinking - Feeling model ==========
# features = get_model(posts_train, traits_train, 2, keys)
# thinking_feeling_naive_bayes_model = NaiveBayesClassifier.train(features)
# print("Thinking Feeling Naive Bayes Model Train accuracy percent:",
#       nltk.classify.util.accuracy(thinking_feeling_naive_bayes_model, features) * 100)
#
# features_test = get_model(posts_test, traits_test, 2, keys)
# print("Thinking Feeling Naive Bayes Model Test accuracy percent:",
#       nltk.classify.util.accuracy(thinking_feeling_naive_bayes_model, features_test) * 100)
#
# # =========== Judging - Perceiving model ==========
# features = get_model(posts_train, traits_train, 3, keys)
# judging_perceiving_naive_bayes_model = NaiveBayesClassifier.train(features)
# print("Judging Perceiving Naive Bayes Model Train accuracy percent:",
#       nltk.classify.util.accuracy(judging_perceiving_naive_bayes_model, features) * 100)
#
# features_test = get_model(posts_test, traits_test, 3, keys)
# print("Judging Perceiving Naive Bayes Model Test accuracy percent:",
#       nltk.classify.util.accuracy(judging_perceiving_naive_bayes_model, features_test) * 100)

print("======================================================================")

log_reg_grid = {"C": np.logspace(-4, 4, 30),
                    "solver": ["liblinear"]}
grid_search_for_model(log_reg_grid, "Logistic Regression", 0)
grid_search_for_model(log_reg_grid, "Logistic Regression", 1)
grid_search_for_model(log_reg_grid, "Logistic Regression", 2)
grid_search_for_model(log_reg_grid, "Logistic Regression", 3)
rf_grid = {#"n_estimators": np.arange(100, 1000, 100),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 3),
           #"min_samples_leaf": np.arange(1, 20, 3)
           }
grid_search_for_model(rf_grid, "Random Forest", 0)
grid_search_for_model(rf_grid, "Random Forest", 1)
grid_search_for_model(rf_grid, "Random Forest", 2)
grid_search_for_model(rf_grid, "Random Forest", 3)
svm_grid = {'C': [0.1, 1, 10, 100],
            #'gamma': [1, 0.1, 0.01, 0.001],
            #'kernel': ['rbf', 'poly', 'sigmoid']
            }
grid_search_for_model(svm_grid, "SVM", 0)
grid_search_for_model(svm_grid, "SVM", 1)
grid_search_for_model(svm_grid, "SVM", 2)
grid_search_for_model(svm_grid, "SVM", 3)
dt_grid =  {
    'min_samples_leaf': [1, 2, 3],
    'max_depth': [1, 2, 3],
    'criterion': ['gini', 'entropy']
}
grid_search_for_model(dt_grid, "Decision Tree", 0)
grid_search_for_model(dt_grid, "Decision Tree", 1)
grid_search_for_model(dt_grid, "Decision Tree", 2)
grid_search_for_model(dt_grid, "Decision Tree", 3)