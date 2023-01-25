import nltk
from nltk import NaiveBayesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pickle

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
            # save_classifier = open("logistic_regression0.pickle", "wb")
            # pickle.dump(gs_log_reg, save_classifier)
            # save_classifier.close()
            print("Introversion Extroversion Logistic Regression Model Train accuracy percent:",
                  gs_log_reg.score(X_train, y_train) * 100)
            print("Introversion Extroversion Logistic Regression Model Test accuracy percent:",
                  gs_log_reg.score(X_test, y_test) * 100)
        elif position==1:
            # save_classifier = open("logistic_regression1.pickle", "wb")
            # pickle.dump(gs_log_reg, save_classifier)
            # save_classifier.close()
            print("Intuition Sensing Logistic Regression Model Train accuracy percent:",
                  gs_log_reg.score(X_train, y_train) * 100)
            print("Intuition Sensing Logistic Regression Model Test accuracy percent:",
                  gs_log_reg.score(X_test, y_test) * 100)
        if position==2:
            # save_classifier = open("logistic_regression2.pickle", "wb")
            # pickle.dump(gs_log_reg, save_classifier)
            # save_classifier.close()
            print("Thinking Feeling Logistic Regression Model Train accuracy percent:",
                  gs_log_reg.score(X_train, y_train) * 100)
            print("Thinking Feeling Logistic Regression Model Test accuracy percent:",
                  gs_log_reg.score(X_test, y_test) * 100)
        elif position==3:
            # save_classifier = open("logistic_regression3.pickle", "wb")
            # pickle.dump(gs_log_reg, save_classifier)
            # save_classifier.close()
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
            # save_classifier = open("random_forest0.pickle", "wb")
            # pickle.dump(gs_rf, save_classifier)
            # save_classifier.close()
            print("Introversion Extroversion Random Forest Model Train accuracy percent:",
                  gs_rf.score(X_train, y_train) * 100)
            print("Introversion Extroversion Random Forest Model Test accuracy percent:",
                  gs_rf.score(X_test, y_test) * 100)
        elif position==1:
            # save_classifier = open("random_forest1.pickle", "wb")
            # pickle.dump(gs_rf, save_classifier)
            # save_classifier.close()
            print("Intuition Sensing Random Forest Model Train accuracy percent:",
                  gs_rf.score(X_train, y_train) * 100)
            print("Intuition Sensing Random Forest Model Test accuracy percent:",
                  gs_rf.score(X_test, y_test) * 100)
        if position==2:
            # save_classifier = open("random_forest2.pickle", "wb")
            # pickle.dump(gs_rf, save_classifier)
            # save_classifier.close()
            print("Thinking Feeling Random Forest Model Train accuracy percent:",
                  gs_rf.score(X_train, y_train) * 100)
            print("Thinking Feeling Random Forest Model Test accuracy percent:",
                  gs_rf.score(X_test, y_test) * 100)
        elif position==3:
            # save_classifier = open("random_forest3.pickle", "wb")
            # pickle.dump(gs_rf, save_classifier)
            # save_classifier.close()
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
            # save_classifier = open("svm0.pickle", "wb")
            # pickle.dump(gs_svm, save_classifier)
            # save_classifier.close()
            print("Introversion Extroversion SVM Model Train accuracy percent:",
                  gs_svm.score(X_train, y_train) * 100)
            print("Introversion Extroversion SVM Model Test accuracy percent:",
                  gs_svm.score(X_test, y_test) * 100)
        elif position == 1:
            # save_classifier = open("svm1.pickle", "wb")
            # pickle.dump(gs_svm, save_classifier)
            # save_classifier.close()
            print("Intuition Sensing SVM Model Train accuracy percent:",
                  gs_svm.score(X_train, y_train) * 100)
            print("Intuition Sensing SVM Model Test accuracy percent:",
                  gs_svm.score(X_test, y_test) * 100)
        if position == 2:
            # save_classifier = open("svm2.pickle", "wb")
            # pickle.dump(gs_svm, save_classifier)
            # save_classifier.close()
            print("Thinking Feeling SVM Model Train accuracy percent:",
                  gs_svm.score(X_train, y_train) * 100)
            print("Thinking Feeling SVM Model Test accuracy percent:",
                  gs_svm.score(X_test, y_test) * 100)
        elif position == 3:
            # save_classifier = open("svm3.pickle", "wb")
            # pickle.dump(gs_svm, save_classifier)
            # save_classifier.close()
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
            # save_classifier = open("decision_tree0.pickle", "wb")
            # pickle.dump(gs_dt, save_classifier)
            # save_classifier.close()
            print("Introversion Extroversion Decision Tree Model Train accuracy percent:",
                  gs_dt.score(X_train, y_train) * 100)
            print("Introversion Extroversion Decision Tree Model Test accuracy percent:",
                  gs_dt.score(X_test, y_test) * 100)
        elif position == 1:
            # save_classifier = open("decision_tree1.pickle", "wb")
            # pickle.dump(gs_dt, save_classifier)
            # save_classifier.close()
            print("Intuition Sensing Decision Tree Model Train accuracy percent:",
                  gs_dt.score(X_train, y_train) * 100)
            print("Intuition Sensing Decision Tree Model Test accuracy percent:",
                  gs_dt.score(X_test, y_test) * 100)
        if position == 2:
            # save_classifier = open("decision_tree2.pickle", "wb")
            # pickle.dump(gs_dt, save_classifier)
            # save_classifier.close()
            print("Thinking Feeling Decision Tree Model Train accuracy percent:",
                  gs_dt.score(X_train, y_train) * 100)
            print("Thinking Feeling Decision Tree Model Test accuracy percent:",
                  gs_dt.score(X_test, y_test) * 100)
        elif position == 3:
            # save_classifier = open("decision_tree3.pickle", "wb")
            # pickle.dump(gs_dt, save_classifier)
            # save_classifier.close()
            print("Judging Perceiving Decision Tree Model Train accuracy percent:",
                  gs_dt.score(X_train, y_train) * 100)
            print("Judging Perceiving Decision Tree Model Test accuracy percent:",
                  gs_dt.score(X_test, y_test) * 100)
    elif model_name == "KNN":
        features = get_model(posts_train, traits_train, position, keys)
        features_test = get_model(posts_test, traits_test, position, keys)
        X_train, y_train, X_test, y_test = obtain_data(features, features_test)
        gs_knn = GridSearchCV(estimator=KNeighborsClassifier(),
                             param_grid=grid_parameters,
                             cv=10,
                             n_jobs=1,
                             verbose=2)
        gs_knn.fit(X_train, y_train)
        print(gs_knn.best_params_)
        if position == 0:
            save_classifier = open("knn0.pickle", "wb")
            pickle.dump(gs_knn, save_classifier)
            save_classifier.close()
            print("Introversion Extroversion KNN Model Train accuracy percent:",
                  gs_knn.score(X_train, y_train) * 100)
            print("Introversion Extroversion KNN Model Test accuracy percent:",
                  gs_knn.score(X_test, y_test) * 100)
        elif position == 1:
            save_classifier = open("knn1.pickle", "wb")
            pickle.dump(gs_knn, save_classifier)
            save_classifier.close()
            print("Intuition Sensing KNN Model Train accuracy percent:",
                  gs_knn.score(X_train, y_train) * 100)
            print("Intuition Sensing KNN Model Test accuracy percent:",
                  gs_knn.score(X_test, y_test) * 100)
        if position == 2:
            save_classifier = open("knn2.pickle", "wb")
            pickle.dump(gs_knn, save_classifier)
            save_classifier.close()
            print("Thinking Feeling KNN Model Train accuracy percent:",
                  gs_knn.score(X_train, y_train) * 100)
            print("Thinking Feeling KNN Model Test accuracy percent:",
                  gs_knn.score(X_test, y_test) * 100)
        elif position == 3:
            save_classifier = open("knn3.pickle", "wb")
            pickle.dump(gs_knn, save_classifier)
            save_classifier.close()
            print("Judging Perceiving KNN Model Train accuracy percent:",
                  gs_knn.score(X_train, y_train) * 100)
            print("Judging Perceiving KNN Model Test accuracy percent:",
                  gs_knn.score(X_test, y_test) * 100)
    elif model_name == "Naive Bayes":
        features = get_model(posts_train, traits_train, position, keys)
        features_test = get_model(posts_test, traits_test, position, keys)
        gs_nb = NaiveBayesClassifier.train(features)
        if position == 0:
            save_classifier = open("nb0.pickle", "wb")
            pickle.dump(gs_nb, save_classifier)
            save_classifier.close()
            print("Introversion Extroversion Naive Bayes Model Train accuracy percent:",
                  nltk.classify.util.accuracy(gs_nb, features) * 100)
            print("Introversion Extroversion Naive Bayes Model Test accuracy percent:",
                  nltk.classify.util.accuracy(gs_nb, features_test) * 100)
        elif position == 1:
            save_classifier = open("nb1.pickle", "wb")
            pickle.dump(gs_nb, save_classifier)
            save_classifier.close()
            print("Introversion Extroversion Naive Bayes Model Train accuracy percent:",
                  nltk.classify.util.accuracy(gs_nb, features) * 100)
            print("Introversion Extroversion Naive Bayes Model Test accuracy percent:",
                  nltk.classify.util.accuracy(gs_nb, features_test) * 100)
        if position == 2:
            save_classifier = open("nb2.pickle", "wb")
            pickle.dump(gs_nb, save_classifier)
            save_classifier.close()
            print("Introversion Extroversion Naive Bayes Model Train accuracy percent:",
                  nltk.classify.util.accuracy(gs_nb, features) * 100)
            print("Introversion Extroversion Naive Bayes Model Test accuracy percent:",
                  nltk.classify.util.accuracy(gs_nb, features_test) * 100)
        elif position == 3:
            save_classifier = open("nb3.pickle", "wb")
            pickle.dump(gs_nb, save_classifier)
            save_classifier.close()
            print("Introversion Extroversion Naive Bayes Model Train accuracy percent:",
                  nltk.classify.util.accuracy(gs_nb, features) * 100)
            print("Introversion Extroversion Naive Bayes Model Test accuracy percent:",
                  nltk.classify.util.accuracy(gs_nb, features_test) * 100)
    elif model_name == "Ensemble":
        features = get_model(posts_train, traits_train, position, keys)
        features_test = get_model(posts_test, traits_test, position, keys)
        X_train, y_train, X_test, y_test = obtain_data(features, features_test)

        if position == 0:
            classifier_f = open("knn0.pickle", "rb")
            knn0 = pickle.load(classifier_f)
            classifier_f = open("random_forest0.pickle", "rb")
            random_forest0 = pickle.load(classifier_f)
            est_Ensemble = VotingClassifier(estimators=[('KNN', knn0), ('RF', random_forest0)],
                                            voting='soft',
                                            weights=[1, 1])
            est_Ensemble.fit(X_train, y_train)
            save_classifier = open("ensemble0.pickle", "wb")
            pickle.dump(est_Ensemble, save_classifier)
            save_classifier.close()
            print("Introversion Extroversion Ensemble Model Train accuracy percent:",
                  est_Ensemble.score * 100)
            print("Introversion Extroversion Ensemble Model Test accuracy percent:",
                  est_Ensemble.score * 100)
        elif position == 1:
            classifier_f = open("knn1.pickle", "rb")
            knn0 = pickle.load(classifier_f)
            classifier_f = open("random_forest1.pickle", "rb")
            random_forest0 = pickle.load(classifier_f)
            est_Ensemble = VotingClassifier(estimators=[('KNN', knn0), ('RF', random_forest0)],
                                            voting='soft',
                                            weights=[1, 1])
            est_Ensemble.fit(X_train, y_train)
            save_classifier = open("ensemble1.pickle", "wb")
            pickle.dump(est_Ensemble, save_classifier)
            save_classifier.close()
            print("Introversion Extroversion Ensemble Model Train accuracy percent:",
                  est_Ensemble.score * 100)
            print("Introversion Extroversion Ensemble Model Test accuracy percent:",
                  est_Ensemble.score * 100)
        elif position == 2:
            classifier_f = open("knn2.pickle", "rb")
            knn0 = pickle.load(classifier_f)
            classifier_f = open("random_forest2.pickle", "rb")
            random_forest0 = pickle.load(classifier_f)
            est_Ensemble = VotingClassifier(estimators=[('KNN', knn0), ('RF', random_forest0)],
                                            voting='soft',
                                            weights=[1, 1])
            est_Ensemble.fit(X_train, y_train)
            save_classifier = open("ensemble2.pickle", "wb")
            pickle.dump(est_Ensemble, save_classifier)
            save_classifier.close()
            print("Introversion Extroversion Ensemble Model Train accuracy percent:",
                  est_Ensemble.score * 100)
            print("Introversion Extroversion Ensemble Model Test accuracy percent:",
                  est_Ensemble.score * 100)
        elif position == 3:
            classifier_f = open("knn3.pickle", "rb")
            knn0 = pickle.load(classifier_f)
            classifier_f = open("random_forest3.pickle", "rb")
            random_forest0 = pickle.load(classifier_f)
            est_Ensemble = VotingClassifier(estimators=[('KNN', knn0), ('RF', random_forest0)],
                                            voting='soft',
                                            weights=[1, 1])
            est_Ensemble.fit(X_train, y_train)
            save_classifier = open("ensemble3.pickle", "wb")
            pickle.dump(est_Ensemble, save_classifier)
            save_classifier.close()
            print("Introversion Extroversion Ensemble Model Train accuracy percent:",
                  est_Ensemble.score * 100)
            print("Introversion Extroversion Ensemble Model Test accuracy percent:",
                  est_Ensemble.score * 100)


print("======================================================================")

# log_reg_grid = {"C": np.logspace(-4, 4, 30),
#                     "solver": ["liblinear"]}
# grid_search_for_model(log_reg_grid, "Logistic Regression", 0)
# grid_search_for_model(log_reg_grid, "Logistic Regression", 1)
# grid_search_for_model(log_reg_grid, "Logistic Regression", 2)
# grid_search_for_model(log_reg_grid, "Logistic Regression", 3)
# rf_grid = {#"n_estimators": np.arange(100, 1000, 100),
#            "max_depth": [None, 3, 5, 10],
#            "min_samples_split": np.arange(2, 20, 3),
#            #"min_samples_leaf": np.arange(1, 20, 3)
#            }
# grid_search_for_model(rf_grid, "Random Forest", 0)
# grid_search_for_model(rf_grid, "Random Forest", 1)
# grid_search_for_model(rf_grid, "Random Forest", 2)
# grid_search_for_model(rf_grid, "Random Forest", 3)
# svm_grid = {'C': [0.1, 1, 10, 100],
#             #'gamma': [1, 0.1, 0.01, 0.001],
#             #'kernel': ['rbf', 'poly', 'sigmoid']
#             }
# grid_search_for_model(svm_grid, "SVM", 0)
# grid_search_for_model(svm_grid, "SVM", 1)
# grid_search_for_model(svm_grid, "SVM", 2)
# grid_search_for_model(svm_grid, "SVM", 3)
# dt_grid =  {
#     'min_samples_leaf': [1, 2, 3],
#     'max_depth': [1, 2, 3],
#     'criterion': ['gini', 'entropy']
# }
# grid_search_for_model(dt_grid, "Decision Tree", 0)
# grid_search_for_model(dt_grid, "Decision Tree", 1)
# grid_search_for_model(dt_grid, "Decision Tree", 2)
# grid_search_for_model(dt_grid, "Decision Tree", 3)

# knn_grid = {
#     'n_neighbors' : [2, 4],
#     'leaf_size' : [10, 20],
#     'p' : [1,2]
# }
# grid_search_for_model(knn_grid, "KNN", 0)
# grid_search_for_model(knn_grid, "KNN", 1)
# grid_search_for_model(knn_grid, "KNN", 2)
# grid_search_for_model(knn_grid, "KNN", 3)

knn_grid = {
}
# grid_search_for_model(knn_grid, "Naive Bayes", 0)
# grid_search_for_model(knn_grid, "Naive Bayes", 1)
# grid_search_for_model(knn_grid, "Naive Bayes", 2)
# grid_search_for_model(knn_grid, "Naive Bayes", 3)

grid_search_for_model(knn_grid, "Ensemble", 0)
grid_search_for_model(knn_grid, "Ensemble", 1)
grid_search_for_model(knn_grid, "Ensemble", 2)
grid_search_for_model(knn_grid, "Ensemble", 3)

