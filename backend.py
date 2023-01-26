from load_classifiers import modify_data
from load_classifiers import grid_search_for_model


def svm(text):
    response = grid_search_for_model("SVM", modify_data(text))
    return response


def decision_tree(text):
    response = grid_search_for_model("Decision Tree", modify_data(text))
    return response


def random_forest(text):
    response = grid_search_for_model("Random Forest", modify_data(text))
    return response


def logistic_regression(text):
    response = grid_search_for_model("Logistic Regression", modify_data(text))
    return response


def naive_bayes(text):
    response = grid_search_for_model("Naive Bayes", modify_data(text))
    return response


def knn(text):
    response = grid_search_for_model("KNN", modify_data(text))
    return response


def ensemble(text):
    response = grid_search_for_model("Ensemble", modify_data(text))
    return response