import os
import numpy as np
import sklearn
import feature_extraction
import kernels
import parsers
import function_words

federalist_papers_path = os.path.join(os.path.dirname(__file__), 'input\\The_federalist_papers.txt')
federalist_papers_authors = {
    "HAMILTON": 0,
    "MADISON": 1,
    "JAY": 2
}


def svm_with_rank_distance(train_data, eval_data):
    # train
    train_features, train_labels = (np.array(train_data[0]), np.array(train_data[1]))

    svm = sklearn.svm.SVC(kernel=kernels.linear_normalized, probability=True)
    svm.fit(train_features, train_labels)

    # eval
    eval_features, eval_labels = (np.array(eval_data[0]), np.array(eval_data[1]))

    predict = svm.predict(eval_features)
    predict_log = svm.predict_proba(eval_features)
    predict_score = svm.score(eval_features, eval_labels)

    return predict, predict_log, predict_score


def k_neighbors_with_rank_distance(train_data, eval_data):
    # train
    train_features, train_labels = (np.array(train_data[0]), np.array(train_data[1]))

    svm = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
    svm.fit(train_features, train_labels)

    # eval
    eval_features, eval_labels = (np.array(eval_data[0]), np.array(eval_data[1]))

    predict = svm.predict(eval_features)
    predict_log = svm.predict_proba(eval_features)
    predict_score = svm.score(eval_features, eval_labels)

    return predict, predict_log, predict_score


def print_results(eval_dataset_values, eval_labels, predict, predict_log, predict_score):
    print("Eval:")
    print("Score: {}".format(predict_score))
    for index, dataset_value in enumerate(eval_dataset_values):
        print('{} -- {} -- {} -- {} -- {}'.format(
            dataset_value[0],
            dataset_value[1],
            eval_labels[index],
            predict[index],
            predict_log[index]))


def run():
    dataset = parsers.parse_federalist_papers(federalist_papers_path)
    stop_words = function_words.mosteller_and_wallace

    train_dataset_values = list(filter(lambda x: x[1] in list(federalist_papers_authors.keys())[:2], dataset.values))
    train_data = (list(map(lambda x: feature_extraction.ranking_distance(x[2], stop_words), train_dataset_values)),
                  list(map(lambda x: federalist_papers_authors[x[1]], train_dataset_values)))

    eval_dataset_values = list(filter(lambda x: x[1] == "HAMILTON OR MADISON", dataset.values))
    eval_data = (list(map(lambda x: feature_extraction.ranking_distance(x[2], stop_words), eval_dataset_values)),
                 [1] * 11)

    print("\nSVM with rank distance:")
    for i in range(1):
        predict, predict_log, predict_score = svm_with_rank_distance(train_data, eval_data)
        print_results(eval_dataset_values, eval_data[1], predict, predict_log, predict_score)

    print("\nKNeighbors with rank distance:")
    for i in range(1):
        predict, predict_log, predict_score = k_neighbors_with_rank_distance(train_data, eval_data)
        print_results(eval_dataset_values, eval_data[1], predict, predict_log, predict_score)


if __name__ == "__main__":
    run()
