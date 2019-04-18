import os
import numpy as np
import sklearn
import feature_extraction
import kernels
import parsers
import function_words

cfg = {
    "run_federalist": 1,
    "svm_and_rank_distance": 0,
    "kneighbors_and_rank_distance": 0,
    "svm_and_ngrams": 1,
    "kneighbors_and_ngrams": 1
}

federalist_papers_path = os.path.join(os.path.dirname(__file__), 'input\\The_federalist_papers.txt')
federalist_papers_authors = {
    "HAMILTON": 0,
    "MADISON": 1,
    "JAY": 2
}


def run_svm(train_data, eval_data, kernel):
    # train
    train_features, train_labels = (np.array(train_data[0]), np.array(train_data[1]))

    svm = sklearn.svm.SVC(kernel=kernel, probability=True)
    svm.fit(train_features, train_labels)

    # eval
    eval_features, eval_labels = (np.array(eval_data[0]), np.array(eval_data[1]))

    predict = svm.predict(eval_features)
    predict_log = svm.predict_proba(eval_features)
    predict_score = svm.score(eval_features, eval_labels)

    return predict, predict_log, predict_score


def run_k_neighbors(train_data, eval_data, n_neighbors):
    # train
    train_features, train_labels = (np.array(train_data[0]), np.array(train_data[1]))

    svm = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    svm.fit(train_features, train_labels)

    # eval
    eval_features, eval_labels = (np.array(eval_data[0]), np.array(eval_data[1]))

    predict = svm.predict(eval_features)
    predict_log = svm.predict_proba(eval_features)
    predict_score = svm.score(eval_features, eval_labels)

    return predict, predict_log, predict_score


def run_with_rank_distance(train_dataset_values, eval_dataset_values, stop_words):
    train_data = (
        list(map(lambda x: feature_extraction.ranking_distance(x[2], stop_words), train_dataset_values)),
        list(map(lambda x: federalist_papers_authors[x[1]], train_dataset_values))
    )

    eval_data = (
        list(map(lambda x: feature_extraction.ranking_distance(x[2], stop_words), eval_dataset_values)),
        [1] * 11
    )

    if cfg['svm_and_rank_distance'] > 0:
        print("\nSVM with rank distance:")
        for i in range(cfg['svm_and_rank_distance']):
            predict, predict_log, predict_score = run_svm(train_data, eval_data, kernels.linear)
            print_results(eval_dataset_values, eval_data[1], predict, predict_log, predict_score)

    if cfg['kneighbors_and_rank_distance'] > 0:
        print("\nKNeighbors with rank distance:")
        for i in range(cfg['kneighbors_and_rank_distance']):
            predict, predict_log, predict_score = run_k_neighbors(train_data, eval_data, 3)
            print_results(eval_dataset_values, eval_data[1], predict, predict_log, predict_score)


def run_with_n_grams(train_dataset_values, eval_dataset_values, n_grams_length):
    inputs = list(map(lambda x: x[2], train_dataset_values))
    n_grams_keys = feature_extraction.n_grams_keys(inputs, n_grams_length)

    train_data = (
        list(map(lambda x: feature_extraction.n_grams(x[2], n_grams_keys, n_grams_length), train_dataset_values)),
        list(map(lambda x: federalist_papers_authors[x[1]], train_dataset_values))
    )

    eval_data = (
        list(map(lambda x: feature_extraction.n_grams(x[2], n_grams_keys, n_grams_length), eval_dataset_values)),
        [1] * 11
    )

    if cfg['svm_and_ngrams'] > 0:
        print("\nSVM with ngrams:")
        for i in range(cfg['svm_and_ngrams']):
            predict, predict_log, predict_score = run_svm(train_data, eval_data, kernels.linear)
            print_results(eval_dataset_values, eval_data[1], predict, predict_log, predict_score)

    if cfg['kneighbors_and_ngrams'] > 0:
        print("\nKNeighbors with ngrams:")
        for i in range(cfg['kneighbors_and_ngrams']):
            predict, predict_log, predict_score = run_k_neighbors(train_data, eval_data, 3)
            print_results(eval_dataset_values, eval_data[1], predict, predict_log, predict_score)


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


def run_federalist():
    print("\nRun federalist papers test")
    dataset = parsers.parse_federalist_papers(federalist_papers_path)

    train_dataset_values = list(filter(lambda x: x[1] in list(federalist_papers_authors.keys())[:2], dataset.values))
    eval_dataset_values = list(filter(lambda x: x[1] == "HAMILTON OR MADISON", dataset.values))

    if cfg['svm_and_rank_distance'] or cfg['kneighbors_and_rank_distance']:
        stop_words = function_words.mosteller_and_wallace
        run_with_rank_distance(train_dataset_values, eval_dataset_values, stop_words)

    if cfg['svm_and_ngrams'] or cfg['kneighbors_and_ngrams']:
        run_with_n_grams(train_dataset_values, eval_dataset_values, n_grams_length=5)


if __name__ == "__main__":
    if cfg["run_federalist"]:
        run_federalist()
