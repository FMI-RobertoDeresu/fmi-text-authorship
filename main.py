import os
import numpy as np
import sklearn
import feature_extraction
import kernels
import parsers
import function_words
from sklearn.cross_decomposition import PLSRegression

cfg = {
    "run_test": 0,
    "run_federalist": 1,

    "federalist_papers_path": os.path.join(os.path.dirname(__file__), 'input\\the_federalist_papers.txt'),
    "federalist_papers_authors": {"HAMILTON": 0, "MADISON": 1, "JAY": 2},

    "use_svm_and_rank_distance": 0,
    "use_kneighbors_and_rank_distance": 0,
    "use_pls_and_rank_distance": 0,

    "use_svm_and_ngrams": 1,
    "use_kneighbors_and_ngrams": 0,
    "use_pls_and_ngrams": 0,

    "svm_decision_function": "ovr",

    "k_neighbours_to_use": 3,

    "n_grams_min_length": 4,
    "n_grams_max_length": 5,
    "n_grams_most_common_to_use": 10000,

    "ranking_distance_kernel": kernels.linear,
    "n_grams_kernel": kernels.linear
}


def run_svm(train_data, eval_data, kernel):
    # train
    train_features, train_labels = (kernel(np.array(train_data[0]), np.array(train_data[0])), np.array(train_data[1]))

    svm = sklearn.svm.LinearSVC(max_iter=100000, penalty='l1', dual=False)
    svm.fit(train_features, train_labels)

    # eval
    eval_features, eval_labels = (kernel(np.array(eval_data[0]), np.array(train_data[0])), np.array(eval_data[1]))

    predict = svm.predict(eval_features)
    predict_probs = [None] * len(eval_labels)
    predict_score = svm.score(eval_features, eval_labels)

    return predict, predict_probs, predict_score


def run_k_neighbors(train_data, eval_data, n_neighbors):
    # train
    train_features, train_labels = (np.array(train_data[0]), np.array(train_data[1]))

    svm = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    svm.fit(train_features, train_labels)

    # eval
    eval_features, eval_labels = (np.array(eval_data[0]), np.array(eval_data[1]))

    predict = svm.predict(eval_features)
    predict_probs = svm.predict_proba(eval_features)
    predict_score = svm.score(eval_features, eval_labels)

    return predict, predict_probs, predict_score


def run_pls(train_data, eval_data, kernel):
    # train
    train_features, train_labels = (kernel(np.array(train_data[0]), np.array(train_data[0])), np.array(train_data[1]))

    pls = PLSRegression(n_components=4, scale=True, max_iter=100000)
    pls.fit(train_features, train_labels)

    # eval
    eval_features, eval_labels = (kernel(np.array(eval_data[0]), np.array(train_data[0])), np.array(eval_data[1]))

    predict = pls.predict(eval_features)
    predict_labels = list(map(lambda x: x[0], np.where(predict > 0.5, 1, 0)))
    predict_probs = predict
    predict_score = sum([1 for i, j in zip(predict_labels, eval_labels) if i == j]) / len(eval_labels)

    return predict_labels, predict_probs, predict_score


def run(train_dataset_values, train_labels, eval_dataset_values, eval_labels, ranking_distance_stopwords):
    # ranking distance
    if cfg['use_svm_and_rank_distance'] or cfg['use_kneighbors_and_rank_distance'] or cfg["use_pls_and_rank_distance"]:
        train_data = (
            list(map(lambda x: feature_extraction.ranking_distance(x[2], ranking_distance_stopwords),
                     train_dataset_values)),
            train_labels
        )

        eval_data = (
            list(map(lambda x: feature_extraction.ranking_distance(x[2], ranking_distance_stopwords),
                     eval_dataset_values)),
            eval_labels
        )

        if cfg['use_svm_and_rank_distance'] > 0:
            print("\nSVM with rank distance:")
            for i in range(cfg['use_svm_and_rank_distance']):
                predict, predict_probs, predict_score = run_svm(train_data, eval_data, cfg['ranking_distance_kernel'])
                print_results(eval_dataset_values, eval_data[1], predict, predict_probs, predict_score)

        if cfg['use_kneighbors_and_rank_distance'] > 0:
            print("\nKNeighbors with rank distance:")
            for i in range(cfg['use_kneighbors_and_rank_distance']):
                predict, predict_probs, predict_score = run_k_neighbors(train_data, eval_data,
                                                                        cfg['k_neighbours_to_use'])
                print_results(eval_dataset_values, eval_data[1], predict, predict_probs, predict_score)

        if cfg['use_pls_and_rank_distance'] > 0:
            print("\nPLS with rank distance:")
            for i in range(cfg['use_pls_and_rank_distance']):
                predict, predict_probs, predict_score = run_pls(train_data, eval_data, cfg['ranking_distance_kernel'])
                print_results(eval_dataset_values, eval_data[1], predict, predict_probs, predict_score)

    # n-grams
    if cfg['use_svm_and_ngrams'] or cfg['use_kneighbors_and_ngrams'] or cfg['use_pls_and_ngrams']:
        inputs = list(map(lambda x: x[2], train_dataset_values))
        n_grams_keys = feature_extraction.n_grams_keys(inputs,
                                                       cfg['n_grams_min_length'],
                                                       cfg['n_grams_max_length'],
                                                       cfg['n_grams_most_common_to_use'])

        train_data = (
            list(map(lambda x: feature_extraction.n_grams(x[2],
                                                          n_grams_keys,
                                                          cfg['n_grams_min_length'],
                                                          cfg['n_grams_max_length']),
                     train_dataset_values)),
            train_labels
        )

        eval_data = (
            list(map(lambda x: feature_extraction.n_grams(x[2],
                                                          n_grams_keys,
                                                          cfg['n_grams_min_length'],
                                                          cfg['n_grams_max_length']),
                     eval_dataset_values)),
            eval_labels
        )

        if cfg['use_svm_and_ngrams'] > 0:
            print("\nSVM with ngrams:")
            for i in range(cfg['use_svm_and_ngrams']):
                predict, predict_probs, predict_score = run_svm(train_data, eval_data, cfg['n_grams_kernel'])
                print_results(eval_dataset_values, eval_data[1], predict, predict_probs, predict_score)

        if cfg['use_kneighbors_and_ngrams'] > 0:
            print("\nKNeighbors with ngrams:")
            for i in range(cfg['use_kneighbors_and_ngrams']):
                predict, predict_probs, predict_score = run_k_neighbors(train_data, eval_data,
                                                                        cfg['k_neighbours_to_use'])
                print_results(eval_dataset_values, eval_data[1], predict, predict_probs, predict_score)

        if cfg['use_pls_and_ngrams'] > 0:
            print("\nPLS with ngrams:")
            for i in range(cfg['use_pls_and_ngrams']):
                predict, predict_probs, predict_score = run_pls(train_data, eval_data, cfg['n_grams_kernel'])
                print_results(eval_dataset_values, eval_data[1], predict, predict_probs, predict_score)


def print_results(eval_dataset_values, eval_labels, predict, predict_probs, predict_score):
    print("Eval:")
    print("Score: {}".format(predict_score))
    for index, dataset_value in enumerate(eval_dataset_values):
        print('{} -- {} -- {} -- {} -- {}'.format(
            dataset_value[0],
            dataset_value[1],
            eval_labels[index],
            predict[index],
            predict_probs[index]))


def run_test():
    print("\nRun test")
    dataset = parsers.parse_test()

    train_dataset_values = dataset.values[:10]
    train_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    eval_dataset_values = dataset.values[10:]
    eval_labels = [0, 0, 1, 1]

    run(train_dataset_values, train_labels, eval_dataset_values, eval_labels, function_words.romanian)


def run_federalist():
    print("\nRun federalist papers")
    authors = cfg["federalist_papers_authors"]
    dataset = parsers.parse_federalist_papers(cfg["federalist_papers_path"])

    train_dataset_values = list(filter(lambda x: x[1] in list(authors.keys())[:2], dataset.values))
    train_labels = list(map(lambda x: authors[x[1]], train_dataset_values))

    eval_dataset_values = list(filter(lambda x: x[1] == "HAMILTON OR MADISON", dataset.values))
    eval_labels = [1] * 11

    run(train_dataset_values, train_labels, eval_dataset_values, eval_labels, function_words.mosteller_and_wallace)


if __name__ == "__main__":
    if cfg["run_test"]:
        run_test()

    if cfg["run_federalist"]:
        run_federalist()
