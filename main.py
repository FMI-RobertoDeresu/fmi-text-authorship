import os
import numpy as np
import sklearn
import re
import operator
import feature_extraction
import kernels
import parsers
import function_words
from sklearn.cross_decomposition import PLSRegression

cfg = {
    "run_test": 0,
    "run_federalist": 0,
    "run_pan_11": 0,
    "run_pan_12": 1,

    "federalist_papers_path": os.path.join(os.path.dirname(__file__), 'input\\federalist_papers.txt'),
    "federalist_papers_authors": {"HAMILTON": 0, "MADISON": 1, "JAY": 2},

    "pan_11_path": os.path.join(os.path.dirname(__file__), 'input\\pan11\\small'),
    # "pan_11_path": os.path.join(os.path.dirname(__file__), 'input\\pan11\\large'),
    "pan_12_path": os.path.join(os.path.dirname(__file__), 'input\\pan12'),

    "use_svm_and_rank_distance": 0,
    "use_kneighbors_and_rank_distance": 0,
    "use_pls_and_rank_distance": 0,

    "use_svm_and_ngrams": 1,
    "use_kneighbors_and_ngrams": 0,
    "use_pls_and_ngrams": 0,

    "k_neighbours_to_use": 3,

    "n_grams_min_length": 4,
    "n_grams_max_length": 5,
    "n_grams_most_common": 10000,
    "n_grams_frequency_threshold": 5,

    "ranking_distance_kernel": kernels.linear,
    "n_grams_kernel": kernels.linear,

    "print_probs": True
    # "print_probs": False
}


def run_svm(train_data, eval_data, kernel):
    # train
    train_features, train_labels = (kernel(np.array(train_data[0]), np.array(train_data[0])), np.array(train_data[1]))

    svm = sklearn.svm.LinearSVC(max_iter=1000000, penalty='l1', dual=False)
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


def run(train_dataset_values, train_labels, eval_dataset_values, eval_labels, stopwords):
    # ranking distance
    if cfg['use_svm_and_rank_distance'] or cfg['use_kneighbors_and_rank_distance'] or cfg["use_pls_and_rank_distance"]:
        train_data = (
            list(map(lambda x: feature_extraction.ranking_distance(x[2], stopwords), train_dataset_values)),
            train_labels
        )

        eval_data = (
            list(map(lambda x: feature_extraction.ranking_distance(x[2], stopwords), eval_dataset_values)),
            eval_labels
        )

        if cfg['use_svm_and_rank_distance'] > 0:
            print("\nSVM with rank distance:")
            for i in range(cfg['use_svm_and_rank_distance']):
                predict, predict_probs, predict_score = run_svm(train_data, eval_data, cfg['ranking_distance_kernel'])
                print_results(eval_dataset_values, eval_labels, predict, predict_probs, predict_score)

        if cfg['use_kneighbors_and_rank_distance'] > 0:
            print("\nKNeighbors with rank distance:")
            for i in range(cfg['use_kneighbors_and_rank_distance']):
                predict, predict_probs, predict_score = run_k_neighbors(train_data, eval_data,
                                                                        cfg['k_neighbours_to_use'])
                print_results(eval_dataset_values, eval_labels, predict, predict_probs, predict_score)

        if cfg['use_pls_and_rank_distance'] > 0:
            print("\nPLS with rank distance:")
            for i in range(cfg['use_pls_and_rank_distance']):
                predict, predict_probs, predict_score = run_pls(train_data, eval_data, cfg['ranking_distance_kernel'])
                print_results(eval_dataset_values, eval_labels, predict, predict_probs, predict_score)

    # n-grams
    if cfg['use_svm_and_ngrams'] or cfg['use_kneighbors_and_ngrams'] or cfg['use_pls_and_ngrams']:
        inputs = list(map(lambda x: x[2], train_dataset_values))
        ngrams_min = cfg['n_grams_min_length']
        ngrams_max = cfg['n_grams_max_length']
        ngrams_top = cfg['n_grams_most_common']
        freq_th = cfg['n_grams_frequency_threshold']

        n_grams_vocabulary = feature_extraction.n_grams_vocabulary(inputs, ngrams_min, ngrams_max, ngrams_top,  freq_th)
        print("NGrams vocabulary length " + str(len(n_grams_vocabulary)))

        train_texts = list(map(operator.itemgetter(2), train_dataset_values))
        train_texts_ngrams = feature_extraction.n_grams(train_texts, n_grams_vocabulary, ngrams_min, ngrams_max)
        train_data = (train_texts_ngrams, train_labels)

        eval_texts = list(map(operator.itemgetter(2), eval_dataset_values))
        eval_texts_ngrams = feature_extraction.n_grams(eval_texts, n_grams_vocabulary, ngrams_min, ngrams_max)
        eval_data = (eval_texts_ngrams, eval_labels)

        if cfg['use_svm_and_ngrams'] > 0:
            print("\nSVM with ngrams:")
            for i in range(cfg['use_svm_and_ngrams']):
                predict, predict_probs, predict_score = run_svm(train_data, eval_data, cfg['n_grams_kernel'])
                print_results(eval_dataset_values, eval_labels, predict, predict_probs, predict_score)

        if cfg['use_kneighbors_and_ngrams'] > 0:
            print("\nKNeighbors with ngrams:")
            for i in range(cfg['use_kneighbors_and_ngrams']):
                predict, predict_probs, predict_score = run_k_neighbors(train_data, eval_data,
                                                                        cfg['k_neighbours_to_use'])
                print_results(eval_dataset_values, eval_labels, predict, predict_probs, predict_score)

        if cfg['use_pls_and_ngrams'] > 0:
            print("\nPLS with ngrams:")
            for i in range(cfg['use_pls_and_ngrams']):
                predict, predict_probs, predict_score = run_pls(train_data, eval_data, cfg['n_grams_kernel'])
                print_results(eval_dataset_values, eval_labels, predict, predict_probs, predict_score)


def print_results(eval_dataset_values, eval_labels, predict, predict_probs, predict_score):
    print("Eval:")
    print("Score: {}".format(predict_score))

    indexes = list(enumerate(eval_labels))
    indexes.sort(key=operator.itemgetter(1, 0))
    indexes = list(list(zip(*indexes))[0])

    for index in indexes:
        dataset_value = eval_dataset_values[index]
        probs_str = ' '.join(str(x) for x in predict_probs[index]) if predict_probs[index] is not None else "None"
        print('{} -- {} -- {} -- {} -- {} -- {}'.format(
            "OK   " if eval_labels[index] == predict[index] else "FAIL ",
            dataset_value[0],
            dataset_value[1],
            eval_labels[index],
            predict[index],
            probs_str if cfg["print_probs"] else "no-print"))


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


def run_pan_11():
    print("\nRun pan 11 " + "small" if cfg["pan_11_path"].endswith('small') else "large")
    dataset = parsers.parse_pan_11_dataset(cfg["pan_11_path"])

    train_dataset_values = list(filter(lambda x: not (x[1].startswith('unknown')), dataset.values))
    train_labels = list(map(lambda x: int(re.search("\d+", x[1]).group()), train_dataset_values))

    eval_dataset_values = list(filter(lambda x: x[1].startswith('unknown'), dataset.values))
    eval_labels = list(map(lambda x: int(re.search("\d+", x[3]).group()), eval_dataset_values))

    run(train_dataset_values, train_labels, eval_dataset_values, eval_labels, function_words.mosteller_and_wallace)


def run_pan_12():
    print("\nRun pan 12")
    datasets = parsers.parse_pan_12_dataset(cfg["pan_12_path"])

    for dataset_name, dataset in datasets.items():
        print("\nDataset " + dataset_name)

        train_dataset_values = list(filter(lambda x: not (x[1].startswith('unknown')), dataset.values))
        train_labels = list(map(lambda x: int(re.search("\d+", x[1]).group()), train_dataset_values))

        eval_dataset_values = list(filter(lambda x: x[1].startswith('unknown'), dataset.values))
        eval_labels = list(map(lambda x: int(re.search("\d+", x[3]).group()), eval_dataset_values))

        run(train_dataset_values, train_labels, eval_dataset_values, eval_labels, function_words.mosteller_and_wallace)


if __name__ == "__main__":
    if cfg["run_test"]:
        run_test()

    if cfg["run_federalist"]:
        run_federalist()

    if cfg["run_pan_11"]:
        run_pan_11()

    if cfg["run_pan_12"]:
        run_pan_12()
