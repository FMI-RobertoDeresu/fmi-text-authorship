import os
import numpy as np
import sklearn
import re
import operator
import feature_extraction
import kernels
import parsers
import function_words
from sklearn import multiclass

cfg = {
    "run_test": 0,
    "run_federalist": 1,
    "run_pan_11": 0,
    "run_pan_12": 0,

    "federalist_papers_path": os.path.join(os.path.dirname(__file__), 'input\\federalist_papers.txt'),
    "federalist_papers_authors": {"HAMILTON": 0, "MADISON": 1, "JAY": 2},

    "pan_11_path": os.path.join(os.path.dirname(__file__), 'input\\pan11'),
    "pan_12_path": os.path.join(os.path.dirname(__file__), 'input\\pan12'),

    "use_svm_and_rank_distance": 0,
    "use_kneighbors_and_rank_distance": 0,

    "use_svm_and_ngrams": 1,
    "use_kneighbors_and_ngrams": 0,

    "k_neighbours_to_use": 3,

    "n_grams_min_length": 4,
    "n_grams_max_length": 4,
    "n_grams_most_common": 10000,
    "n_grams_frequency_threshold": 100,

    "ranking_distance_kernel": kernels.linear,
    "n_grams_kernel": kernels.linear,

    "print_probs": True
    # "print_probs": False
}


def run_svm(train_data, eval_data, kernel):
    train_features, train_labels = (train_data[0], train_data[1])
    eval_features, eval_labels = (eval_data[0], eval_data[1])

    # train
    svm = sklearn.svm.SVC(kernel=kernel, probability=True)
    clf = multiclass.OneVsRestClassifier(svm).fit(train_features, train_labels)

    # eval
    predict = clf.predict(eval_features)
    predict_probs = clf.predict_proba(eval_features)
    predict_score = clf.score(eval_features, eval_labels)

    return predict, predict_probs, predict_score


def run_k_neighbors(train_data, eval_data, n_neighbors):
    train_features, train_labels = (train_data[0], train_data[1])
    eval_features, eval_labels = (eval_data[0], eval_data[1])

    # train
    svm = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    svm.fit(train_features, train_labels)

    # eval
    predict = svm.predict(eval_features)
    predict_probs = svm.predict_proba(eval_features)
    predict_score = svm.score(eval_features, eval_labels)

    return predict, predict_probs, predict_score


def run(train_dataset_values, train_labels, eval_dataset_values, eval_labels, stopwords, n_grams_vocab_file=None):
    max_abs_scaler = sklearn.preprocessing.MaxAbsScaler()
    scores = np.zeros(4)

    # ranking distance
    if cfg['use_svm_and_rank_distance'] or cfg['use_kneighbors_and_rank_distance']:
        train_features = list(map(lambda x: feature_extraction.ranking_distance(x[2], stopwords), train_dataset_values))
        train_data = (np.array(train_features), np.array(train_labels))

        eval_features = list(map(lambda x: feature_extraction.ranking_distance(x[2], stopwords), eval_dataset_values))
        eval_data = (np.array(eval_features), np.array(eval_labels))

        if cfg['use_svm_and_rank_distance'] > 0:
            print("\nSVM with rank distance:")
            for i in range(cfg['use_svm_and_rank_distance']):
                predict, predict_probs, predict_score = run_svm(train_data, eval_data, cfg['ranking_distance_kernel'])
                scores[0] = predict_score
                print_results(eval_dataset_values, eval_labels, predict, predict_probs, predict_score)

        if cfg['use_kneighbors_and_rank_distance'] > 0:
            print("\nKNeighbors with rank distance:")
            for i in range(cfg['use_kneighbors_and_rank_distance']):
                predict, predict_probs, predict_score = run_k_neighbors(train_data, eval_data,
                                                                        cfg['k_neighbours_to_use'])
                scores[1] = predict_score
                print_results(eval_dataset_values, eval_labels, predict, predict_probs, predict_score)

    # n-grams
    if cfg['use_svm_and_ngrams'] or cfg['use_kneighbors_and_ngrams']:
        inputs = list(map(lambda x: x[2], train_dataset_values))
        ngrams_min = cfg['n_grams_min_length']
        ngrams_max = cfg['n_grams_max_length']
        ngrams_top = cfg['n_grams_most_common']
        freq_th = cfg['n_grams_frequency_threshold']

        n_grams_vocabulary = feature_extraction \
            .n_grams_vocabulary(inputs, ngrams_min, ngrams_max, ngrams_top, freq_th, n_grams_vocab_file)

        print("NGrams vocabulary length " + str(len(n_grams_vocabulary)))

        train_texts = list(map(operator.itemgetter(2), train_dataset_values))
        train_texts_ngrams = feature_extraction.n_grams(train_texts, n_grams_vocabulary, ngrams_min, ngrams_max)
        train_features = max_abs_scaler.fit_transform(np.array(train_texts_ngrams))
        train_data = (np.array(train_features), np.array(train_labels))

        eval_texts = list(map(operator.itemgetter(2), eval_dataset_values))
        eval_texts_ngrams = feature_extraction.n_grams(eval_texts, n_grams_vocabulary, ngrams_min, ngrams_max)
        eval_features = max_abs_scaler.fit_transform(np.array(eval_texts_ngrams))
        eval_data = (np.array(eval_features), np.array(eval_labels))

        if cfg['use_svm_and_ngrams'] > 0:
            print("\nSVM with ngrams:")
            for i in range(cfg['use_svm_and_ngrams']):
                predict, predict_probs, predict_score = run_svm(train_data, eval_data, cfg['n_grams_kernel'])
                scores[2] = predict_score
                print_results(eval_dataset_values, eval_labels, predict, predict_probs, predict_score)

        if cfg['use_kneighbors_and_ngrams'] > 0:
            print("\nKNeighbors with ngrams:")
            for i in range(cfg['use_kneighbors_and_ngrams']):
                predict, predict_probs, predict_score = run_k_neighbors(train_data, eval_data,
                                                                        cfg['k_neighbours_to_use'])
                scores[3] = predict_score
                print_results(eval_dataset_values, eval_labels, predict, predict_probs, predict_score)

    return scores


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
            "OK  " if eval_labels[index] == predict[index] else "FAIL",
            dataset_value[0],
            dataset_value[1],
            eval_labels[index],
            predict[index],
            probs_str if cfg["print_probs"] else "no-print"))


def print_final_scores(scores):
    print("\nTotal scores:")
    if cfg['use_svm_and_rank_distance'] > 0:
        print("SVM with rank distance: " + str(scores[0]))
    if cfg['use_kneighbors_and_rank_distance'] > 0:
        print("KNeighbors with rank distance: " + str(scores[1]))
    if cfg['use_svm_and_ngrams'] > 0:
        print("SVM with ngrams: " + str(scores[2]))
    if cfg['use_kneighbors_and_ngrams'] > 0:
        print("KNeighbors with ngrams: " + str(scores[3]))


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

    n_grams_vocab_file = get_ngrams_vocab_file_path("federalist_papers")

    run(train_dataset_values, train_labels, eval_dataset_values, eval_labels, function_words.mosteller_and_wallace,
        n_grams_vocab_file)


def run_pan_11():
    dataset_name = "small" if cfg["pan_11_path"].endswith('small') else "large"
    print("\nRun pan 11 " + dataset_name)
    dataset = parsers.parse_pan_11_dataset(cfg["pan_11_path"])

    train_dataset_values = list(filter(lambda x: not (x[1].startswith('unknown')), dataset.values))
    train_labels = list(map(lambda x: int(re.search("\d+", x[1]).group()), train_dataset_values))

    eval_dataset_values = list(filter(lambda x: x[1].startswith('unknown'), dataset.values))
    eval_labels = list(map(lambda x: int(re.search("\d+", x[3]).group()), eval_dataset_values))

    n_grams_vocab_file = get_ngrams_vocab_file_path("pan_11_" + dataset_name)

    run(train_dataset_values, train_labels, eval_dataset_values, eval_labels, function_words.nltk_english,
        n_grams_vocab_file)


def run_pan_12():
    print("\nRun pan 12")
    datasets = parsers.parse_pan_12_dataset(cfg["pan_12_path"])
    total_scores = np.zeros(4)

    for dataset_name, dataset in datasets.items():
        print("\nDataset " + dataset_name)

        train_dataset_values = list(filter(lambda x: not (x[1].startswith('unknown')), dataset.values))
        train_labels = list(map(lambda x: int(re.search("\d+", x[1]).group()), train_dataset_values))

        eval_dataset_values = list(filter(lambda x: x[1].startswith('unknown'), dataset.values))
        eval_labels = list(map(lambda x: int(re.search("\d+", x[3]).group()), eval_dataset_values))

        n_grams_vocab_file = get_ngrams_vocab_file_path("pan_12_" + dataset_name)

        scores = run(train_dataset_values, train_labels, eval_dataset_values, eval_labels,
                     function_words.nltk_english, n_grams_vocab_file)
        total_scores = total_scores + scores

    total_scores /= len(datasets)
    print_final_scores(total_scores)


def find_best_config(method):
    for ngrams_min in [3, 4, 5, 6]:
        for ngrams_max in range(ngrams_min, 7):
            for freq_th in [50, 100, 200]:
                cfg['n_grams_min_length'] = ngrams_min
                cfg['n_grams_max_length'] = ngrams_max
                cfg['n_grams_frequency_threshold'] = freq_th

                print("\n\n" + str(ngrams_min) + " -- " + str(ngrams_max) + " -- " + str(freq_th))
                method()


def get_ngrams_vocab_file_path(file_prefix):
    file_name = file_prefix + "_ngrams_vocab_" + \
                str(cfg['n_grams_min_length']) + '_' + \
                str(cfg['n_grams_max_length']) + '_' + \
                str(cfg['n_grams_frequency_threshold']) + '_' + \
                str(cfg['n_grams_most_common']) + ".npy"

    file_path = os.path.join(os.path.dirname(__file__), "output", "ngram_vocabs", file_name)
    return file_path


if __name__ == "__main__":
    find_best_config(run_pan_12)
    pass

    if cfg["run_test"]:
        run_test()

    if cfg["run_federalist"]:
        run_federalist()

    if cfg["run_pan_11"]:
        run_pan_11()

    if cfg["run_pan_12"]:
        run_pan_12()
