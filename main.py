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
    "run_federalist": 0,
    "run_pan_11": 1,
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

    # "ngrams_range": (3, 4), #pan
    "ngrams_range": (4, 4),  # federalist
    "ngrams_frequency_threshold": 100,

    "ranking_distance_kernel": kernels.linear,
    "ngrams_kernel": kernels.linear,

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


def run(train_dataset_values, train_labels, eval_dataset_values, eval_labels, stopwords, ngrams_vocab_file=None):
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
        ngrams_range = cfg['ngrams_range']
        freq_th = cfg['ngrams_frequency_threshold']

        ngrams_vocabulary = feature_extraction.ngrams_vocabulary(inputs, ngrams_range, freq_th, ngrams_vocab_file)
        print("NGrams vocabulary length " + str(len(ngrams_vocabulary)))

        train_texts = list(map(operator.itemgetter(2), train_dataset_values))
        train_texts_ngrams = feature_extraction.ngrams(train_texts, ngrams_vocabulary, ngrams_range)
        train_features = max_abs_scaler.fit_transform(np.array(train_texts_ngrams))
        train_data = (np.array(train_features), np.array(train_labels))

        eval_texts = list(map(operator.itemgetter(2), eval_dataset_values))
        eval_texts_ngrams = feature_extraction.ngrams(eval_texts, ngrams_vocabulary, ngrams_range)
        eval_features = max_abs_scaler.fit_transform(np.array(eval_texts_ngrams))
        eval_data = (np.array(eval_features), np.array(eval_labels))

        if cfg['use_svm_and_ngrams'] > 0:
            print("\nSVM with ngrams:")
            for i in range(cfg['use_svm_and_ngrams']):
                predict, predict_probs, predict_score = run_svm(train_data, eval_data, cfg['ngrams_kernel'])
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
    return

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

    ngrams_vocab_file = get_ngrams_vocab_file_path("federalist_papers")

    run(train_dataset_values, train_labels, eval_dataset_values, eval_labels, function_words.mosteller_and_wallace,
        ngrams_vocab_file)


def run_pan_11():
    print("\nRun pan 11")
    datasets = parsers.parse_pan_datasets(cfg["pan_11_path"])
    total_scores = np.zeros(4)

    for dataset_name, dataset in list(datasets.items())[1:]:
        print("\nDataset " + dataset_name)

        train_dataset_values = list(filter(lambda x: not (x[1].startswith('unknown')), dataset.values))
        train_labels = list(map(lambda x: int(re.search("\d+", x[1]).group()), train_dataset_values))

        eval_dataset_values = list(filter(lambda x: x[1].startswith('unknown'), dataset.values))
        eval_labels = list(map(lambda x: int(re.search("\d+", x[3]).group()), eval_dataset_values))

        ngrams_vocab_file = get_ngrams_vocab_file_path("pan_11_" + dataset_name)

        scores = run(train_dataset_values, train_labels, eval_dataset_values, eval_labels, function_words.nltk_english,
                     ngrams_vocab_file)
        total_scores = total_scores + scores

    total_scores /= len(datasets)
    print_final_scores(total_scores)


def run_pan_12():
    print("\nRun pan 12")
    datasets = parsers.parse_pan_datasets(cfg["pan_12_path"])
    total_scores = np.zeros(4)

    for dataset_name, dataset in datasets.items():
        print("\nDataset " + dataset_name)

        train_dataset_values = list(filter(lambda x: not (x[1].startswith('unknown')), dataset.values))
        train_labels = list(map(lambda x: int(re.search("\d+", x[1]).group()), train_dataset_values))

        eval_dataset_values = list(filter(lambda x: x[1].startswith('unknown'), dataset.values))
        eval_labels = list(map(lambda x: int(re.search("\d+", x[3]).group()), eval_dataset_values))

        ngrams_vocab_file = get_ngrams_vocab_file_path("pan_12_" + dataset_name)

        scores = run(train_dataset_values, train_labels, eval_dataset_values, eval_labels, function_words.nltk_english,
                     ngrams_vocab_file)
        total_scores = total_scores + scores

    total_scores /= len(datasets)
    print_final_scores(total_scores)


def find_best_config(method):
    for ngrams_min in [3, 4, 5]:
        for ngrams_max in range(ngrams_min, 6):
            for freq_th in [50, 100, 200, 500]:
                cfg['ngrams_range'] = (ngrams_min, ngrams_max)
                cfg['ngrams_frequency_threshold'] = freq_th

                print("\n\n" + str(ngrams_min) + " -- " + str(ngrams_max) + " -- " + str(freq_th))

                try:
                    method()
                except:
                    print("ERROR!!")


def get_ngrams_vocab_file_path(file_prefix):
    file_name = file_prefix + "_ngrams_vocab_" + \
                str(cfg['ngrams_range'][0]) + '_' + \
                str(cfg['ngrams_range'][1]) + '_' + \
                str(cfg['ngrams_frequency_threshold']) + ".npy"

    file_path = os.path.join(os.path.dirname(__file__), "output", "ngram_vocabs", file_name)
    return file_path


if __name__ == "__main__":
    find_best_config(run_pan_11)
    pass

    if cfg["run_test"]:
        run_test()

    if cfg["run_federalist"]:
        run_federalist()

    if cfg["run_pan_11"]:
        run_pan_11()

    if cfg["run_pan_12"]:
        run_pan_12()
