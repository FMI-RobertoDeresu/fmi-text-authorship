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

kernels_list = [
    ("linear", kernels.linear),
    ("intersection", kernels.intersection),
    # ("binary", kernels.binary)
]

cfg = {
    "run_test": 0,
    "run_federalist": 1,
    "run_pan_11": 0,
    "run_pan_12": 0,
    "run_all": 1,

    # === COMMON ===
    "use_svm_and_ranking_distance": 1,
    "use_svm_and_ngrams": 0,

    "string_kernel": kernels_list[0],  # linear
    # "string_kernel": kernels_list[1],  # intersection
    # "string_kernel": kernels_list[2],  # binary

    # "print_each_test": True,
    "print_each_test": False,
    # "print_probs": True,
    "print_probs": False,

    # === TEST ===
    "test_function_words": ("romanian", function_words.romanian),
    "test_ngrams_range": (4, 4),
    "test_ngrams_frequency_threshold": 10,

    # === FEDERALIST PAPERS ===
    "federalist_papers_path": os.path.join(os.path.dirname(__file__), 'input\\federalist_papers.txt'),
    "federalist_papers_authors": {"HAMILTON": 0, "MADISON": 1, "JAY": 2},

    "federalist_papers_ranking_function_words": ("nltk_english", function_words.nltk_english),
    "federalist_papers_ngrams_range": (4, 4),
    "federalist_papers_ngrams_frequency_threshold": 100,

    # === PAN 11 ===
    "pan_11_path": os.path.join(os.path.dirname(__file__), 'input\\pan11'),

    "pan_11_ranking_function_words": ("nltk_english", function_words.nltk_english),
    "pan_11_ngrams_range": (3, 4),
    "pan_11_ngrams_frequency_threshold": 100,

    # === PAN 12 ===
    "pan_12_path": os.path.join(os.path.dirname(__file__), 'input\\pan12'),

    "pan_12_ranking_function_words": ("nltk_english", function_words.nltk_english),
    "pan_12_ngrams_range": (3, 4),
    "pan_12_ngrams_frequency_threshold": 100,
}


def run_svm(train_data, eval_data, kernel):
    train_features, train_labels = (train_data[0], train_data[1])
    eval_features, eval_labels = (eval_data[0], eval_data[1])

    # train
    svm = sklearn.svm.SVC(kernel=kernel, probability=True, verbose=0)
    clf = multiclass.OneVsRestClassifier(svm).fit(train_features, train_labels)

    # eval
    predict = clf.predict(eval_features)
    predict_probs = clf.predict_proba(eval_features)
    predict_score = clf.score(eval_features, eval_labels)

    return predict, predict_probs, predict_score


def run(train_dataset_values, train_labels, eval_dataset_values, eval_labels, ranking_distance_function_words,
        ngrams_range, ngrams_freq_th, string_kernel, ranking_distance_file=None, ngrams_vocab_file=None):
    max_abs_scaler = sklearn.preprocessing.MaxAbsScaler()
    scores = np.zeros(2)

    # ranking distance
    if cfg["use_svm_and_ranking_distance"]:
        print("\nSVM with rank distance:")

        texts = list(map(operator.itemgetter(2), train_dataset_values + eval_dataset_values))
        rankings = feature_extraction.ranking_distance(texts, ranking_distance_function_words, ranking_distance_file)
        # train_features = max_abs_scaler.fit_transform(np.array(rankings[:len(train_dataset_values)]))
        train_features = np.array(rankings[:len(train_dataset_values)])
        train_data = (np.array(train_features), np.array(train_labels))

        # eval_features = max_abs_scaler.fit_transform(np.array(rankings[len(train_dataset_values):]))
        eval_features = np.array(rankings[len(train_dataset_values):])
        eval_data = (np.array(eval_features), np.array(eval_labels))

        predict, predict_probs, predict_score = run_svm(train_data, eval_data, string_kernel)
        scores[0] = predict_score
        print_results(eval_dataset_values, eval_labels, predict, predict_probs, predict_score)

    # n-grams
    if cfg["use_svm_and_ngrams"]:
        print("\nSVM with ngrams:")

        texts = list(map(operator.itemgetter(2), train_dataset_values))
        ngrams_vocabulary = feature_extraction.ngrams_vocabulary(texts, ngrams_range, ngrams_freq_th, ngrams_vocab_file)
        print("NGrams vocabulary length " + str(len(ngrams_vocabulary)))

        train_texts = list(map(operator.itemgetter(2), train_dataset_values))
        train_texts_ngrams = feature_extraction.ngrams(train_texts, ngrams_vocabulary, ngrams_range)
        train_features = max_abs_scaler.fit_transform(np.array(train_texts_ngrams))
        train_data = (np.array(train_features), np.array(train_labels))

        eval_texts = list(map(operator.itemgetter(2), eval_dataset_values))
        eval_texts_ngrams = feature_extraction.ngrams(eval_texts, ngrams_vocabulary, ngrams_range)
        eval_features = max_abs_scaler.fit_transform(np.array(eval_texts_ngrams))
        eval_data = (np.array(eval_features), np.array(eval_labels))

        predict, predict_probs, predict_score = run_svm(train_data, eval_data, string_kernel)
        scores[1] = predict_score
        print_results(eval_dataset_values, eval_labels, predict, predict_probs, predict_score)

    return scores


def run_test():
    print("\n === Run test === ")
    dataset = parsers.parse_test()

    train_dataset_values = list(dataset.values[:10])
    train_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    eval_dataset_values = list(dataset.values[10:])
    eval_labels = [0, 0, 1, 1]

    func_words = cfg["test_function_words"]
    ngrams_range = cfg["test_ngrams_range"]
    ngrams_freq_th = cfg["test_ngrams_frequency_threshold"]
    string_kernel = cfg["string_kernel"]
    ranking_distance_file = get_ranking_distance_file_path("test", func_words[0])
    ngrams_vocab_file = get_ngrams_vocab_file_path("test", ngrams_range, ngrams_freq_th)

    scores = run(train_dataset_values,
                 train_labels,
                 eval_dataset_values,
                 eval_labels,
                 func_words[1],
                 ngrams_range,
                 ngrams_freq_th,
                 string_kernel[1],
                 ranking_distance_file,
                 ngrams_vocab_file)

    print_final_scores(scores, cfg["string_kernel"][0])
    return scores


def run_federalist():
    print("\n === Run federalist papers === ")
    authors = cfg["federalist_papers_authors"]
    dataset = parsers.parse_federalist_papers(cfg["federalist_papers_path"])

    train_dataset_values = list(filter(lambda x: x[1] in list(authors.keys())[:2], dataset.values))
    train_labels = list(map(lambda x: authors[x[1]], train_dataset_values))

    eval_dataset_values = list(filter(lambda x: x[1] == "HAMILTON OR MADISON", dataset.values))
    eval_labels = [1] * 11

    func_words = cfg["federalist_papers_ranking_function_words"]
    ngrams_range = cfg["federalist_papers_ngrams_range"]
    ngrams_freq_th = cfg["federalist_papers_ngrams_frequency_threshold"]
    string_kernel = cfg["string_kernel"]
    ranking_distance_file = get_ranking_distance_file_path("federalist_papers", func_words[0])
    ngrams_vocab_file = get_ngrams_vocab_file_path("federalist_papers", ngrams_range, ngrams_freq_th)

    scores = run(train_dataset_values,
                 train_labels,
                 eval_dataset_values,
                 eval_labels,
                 func_words[1],
                 ngrams_range,
                 ngrams_freq_th,
                 string_kernel[1],
                 ranking_distance_file,
                 ngrams_vocab_file)

    print_final_scores(scores, cfg["string_kernel"][0])
    return scores


def run_pan_11():
    print("\n === Run pan 11 === ")
    datasets = parsers.parse_pan_datasets(cfg["pan_11_path"])
    total_scores = np.zeros(2)

    for dataset_name, dataset in datasets.items():
        print("\nDataset " + dataset_name)

        train_dataset_values = list(filter(lambda x: not (x[1].startswith('unknown')), dataset.values))
        train_labels = list(map(lambda x: int(re.search("\d+", x[1]).group()), train_dataset_values))

        eval_dataset_values = list(filter(lambda x: x[1].startswith('unknown'), dataset.values))
        eval_labels = list(map(lambda x: int(re.search("\d+", x[3]).group()), eval_dataset_values))

        func_words = cfg["pan_11_ranking_function_words"]
        ngrams_range = cfg["pan_11_ngrams_range"]
        ngrams_freq_th = cfg["pan_11_ngrams_frequency_threshold"]
        string_kernel = cfg["string_kernel"]
        ranking_distance_file = get_ranking_distance_file_path("pan_11_" + dataset_name, func_words[0])
        ngrams_vocab_file = get_ngrams_vocab_file_path("pan_11_" + dataset_name, ngrams_range, ngrams_freq_th)

        scores = run(train_dataset_values,
                     train_labels,
                     eval_dataset_values,
                     eval_labels,
                     func_words[1],
                     ngrams_range,
                     ngrams_freq_th,
                     string_kernel[1],
                     ranking_distance_file,
                     ngrams_vocab_file)

        total_scores = total_scores + scores

    total_scores /= len(datasets)
    print_final_scores(total_scores, cfg["string_kernel"][0])
    return total_scores


def run_pan_12():
    print("\n === Run pan 12 === ")
    datasets = parsers.parse_pan_datasets(cfg["pan_12_path"])
    total_scores = np.zeros(2)

    for dataset_name, dataset in datasets.items():
        print("\nDataset " + dataset_name)

        train_dataset_values = list(filter(lambda x: not (x[1].startswith('unknown')), dataset.values))
        train_labels = list(map(lambda x: int(re.search("\d+", x[1]).group()), train_dataset_values))

        eval_dataset_values = list(filter(lambda x: x[1].startswith('unknown'), dataset.values))
        eval_labels = list(map(lambda x: int(re.search("\d+", x[3]).group()), eval_dataset_values))

        func_words = cfg["pan_12_ranking_function_words"]
        ngrams_range = cfg["pan_12_ngrams_range"]
        ngrams_freq_th = cfg["pan_12_ngrams_frequency_threshold"]
        string_kernel = cfg["string_kernel"]
        ranking_distance_file = get_ranking_distance_file_path("pan_12_" + dataset_name, func_words[0])
        ngrams_vocab_file = get_ngrams_vocab_file_path("pan_12_" + dataset_name, ngrams_range, ngrams_freq_th)

        scores = run(train_dataset_values,
                     train_labels,
                     eval_dataset_values,
                     eval_labels,
                     func_words[1],
                     ngrams_range,
                     ngrams_freq_th,
                     string_kernel[1],
                     ranking_distance_file,
                     ngrams_vocab_file)

        total_scores = total_scores + scores

    total_scores /= len(datasets)
    print_final_scores(total_scores, cfg["string_kernel"][0])
    return total_scores


def print_results(eval_dataset_values, eval_labels, predict, predict_probs, predict_score):
    print("Score: {}".format(predict_score))

    if cfg["print_each_test"]:
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


def print_final_scores(scores, kernel_name):
    print("\nTotal scores:")
    print("String kernel: " + kernel_name)

    if cfg["use_svm_and_ranking_distance"]:
        print("SVM with rank distance: " + str(scores[0]))

    if cfg["use_svm_and_ngrams"]:
        print("SVM with ngrams: " + str(scores[1]))


def get_ranking_distance_file_path(file_prefix, function_words_name):
    file_name = file_prefix + "_ranking_distance_" + \
                str(function_words_name) + ".npy"

    file_path = os.path.join(os.path.dirname(__file__), "output", "ranking_distances", file_name)
    return file_path


def get_ngrams_vocab_file_path(file_prefix, ngrams_range, ngrams_freq_th):
    file_name = file_prefix + "_ngrams_vocab_" + \
                str(ngrams_range[0]) + '_' + \
                str(ngrams_range[1]) + '_' + \
                str(ngrams_freq_th) + ".npy"

    file_path = os.path.join(os.path.dirname(__file__), "output", "ngram_vocabs", file_name)
    return file_path


def find_best_ngrams_config(method):
    for ngrams_min in [3, 4, 5]:
        for ngrams_max in range(ngrams_min, 6):
            for freq_th in [10, 50, 100, 150]:
                cfg["ngrams_range"] = (ngrams_min, ngrams_max)
                cfg["ngrams_frequency_threshold"] = freq_th

                print("\n\n" + str(ngrams_min) + " -- " + str(ngrams_max) + " -- " + str(freq_th))

                try:
                    method()
                except:
                    print("ERROR!!")


if __name__ == "__main__":
    # find_best_ngrams_config(run_federalist)

    if cfg["run_all"]:
        cfg["use_svm_and_ranking_distance"] = 1
        cfg["use_svm_and_ngrams"] = 1

        for kernel in kernels_list:
            cfg["string_kernel"] = kernel
            # run_test()
            run_federalist()
            # run_pan_11()
            run_pan_12()
    else:
        if cfg["run_test"]:
            run_test()

        if cfg["run_federalist"]:
            run_federalist()

        if cfg["run_pan_11"]:
            run_pan_11()

        if cfg["run_pan_12"]:
            run_pan_12()
