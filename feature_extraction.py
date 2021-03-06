import os
import numpy as np
import nltk
import itertools
import operator
import collections
import sklearn
import time

ranking_distance_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
n_grams_tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-z\,]')


def ranking_distance(texts, stop_words, ranking_distance_file):
    if ranking_distance_file is not None and os.path.exists(ranking_distance_file):
        texts_rankings = list(np.load(ranking_distance_file))
        return texts_rankings

    texts_rankings = []
    for text in texts:
        # function word frequencies
        words = ranking_distance_tokenizer.tokenize(text.lower())

        function_words_freq = []
        for i, function_word in enumerate(stop_words):
            function_words_freq.append((i, words.count(function_word)))

        function_words_freq.sort(key=operator.itemgetter(1), reverse=True)

        # 'if k objects will claim the same rank and the first x ranks are already used by other objects,
        # then they will share the ranks x + 1, x + 2, . . . , x + k and all of them will receive as rank
        # the number: (x+1)+(x+2)+...+(x+k) / k = x + (k+1) / 2'
        groups = list(map(lambda x: (x[0], list(x[1])),
                          itertools.groupby(function_words_freq, key=operator.itemgetter(1))))

        ranking = []
        rank = 0
        for key, values in groups:
            group_rank = rank + (len(values) + 1) / 2
            rank += len(values)
            for value in values:
                ranking.append((value[0], group_rank))

        ranking.sort(key=operator.itemgetter(0))
        ranking = list(map(operator.itemgetter(1), ranking))

        texts_rankings.append(ranking)

    if ranking_distance_file is not None:
        np.save(ranking_distance_file, np.array(texts_rankings))

    return texts_rankings


def ngrams_vocabulary(texts, ngrams_range, frequency_threshold, file_path=None):
    if file_path is not None and os.path.exists(file_path):
        n_grams_vocab = list(np.load(file_path))
        return n_grams_vocab

    t = time.time()
    n_grams_vocab = []
    for text in texts:
        chars = n_grams_tokenizer.tokenize(text.lower())
        for n in range(ngrams_range[0], ngrams_range[1] + 1):
            n_grams_list = list(map(lambda x: ''.join(x), nltk.ngrams(chars, n)))
            n_grams_vocab = n_grams_vocab + n_grams_list

    n_grams_counter = collections.Counter(n_grams_vocab)
    n_grams_above_threshold = list(filter(lambda x: x[1] >= frequency_threshold, n_grams_counter.most_common()))
    n_grams_vocab = list(map(operator.itemgetter(0), n_grams_above_threshold))
    print("NGrams Vocabulary (time): " + str(time.time() - t))

    if file_path is not None:
        np.save(file_path, np.array(n_grams_vocab))

    return n_grams_vocab


def ngrams(texts, n_grams_vocabulary, ngrams_range):
    t = time.time()

    vectorizer = sklearn.feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=ngrams_range,
                                                                 lowercase=False, vocabulary=n_grams_vocabulary)
    n_grams_values = vectorizer.fit_transform(texts)
    n_grams_values = n_grams_values.astype(float).toarray()

    for index, text in enumerate(texts):
        n_grams_values[index] = n_grams_values[index] / len(text)

    print("Vectorizer (time): " + str(time.time() - t))
    return n_grams_values
