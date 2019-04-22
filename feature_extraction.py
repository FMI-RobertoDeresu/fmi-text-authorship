import os
import numpy as np
import nltk
from itertools import groupby
import operator
import collections
import sklearn
import time

ranking_distance_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
n_grams_tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-z\,]')


def ranking_distance(input, stop_words):
    # function word frequencies
    input = input.lower()
    words = ranking_distance_tokenizer.tokenize(input)

    function_words_freq = []
    for i, function_word in enumerate(stop_words):
        function_words_freq.append((i, words.count(function_word)))

    function_words_freq.sort(key=operator.itemgetter(1), reverse=True)

    # 'if k objects will claim the same rank and the first x ranks are already used by other objects,
    # then they will share the ranks x + 1, x + 2, . . . , x + k and all of them will receive as rank
    # the number: (x+1)+(x+2)+...+(x+k) / k = x + (k+1) / 2'
    groups = list(map(lambda x: (x[0], list(x[1])), groupby(function_words_freq, key=operator.itemgetter(1))))
    ranking = []
    rank = 0
    for key, values in groups:
        group_rank = rank + (len(values) + 1) / 2
        rank += len(values)
        for value in values:
            ranking.append((value[0], group_rank))

    ranking.sort(key=operator.itemgetter(0))
    ranking = list(map(operator.itemgetter(1), ranking))

    return ranking


def n_grams_vocabulary(inputs, n_min, n_max, most_common, frequency_threshold, file_path=None):
    if file_path is not None and os.path.exists(file_path):
        n_grams_vocab = list(np.load(file_path))
        return n_grams_vocab

    t = time.time()
    n_grams_vocab = []
    for input in inputs:
        chars = n_grams_tokenizer.tokenize(input.lower())
        for n in range(n_min, n_max + 1):
            n_grams_list = list(map(lambda x: ''.join(x), nltk.ngrams(chars, n)))
            n_grams_vocab = n_grams_vocab + n_grams_list

    n_grams_counter = collections.Counter(n_grams_vocab)
    n_grams_most_common = n_grams_counter.most_common(most_common)
    n_grams_above_threshold = list(filter(lambda x: x[1] >= frequency_threshold, n_grams_most_common))
    n_grams_vocab = list(map(operator.itemgetter(0), n_grams_above_threshold))
    print("NGrams Vocabulary (time): " + str(time.time() - t))

    if file_path is not None:
        np.save(file_path, np.array(n_grams_vocab))

    return n_grams_vocab


def n_grams(texts, n_grams_vocabulary, n_min, n_max):
    t = time.time()

    vectorizer = sklearn.feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(n_min, n_max),
                                                                 lowercase=False, vocabulary=n_grams_vocabulary)
    n_grams_values = vectorizer.fit_transform(texts)
    n_grams_values = n_grams_values.astype(float).toarray()

    for index, text in enumerate(texts):
        n_grams_values[index] = n_grams_values[index] / len(text)

    print("Vectorizer (time): " + str(time.time() - t))
    return n_grams_values
