import nltk
from itertools import groupby
from operator import itemgetter
import collections
import re

ranking_distance_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
n_grams_tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-z\,]')


def ranking_distance(input, stop_words):
    # function word frequencies
    input = input.lower()
    words = ranking_distance_tokenizer.tokenize(input)

    function_words_freq = []
    for i, function_word in enumerate(stop_words):
        function_words_freq.append((i, words.count(function_word)))

    function_words_freq.sort(key=itemgetter(1), reverse=True)

    # 'if k objects will claim the same rank and the first x ranks are already used by other objects,
    # then they will share the ranks x + 1, x + 2, . . . , x + k and all of them will receive as rank
    # the number: (x+1)+(x+2)+...+(x+k) / k = x + (k+1) / 2'
    groups = list(map(lambda x: (x[0], list(x[1])), groupby(function_words_freq, key=itemgetter(1))))
    ranking = []
    rank = 0
    for key, values in groups:
        group_rank = rank + (len(values) + 1) / 2
        rank += len(values)
        for value in values:
            ranking.append((value[0], group_rank))

    ranking.sort(key=itemgetter(0))
    ranking = list(map(itemgetter(1), ranking))

    return ranking


def n_grams_keys(inputs, n_min, n_max, n_most_common):
    n_grams_keys_list = []
    for input in inputs:
        chars = n_grams_tokenizer.tokenize(input.lower())
        for n in range(n_min, n_max+1):
            n_grams_list = list(map(lambda x: ''.join(x), nltk.ngrams(chars, n)))
            n_grams_keys_list = n_grams_keys_list + n_grams_list

    n_grams_counter = collections.Counter(n_grams_keys_list)
    n_grams_keys_list = list(map(lambda x: x[0], n_grams_counter.most_common(n_most_common)))
    # n_grams_keys_list = list(filter(lambda x: re.search("[\.\,]", x) is not None, n_grams_keys_list))

    return n_grams_keys_list


def n_grams(input, n_grams_keys_list, n_min, n_max):
    chars = n_grams_tokenizer.tokenize(input.lower())

    n_grams_list = []
    for n in range(n_min, n_max + 1):
        n_grams_list = n_grams_list + list(map(lambda x: ''.join(x), nltk.ngrams(chars, n)))

    n_grams_counter = collections.Counter(n_grams_list)
    n_grams_values = [n_grams_counter[ngram_key] for ngram_key in n_grams_keys_list]

    return n_grams_values
