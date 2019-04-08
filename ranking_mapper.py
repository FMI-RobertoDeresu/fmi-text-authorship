from function_words import function_words
from itertools import groupby
from operator import itemgetter


def map_data_list(data):
    mapped_data = list(map(lambda x: map_data(x), data))
    return mapped_data


def map_data(data_str):
    # function word frequencies
    data_words = data_str.lower().split()
    function_words_freq = []
    for i, function_word in enumerate(function_words):
        function_words_freq.append((i, data_words.count(function_word)))

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
