import os
import numpy as np
import tensorflow as tf
import ranking_mapper
from parsers import federalist_parser

tf.logging.set_verbosity(tf.logging.ERROR)

federalist_papers_path = os.path.join(os.path.dirname(__file__), 'input\\The_federalist_papers.txt')

federalist_papers_authors = {
    "HAMILTON": 0,
    "MADISON": 1
}


def train(train_data, predict_data):
    x, y = (np.array(train_data[0]), np.array(train_data[1]))

    features_column_name = 'features'
    example_id_column_name = 'example_id'
    example_id = np.array(['%d' % i for i in range(len(y))])

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={features_column_name: x, example_id_column_name: example_id},
        y=y,
        num_epochs=None,
        shuffle=True)

    svm = tf.contrib.learn.SVM(
        example_id_column=example_id_column_name,
        feature_columns=[tf.contrib.layers.real_valued_column(column_name=features_column_name, dimension=70)],
        l2_regularization=0.1
    )

    res = svm.fit(input_fn=train_input_fn, steps=200)
    print(res)


def run_federalist():
    dataset = federalist_parser.parse(federalist_papers_path)

    train_dataset_values = list(filter(lambda x: x[1] in federalist_papers_authors, dataset.values))
    eval_dataset_values = list(filter(lambda x: x[1] not in federalist_papers_authors, dataset.values))

    train_data = (list(map(lambda x: ranking_mapper.map_data(x[2]), train_dataset_values)),
                  list(map(lambda x: federalist_papers_authors[x[1]], train_dataset_values)))

    predict_data = list(map(lambda x: ranking_mapper.map_data(x[2]), eval_dataset_values))

    train(train_data, predict_data)


if __name__ == "__main__":
    run_federalist()
