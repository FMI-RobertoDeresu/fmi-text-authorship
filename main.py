import os
import numpy as np
import tensorflow as tf
import ranking_mapper
from parsers import federalist_parser
import data_plot
import function_words

tf.logging.set_verbosity(tf.logging.ERROR)

federalist_papers_path = os.path.join(os.path.dirname(__file__), 'input\\The_federalist_papers.txt')
federalist_papers_path_authors = {
    "HAMILTON": 0,
    "MADISON": 1,
    # "JAY": 2
}


def svm_with_rank_kernel(dataset):
    # for index, dataset_value in enumerate(dataset.values):
    #     print('{} -- {} -- {}'.format(dataset_value[0], dataset_value[1], ranking_mapper.map_data(dataset_value[2])))

    # data processing
    train_dataset_values = list(filter(lambda x: x[1] in federalist_papers_path_authors.keys(), dataset.values))
    eval_dataset_values = list(filter(lambda x: x[1] == "HAMILTON OR MADISON", dataset.values))

    train_data = (list(map(lambda x: ranking_mapper.map_data(x[2]), train_dataset_values)),
                  list(map(lambda x: federalist_papers_path_authors[x[1]], train_dataset_values)))

    predict_data = (list(map(lambda x: ranking_mapper.map_data(x[2]), eval_dataset_values)),)

    # data_plot.plot(train_data)

    # train
    features, labels = (np.array(train_data[0]), np.array(train_data[1]))

    features_column_name = 'features'
    example_id_column_name = 'example_id'
    example_id = np.array(['%d' % i for i in range(len(features))])
    kernel_features_dimension = len(function_words.use_for_train)

    print("Train:")
    print("Kernel features dimension: {}".format(kernel_features_dimension))
    # for index, dataset_value in enumerate(train_dataset_values):
    #     print('{} -- {} -- {}'.format(dataset_value[0], dataset_value[1], labels[index]))

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={features_column_name: features, example_id_column_name: example_id},
        y=labels,
        num_epochs=None,
        shuffle=True)

    svm = tf.contrib.learn.SVM(
        example_id_column=example_id_column_name,
        feature_columns=[tf.contrib.layers.real_valued_column(
            column_name=features_column_name,
            dimension=kernel_features_dimension)],
        l2_regularization=0.1
    )

    svm.fit(input_fn=train_input_fn, steps=8000)

    # predict
    features = np.array(predict_data[0])
    example_id = np.array(['%d' % i for i in range(len(features))])

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={features_column_name: features, example_id_column_name: example_id},
        num_epochs=1,
        shuffle=False)

    predict = list(svm.predict(input_fn=test_input_fn))

    # data_plot.plot((features, list(map(lambda x: x["classes"], predict))))

    # print results
    print("Predict:")
    for index, dataset_value in enumerate(eval_dataset_values):
        print('{} -- {} -- {} -- {}'.format(
            dataset_value[0],
            dataset_value[1],
            predict[index]["classes"],
            predict[index]["logits"]))


def run_federalist():
    dataset = federalist_parser.parse(federalist_papers_path)
    for i in range(3):
        svm_with_rank_kernel(dataset)


if __name__ == "__main__":
    run_federalist()
