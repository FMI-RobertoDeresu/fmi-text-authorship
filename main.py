import numpy as np
import tensorflow as tf
import time

tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('train_steps', 2000, 'Train steps.')


def get_input_fns(train_data, eval_data):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"features": train_data[0].reshape(train_data[0].shape[0], 784, ).astype(np.int32)},
        train_data[1].astype(np.int32),
        batch_size=256,
        num_epochs=None,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"features": eval_data[0].reshape(eval_data[0].shape[0], 784, ).astype(np.int32)},
        eval_data[1].astype(np.int32),
        batch_size=eval_data[0].shape[0],
        num_epochs=1,
        shuffle=False)

    return train_input_fn, eval_input_fn


def run_simple_model(train_data, eval_data):
    train_input_fn, eval_input_fn = get_input_fns(train_data, eval_data)

    # Specify the feature(s) to be used by the estimator.
    image_column = tf.contrib.layers.real_valued_column('features', dimension=784)
    optimizer = tf.train.FtrlOptimizer(learning_rate=5.0, l2_regularization_strength=1.0)
    estimator = tf.estimator.LinearClassifier(feature_columns=[image_column], n_classes=10, optimizer=optimizer)

    # Train.
    start = time.time()
    estimator.train(input_fn=train_input_fn, steps=FLAGS['train_steps'].value)
    end = time.time()
    print('Elapsed time: {} seconds'.format(end - start))

    # Evaluate and report metrics.
    eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
    print(eval_metrics)


def run_kernel_model(train_data, eval_data):
    train_input_fn, eval_input_fn = get_input_fns(train_data, eval_data)

    # Specify the feature(s) to be used by the estimator. This is identical to the
    # code used for the LinearClassifier.
    image_column = tf.contrib.layers.real_valued_column('features', dimension=784)
    optimizer = tf.train.FtrlOptimizer(learning_rate=50.0, l2_regularization_strength=0.001)

    kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
        input_dim=784, output_dim=2000, stddev=5.0, name='rffm')
    kernel_mappers = {image_column: [kernel_mapper]}
    estimator = tf.contrib.kernel_methods.KernelLinearClassifier(
        n_classes=10, optimizer=optimizer, kernel_mappers=kernel_mappers)

    # Train.
    start = time.time()
    estimator.fit(input_fn=train_input_fn, steps=FLAGS['train_steps'].value)
    end = time.time()
    print('Elapsed time: {} seconds'.format(end - start))

    # Evaluate and report metrics.
    eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
    print(eval_metrics)


def run():
    mnist_train, mnist_eval = tf.keras.datasets.mnist.load_data()
    run_simple_model(mnist_train, mnist_eval)
    # run_kernel_model(mnist_train, mnist_eval)


if __name__ == "__main__":
    run()


# ----------------------------------------------------------------------
def get_iters():
    mnist_train, mnist_validate = tf.keras.datasets.mnist.load_data()

    features_ph = tf.placeholder(tf.uint8, shape=[None, 28, 28])
    labels_ph = tf.placeholder(tf.uint8, shape=[None])

    # datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(features_ph, labels_ph).batch(FLAGS['batch_size'].value).repeat()
    validate_dataset = tf.data.Dataset.from_tensor_slices(features_ph, labels_ph)

    train_iter = train_dataset.make_initializable_iterator()
    validate_iter = validate_dataset.make_initializable_iterator()

    with tf.Session() as sess:
        sess.run(train_iter.initializer, feed_dict={features_ph: mnist_train[0], labels_ph: mnist_train[1]})
        sess.run(validate_iter.initializer, feed_dict={features_ph: mnist_validate[0], labels_ph: mnist_validate[0]})

    return train_iter, validate_iter
