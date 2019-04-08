import numpy as np
import tensorflow as tf
import time

tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('train_method_flag', 2, 'Train method.')
tf.flags.DEFINE_integer('train_steps', 2000, 'Train steps.')
tf.flags.DEFINE_float('train_kernel_learning_rate', 0.1, 'RandomFourierFeatureMapper output_dim.')
tf.flags.DEFINE_integer('train_kernel_stddev', 5, 'RandomFourierFeatureMapper stddev.')
tf.flags.DEFINE_integer('train_kernel_output_dim', 2000, 'RandomFourierFeatureMapper output_dim.')


def get_input_fns(train_data, eval_data):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"features": train_data[0].reshape(train_data[0].shape[0], 784, ).astype(np.float32)},
        train_data[1].astype(np.int32),
        batch_size=256,
        num_epochs=None,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"features": eval_data[0].reshape(eval_data[0].shape[0], 784, ).astype(np.float32)},
        eval_data[1].astype(np.int32),
        batch_size=eval_data[0].shape[0],
        num_epochs=1,
        shuffle=False)

    return train_input_fn, eval_input_fn


def run_simple_model(train_data, eval_data):
    train_input_fn, eval_input_fn = get_input_fns(train_data, eval_data)

    # Specify the feature(s) to be used by the estimator.
    image_column = tf.contrib.layers.real_valued_column('features', dimension=784)
    optimizer = tf.train.FtrlOptimizer(learning_rate=0.01)
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

        image_column = tf.contrib.layers.real_valued_column('features', dimension=784)
        optimizer = tf.train.FtrlOptimizer(learning_rate=FLAGS['train_kernel_learning_rate'].value)

        kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
            input_dim=784,
            output_dim=FLAGS['train_kernel_output_dim'].value,
            stddev=FLAGS['train_kernel_stddev'].value,
            name='rffm')
        kernel_mappers = {image_column: [kernel_mapper]}

        estimator = tf.contrib.kernel_methods.KernelLinearClassifier(
            n_classes=10, optimizer=optimizer, kernel_mappers=kernel_mappers)

        # Train.
        start = time.time()
        estimator.fit(input_fn=train_input_fn, steps=FLAGS['train_steps'].value)
        end = time.time()
        print('Elapsed time: {} seconds (learning_rate={}) (stddev={}, output_dim={})'.format(
            end - start,
            FLAGS['train_kernel_learning_rate'].value,
            FLAGS['train_kernel_stddev'].value,
            FLAGS['train_kernel_output_dim'].value))

        # Evaluate and report metrics.
        eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
        print(eval_metrics)


def run():
    mnist_train, mnist_eval = tf.keras.datasets.mnist.load_data()

    if FLAGS['train_method_flag'].value & 1:
        run_simple_model(mnist_train, mnist_eval)

    if FLAGS['train_method_flag'].value & 2:
        run_kernel_model(mnist_train, mnist_eval)


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
