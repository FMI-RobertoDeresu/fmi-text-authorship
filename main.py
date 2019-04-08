from ranking_mapper import map_data

test = "it was the best of times it was the worst of times it was the age of wisdom it was the age of foolishness"
print (map_data(test))


def run():
    pass
    # mnist_train, mnist_eval = tf.keras.datasets.mnist.load_data()
    #
    # if FLAGS['train_method_flag'].value & 1:
    #     run_simple_model(mnist_train, mnist_eval)
    #
    # if FLAGS['train_method_flag'].value & 2:
    #     run_kernel_model(mnist_train, mnist_eval)


if __name__ == "__main__":
    run()