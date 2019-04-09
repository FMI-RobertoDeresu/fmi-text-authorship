from ranking_mapper import map_data
from federalist_parser import parse_federalist_papers_file

def run():
    dataset = parse_federalist_papers_file('input/The_federalist_papers.txt')
    dataset.head()

    # mnist_train, mnist_eval = tf.keras.datasets.mnist.load_data()
    #
    # if FLAGS['train_method_flag'].value & 1:
    #     run_simple_model(mnist_train, mnist_eval)
    #
    # if FLAGS['train_method_flag'].value & 2:
    #     run_kernel_model(mnist_train, mnist_eval)


if __name__ == "__main__":
    run()