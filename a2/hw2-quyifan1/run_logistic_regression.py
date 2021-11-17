from q3.check_grad import check_grad
from q3.utils import *
from q3.logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_valid()


    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.2,
        "weight_regularization": N,
        "num_iterations": 300
    }
    weights = np.random.randn(M + 1, 1)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    f, df, prob = logistic(weights, train_inputs, train_targets, hyperparameters)
    entropy_test, erro_test = evaluate(train_targets, prob)
    ce_train=[]
    ce_valid=[]
    rate_train=[]
    rate_valid=[]
    tList = np.arange(0, hyperparameters['num_iterations'], 1)
    for t in range(hyperparameters["num_iterations"]):
        f, df, prob = logistic(weights, train_inputs, train_targets, hyperparameters)
        # f is loss func, df derivative, y is problility
        weights = weights - hyperparameters["learning_rate"]*df / N
        predictions_valid = logistic_predict(weights, valid_inputs)
        predictions_train = logistic_predict(weights, train_inputs)
        entropy_train, error_train = evaluate(train_targets, predictions_train)
        entropy_valid, erro_valid = evaluate(valid_targets, predictions_valid)

        ce_train.append(entropy_train)
        ce_valid.append(entropy_valid)
        rate_train.append(1-error_train)
        rate_valid.append(1-erro_valid)

    f, df, prob = logistic(weights, train_inputs, train_targets, hyperparameters)
    fig, graph = plt.subplots()
    graph.plot(tList, ce_train, label="training")
    graph.plot(tList, ce_valid, label="valid")
    graph.set(xlabel='number of iterations', ylabel='Cross Entropy',
              title="Results")
    graph.grid()
    graph.legend()
    fig.savefig("q2_2_entropy.png")
    plt.show()

    fig, graph = plt.subplots()
    graph.plot(tList, rate_train, label="training")
    graph.plot(tList, rate_valid, label="valid")
    graph.set(xlabel='number of iterations', ylabel='classify error',
              title="Results")
    graph.grid()
    graph.legend()
    fig.savefig("q2_2_rate.png")
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################



def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
