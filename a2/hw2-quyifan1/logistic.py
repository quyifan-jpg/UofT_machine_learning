from q3.utils import sigmoid

import numpy as np
import math


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """

    #####################################################################
    # TODO:                                                             #
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################
    bias = np.ones((len(data), 1), dtype=int)
    data_bias = np.append(data, bias, axis=1)
    y = np.dot(data_bias, weights)
    y = sigmoid(y)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO:                                                             #
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################
    ce = None
    frac_correct = None
    result = 0
    result1= 0
    normalizedY = []
    for i in range(len(targets)):
        result -= (targets[i]*np.log(y[i])+(1-targets[i])*np.log(1-y[i]))
        normalizedY.append([1] if y[i] > 0.5 else [0])
    Y = np.array(normalizedY)
    frac_correct = np.sum(Y == targets) / len(targets)
    ce = result
    return ce, frac_correct
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################



def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of parameters, and the probabilities given by    #
    # logistic regression.                                              #
    #####################################################################
    ce, frac_correct = evaluate(targets, y)

    f = None
    df = None
    f = ce
    bias = np.ones((len(data), 1), dtype=int)
    data_bias = np.append(data, bias, axis=1)
    df = np.dot(data_bias.T, y-targets)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y


if __name__ == "__main__":
    pass
