from utils import *
import numpy as np
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt


# from collections import Counter


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for i in range(len(data["is_correct"])):
        is_correct = data["is_correct"][i]
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        d = theta[user_id] - beta[question_id]
        prob = sigmoid(d)
        if is_correct == 1:
            log_lklihood += np.log(prob)
        if is_correct == 0:
            log_lklihood += np.log(1 - prob)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for t in range(len(data['user_id'])):
        i = data['user_id'][t]
        j = data['question_id'][t]
        c = data['is_correct'][t]
        neededSigmoid = sigmoid( theta[i] - beta[j])
        theta[i] -= -lr * (c - neededSigmoid)
        beta[j] -= -lr * (- c + neededSigmoid)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.rand(1 + max(data["user_id"]))
    beta = np.random.rand(1 + max(data["question_id"]))
    # alpha = np.random.rand(1 + max(data["question_id"]))
    val_acc_lst = []
    train_acc_lst = []
    train_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        # log likelihood
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_lld_lst.append(neg_lld)
        val_lld_lst.append(val_neg_lld)
        # score by evaluate
        train_score = evaluate(data=data, theta=theta, beta=beta)
        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(val_score)
        train_acc_lst.append(train_score)
        # print val
        print("NLLK: {} \t Score: {}".format(neg_lld, val_score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_acc_lst, val_lld_lst, train_lld_lst, train_acc_lst, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Train IRT
    lr = 0.005
    num_iteration = 30
    iter_lst = np.arange(num_iteration)
    theta, beta, train_neg_lld_lst, val_neg_lld_lst, train_lld_lst, val_lld_lst, train_acc_list, val_acc_lst = irt(train_data, val_data, lr, num_iteration)

    # Plot training and validation neg_log_likelihood as a function of iterations
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 12})
    plt.plot(iter_lst, train_acc_list, label="Train")
    plt.plot(iter_lst, val_acc_lst, label="Validation")
    plt.xlabel("Iteration")
    plt.ylabel("correctness")
    plt.legend()
    plt.show()

    # Plot training and validation log_likelihood as a function of iterations
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 12})
    plt.plot(iter_lst, train_lld_lst, label="Train")
    plt.plot(iter_lst, val_lld_lst, label="Validation")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.legend()
    plt.show()

    # Report the final validation and test accuracy
    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print("Validation Accuracy: ", val_acc)
    print("Test Accuracy: ", test_acc)

    # Select five questions (betas), plot the possibility of correctness
    # as a function of students (thetas)
    theta_lst = np.arange(-5, 5, 0.01)
    beta_lst = [4, 21, 127, 784, 1371]
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 12})
    for j in beta_lst:
        p = sigmoid(theta_lst - beta[j])
        plt.plot(theta_lst, p, label="Question {}".format(j))
    plt.xlabel("Students(theta)")
    plt.ylabel("Possibility of Correctness")
    plt.legend()
    plt.show()

    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
