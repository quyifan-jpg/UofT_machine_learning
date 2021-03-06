from utils import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import numpy as np


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    u_i = u[n]
    z_i = z[q]
    # apply GSD with no regularization parameter
    error = c - np.dot(u_i, z_i)
    u[n] += lr * error * z_i
    z[q] += lr * error * u_i
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration, val_data):
    """ Performs ALS algorithm, here we use the iterative solution - SGD 
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # print(f"shape u:{u.shape}, shape z:{z.shape}")
    train_losses = []
    val_losses = []
    for iteration in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        # if iteration % 2000 == 0:
        #     train_loss = squared_error_loss(train_data, u, z)
        #     val_loss = squared_error_loss(val_data, u, z)
        #     train_losses.append(train_loss)
        #     val_losses.append(val_loss)
    mat = np.dot(u, z.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, train_losses, val_losses


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    k_values = [3, 5, 7, 9, 13, 20]
    highest_k = 0
    highest_acc = 0
    acc_list = []
    for k in k_values:
        reconst_matrix = svd_reconstruct(train_matrix, k)
        val_acc = sparse_matrix_evaluate(val_data, reconst_matrix)
        acc_list.append(val_acc)
        if val_acc > highest_acc:
            highest_k = k
            highest_acc = val_acc
        print(f"validation accuracy for k = {k}: {val_acc}")

    reconst_matrix = svd_reconstruct(train_matrix, highest_k)
    val_acc = sparse_matrix_evaluate(val_data, reconst_matrix)
    test_acc = sparse_matrix_evaluate(test_data, reconst_matrix)
    print(f"chosen k is {highest_k}, has validation accuracy {val_acc} and test accuracy {test_acc}")
    plt.title("SVD validation accuracy")
    plt.xlabel("k value")
    plt.ylabel("validation accuracy")
    plt.plot(k_values, acc_list)
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    k_values = [7, 9, 15, 25, 50, 100]
    lr = 0.05
    iterations = 100000
    final_matrix = None
    final_train_losses = []
    final_val_losses = []
    highest_k2 = 0
    highest_acc2 = 0
    for k in k_values:
        reconst_matrix, train_losses, val_losses = als(train_data, k, lr, iterations, val_data)
        val_acc = sparse_matrix_evaluate(val_data, reconst_matrix)
        if val_acc > highest_acc2:
            highest_acc2 = val_acc
            highest_k2 = k
        print(f"validation accuracy for k = {k}: {val_acc}")

    final_matrix, final_train_losses, final_val_losses = als(train_data, highest_k2, lr, iterations, val_data)
    test_acc = sparse_matrix_evaluate(test_data, final_matrix)
    print(f"chosen k is {highest_k2}, has validation accuracy {highest_acc2} and test accuracy {test_acc}")

    # plt.ylabel("squared error loss")
    # plt.xlabel("iterations")
    # x_axis = np.arange(0, 100000, 2000)
    # plt.plot(x_axis, final_val_losses, label="validation error")
    # plt.plot(x_axis, final_train_losses, label="training error")
    # plt.title("square error to iteration")
    # plt.legend()
    # plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
