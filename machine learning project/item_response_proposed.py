from utils import *
import numpy as np
import matplotlib.pyplot as plt


# from collections import Counter


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, alpha):
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



def update_theta_beta(data, lr, theta, beta, alpha):
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
    :param alpha: Vector
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
        neededSigmoid = sigmoid(alpha[j] * theta[i] - alpha[j] * beta[j])
        theta[i] -= -lr * (alpha[j] * c - alpha[j] * neededSigmoid)
        beta[j] -= -lr * (-alpha[j] * c + alpha[j] * neededSigmoid)
        alpha[j] -= -lr * (c * theta[i] - c * beta[j] - (
                theta[i] - beta[j]) * neededSigmoid)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, alpha


def irt_first(data, val_data, lr, iterations=20):
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
    theta = np.random.rand(542)
    beta = np.random.rand(1 + max(data["question_id"]))
    alpha = np.ones(1 + max(data["question_id"]))
    val_acc_lst = []
    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha)
        score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {} \t k: {}".format(neg_lld, score, i))
        theta, beta, alpha = update_theta_beta(data, lr, theta, beta, alpha)

    return theta


def split_data_gender(data):
    gender_0 = {
        "user_id": [], "question_id": [], "is_correct": []
    }
    gender_1 = {
        "user_id": [], "question_id": [], "is_correct": []
    }
    gender_2 = {
        "user_id": [], "question_id": [], "is_correct": []
    }
    student_meta = load_student_gender_meta("../data/student_meta.csv")
    uid_0, uid_1, uid_2 = [], [], []
    # split data according to their gender
    for i in range(len(student_meta["user_id"])):
        gender = student_meta["gender"][i]
        user_id = student_meta["user_id"][i]
        if gender == 0:
            uid_0.append(user_id)
        elif gender == 1:
            uid_1.append(user_id)
        else:
            uid_2.append(user_id)

    for i, uid in enumerate(data["user_id"]):
        # gender is 0
        if uid in uid_0:
            gender_0["user_id"].append(uid)
            gender_0["question_id"].append(data["question_id"][i])
            gender_0["is_correct"].append(data["is_correct"][i])
        # gender is 1
        elif uid in uid_1:
            gender_1["user_id"].append(uid)
            gender_1["question_id"].append(data["question_id"][i])
            gender_1["is_correct"].append(data["is_correct"][i])
            # gender is 2
        else:
            gender_2["user_id"].append(uid)
            gender_2["question_id"].append(data["question_id"][i])
            gender_2["is_correct"].append(data["is_correct"][i])

    return gender_0, gender_1, gender_2


def load_student_gender_meta(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "gender": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["user_id"].append(int(row[0]))
                data["gender"].append(int(row[1]))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def evaluate(data, theta, beta, alpha):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (alpha[q] * (theta[u] - beta[q])).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def irt_second(data, val_data, lr, iterations, theta_list,
               student_meta):
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
    theta = np.zeros(1774)
    for i in range(len(student_meta["user_id"])):
        if student_meta["gender"][i] == 0:
            theta[i] = (theta_list[0])
        elif student_meta["gender"][i] == 1:
            theta[i] = (theta_list[1])
        else:
            theta[i] = (theta_list[2])
    beta = np.zeros(1774)
    alpha = np.ones(1774)
    val_acc_lst = []
    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha)
        score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {} \t k: {}".format(neg_lld, score, i))
        theta, beta, alpha = update_theta_beta(data, lr, theta, beta, alpha)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, alpha, val_acc_lst


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    data_gender = load_student_gender_meta("../data/student_meta.csv")

    gender_0, gender_1, gender_2 = split_data_gender(train_data)
    iteration = 15
    learning_rate = 0.02
    itheta = []
    theta_0 = irt_first(gender_0, val_data, learning_rate, iteration)
    theta_1 = irt_first(gender_1, val_data, learning_rate, iteration)
    theta_2 = irt_first(gender_2, val_data, learning_rate, iteration)
    print("stop")
    itheta.append(sum(theta_0) / len(theta_0))
    # add initial value theta for gender 1
    itheta.append(sum(theta_1) / len(theta_1))
    # add initial value theta for gender 2
    itheta.append(sum(theta_2) / len(theta_2))
    theta, beta, alpha, val_acc_lst = irt_second(train_data, val_data, 0.02, 15,
                                                 itheta, data_gender)
    test_accuracy = evaluate(test_data, theta, beta, alpha)
    test_accuracy2 = evaluate(train_data, theta, beta, alpha)
    print("Test Accuracy: ", test_accuracy2)
    print("Test Accuracy: ", test_accuracy)

    pass


if __name__ == "__main__":
    main()
