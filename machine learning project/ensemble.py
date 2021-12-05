# TODO: complete this file.
from utils import *
from part_a.matrix_factorization import *
import numpy as np


def generate_sub_data(train_data, size):
    """
    :param train_data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param size: size of the sub dataset
    :return: a sub dataset choose from train_data
    """
    sub_data = {"user_id": [], "question_id": [], "is_correct": []}
    for i in range(size):
        ind = \
            np.random.choice(len(train_data["question_id"]), 1)[0]
        sub_data["user_id"].append(train_data["user_id"][ind])
        sub_data["question_id"].append(train_data["question_id"][ind])
        sub_data["is_correct"].append(train_data["is_correct"][ind])
    return sub_data


def predict_ensemble(data, matrices, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrices: 2D matrices
    :param threshold: float
    :return: float indicate accuracy of ensemble
    """
    num_prediction = 0
    num_accurate = 0
    for i in range(len(data["is_correct"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        num_correct = 0
        for matrix in matrices:
            if matrix[user_id, question_id] >= threshold and data["is_correct"][i]:
                num_correct += 1
            elif matrix[user_id, question_id] < threshold and not data["is_correct"][i]:
                num_correct += 1
        if num_correct >= 2:
            num_accurate += 1
        num_prediction += 1
    return num_accurate / num_prediction


def factorization_ensemble():
    # train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    rec_matrices = []
    k = 9
    lr = 0.05
    num_iteration = 50000
    for i in range(3):
        resampled_data = generate_sub_data(train_data, int(len(train_data["user_id"])))
        mat, train_losses, val_losses = als(resampled_data, k, lr, num_iteration, val_data)
        rec_matrices.append(mat)
    val_acc = predict_ensemble(val_data, rec_matrices)
    test_acc = predict_ensemble(test_data, rec_matrices)
    print(f"Ensemble validation accuracy: {val_acc}, test accuracy: {test_acc}")

    origin_matrix, train_losses, val_losses = als(train_data, k, lr, num_iteration, val_data)
    val_acc = sparse_matrix_evaluate(val_data, origin_matrix)
    test_acc = sparse_matrix_evaluate(test_data, origin_matrix)
    print(f"Original validation accuracy: {val_acc}, test accuracy: {test_acc}")


def main():
    factorization_ensemble()


if __name__ == "__main__":
    main()
