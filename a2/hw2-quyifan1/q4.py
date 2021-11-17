# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from sklearn.datasets import load_boston
from sklearn import model_selection
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a d x 1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    test_datum1 = test_datum.reshape(1, test_datum.shape[0])
    distance = l2(test_datum1, x_train)
    # 1 X N matrix (distance
    ai_up = np.exp(-distance / (2 * tau * tau))
    # mark
    ai_down = np.exp(logsumexp((-distance) / (2 * (tau ** 2))))
    a_i = ai_up / ai_down
    A = np.diag(a_i[0, :])
    second_half = x_train.T.dot(A).dot(y_train)
    first_half = np.dot((np.dot(x_train.T, A)), x_train) + np.identity(x_train.shape[1])
    w_star = np.linalg.solve(first_half, second_half)
    prediction = np.dot(test_datum.T, w_star)
    return prediction
    ## TODO



def run_validation(x, y, taus, val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    ## TODO
    training_losses = np.empty_like(taus)
    valid_losses = np.empty_like(taus)
    training_X, validation_X, training_target, validation_target = model_selection.train_test_split(x, y, test_size=0.3,
                                                              random_state=41, shuffle=True)

    # Compute average loss for each tau
    for i in range(len(taus)):
        # train_predictions = np.array([
        #     LRLS(datum, training_X, training_target, taus[i])
        #     for datum in training_X
        # ])
        test_predictions = np.array([
            LRLS(datum, training_X, training_target, taus[i])
            for datum in validation_X
        ])

        # Error for each datum
        # train_errs = (train_predictions - training_target)
        test_errs = (test_predictions - validation_target)

        # Use mean squared error
        # training_losses[i] = np.mean(train_errs ** 2)
        valid_losses[i] = np.mean(test_errs ** 2)

    return training_losses, valid_losses
    ## TODO


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    train_losses, test_losses = run_validation(x, y, taus, val_frac=0.3)
    # plt.semilogx(taus, train_losses, color="red", label="Train Losses")
    plt.semilogx(taus, test_losses, color="blue", label="Validation (Test) Losses")
    plt.xlabel('Tau')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
