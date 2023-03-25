from dataset import get_data

######################## Get train/test dataset ########################
X_train, X_test, Y_train, Y_test = get_data("./dataset/forestfires.csv")

########################################################################
######################## Implement you code here #######################
########################################################################
import numpy as np

np.random.seed(0)
X_train, X_test, Y_train, Y_test = X_train, X_test, Y_train.reshape(-1, 1), Y_test.reshape(-1, 1)
lr = 1e-4
N = 100000

def Task1(*, output = True):
    beta = np.random.random((X_train.shape[1], 1))
    for _ in range(N):
        grad = -2 * X_train.T @ (Y_train - X_train @ beta)
        grad /= np.abs(grad.sum())
        beta -= lr * grad
    loss = np.sum((Y_test - X_test @ beta) ** 2)
    if output: print(f"\nTask(1):\nbeta = {beta.flatten()}\nloss = {loss}\n")
    return beta, loss
# Task1()

def Task2(lmd, *, output = True):
    beta = np.linalg.inv(X_train.T @ X_train + lmd * np.eye(X_train.shape[1])) @ X_train.T @ Y_train
    loss = np.sum((Y_test - X_test @ beta) ** 2)
    if output: print(f"\nTask(2) with lmd = {lmd}:\nbeta = {beta.flatten()}\nloss = {loss}\n")
    return beta, loss
# Task2(lmd = 0.5)
# Task2(lmd = 1)
# Task2(lmd = 100000)

def Task3(lmd, gamma, *, output = True):
    K = np.exp( - np.sum((X_train[:, np.newaxis] - X_train) ** 2, axis=2) / (2 * gamma **2))
    c = np.linalg.inv(K + lmd * np.eye(K.shape[0])) @ Y_train
    K_test = np.exp( - np.sum((X_test[:, np.newaxis] - X_train) ** 2, axis=2) / (2 * gamma **2))
    Y_pred = K_test @ c
    loss = np.sum((Y_test - Y_pred) ** 2)
    if output: print(f"\nTask(3) with lmd = {lmd}, gamma = {gamma}:\nc = {c.flatten()}\nloss = {loss}\n")
    return c, loss
# Task3(0.5, 10 ** 5)
# Task3(10, 10 ** 5)

def Task4():
    """
    Spline regression.
    """

def Task5(lmd, *, output = True):
    beta = np.random.random((X_train.shape[1], 1))
    for _ in range(N):
        grad = lmd * np.sign(beta) - X_train.T @ (Y_train - X_train @ beta)
        grad /= np.abs(grad.sum())
        beta -= lr * grad
    loss = np.sum((Y_test - X_test @ beta) ** 2)
    if output: print(f"\nTask(5) with lmd = {lmd}:\nbeta = {beta.flatten()}\nloss = {loss}\n")
    return beta, loss
# Task5(100)   