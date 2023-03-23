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

def Q1():
    beta = np.random.random((X_train.shape[1], 1))
    for _ in range(N):
        grad = -2 * X_train.T @ (Y_train - X_train @ beta)
        grad /= np.abs(grad.sum())
        beta -= lr * grad
    loss = np.sum((Y_test - X_test @ beta) ** 2)
    print(f"\nTask1:\nbeta = {beta.flatten()}\nloss = {loss}\n")
    return beta
# Q1()

def Q2(lmd):
    beta = np.random.random((X_train.shape[1], 1))
    for _ in range(N):
        grad = 2 * lmd * beta - 2 * X_train.T @ (Y_train - X_train @ beta)
        grad /= np.abs(grad.sum())
        beta -= lr * grad
    loss = np.sum((Y_test - X_test @ beta) ** 2) + lmd * np.sum(beta ** 2)
    print(f"\nTask2 with lmd = {lmd}:\nbeta = {beta.flatten()}\nloss = {loss}\n")
    return beta
# Q2(lmd = 0.5)
# Q2(lmd = 1)
