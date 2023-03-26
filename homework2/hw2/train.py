from dataset import get_data

######################## Get train/test dataset ########################
X_train, X_test, Y_train, Y_test = get_data("./dataset/forestfires.csv")

########################################################################
######################## Implement you code here #######################
########################################################################
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
Y_train, Y_test = Y_train.reshape(-1, 1), Y_test.reshape(-1, 1)

def Task1(*, num_iter = 100000, lr = 1e-3, output = True):
    beta = np.random.random((X_train.shape[1], 1))
    for _ in range(num_iter):
        grad = -2 * X_train.T @ (Y_train - X_train @ beta)
        grad /= np.sum(np.abs(grad))
        beta -= lr * grad
    loss = np.sum((Y_test - X_test @ beta) ** 2)
    if output: print(f"\nTask(1):\nbeta = {beta.flatten()}\nloss = {loss}\n")
    return beta, loss

def Task2(lmd, *, output = True):
    beta = np.linalg.inv(X_train.T @ X_train + lmd * np.eye(X_train.shape[1])) @ X_train.T @ Y_train
    loss = np.sum((Y_test - X_test @ beta) ** 2)
    if output: print(f"\nTask(2) with lmd = {lmd}:\nbeta = {beta.flatten()}\nloss = {loss}\n")
    return beta, loss

def Task3(lmd, sigma, *, output = True):
    K = np.exp( - np.sum((X_train[:, np.newaxis] - X_train) ** 2, axis=2) / (2 * sigma **2))
    c = np.linalg.inv(K + lmd * np.eye(K.shape[0])) @ Y_train
    K_test = np.exp( - np.sum((X_test[:, np.newaxis] - X_train) ** 2, axis=2) / (2 * sigma **2))
    Y_pred = K_test @ c
    loss = np.sum((Y_test - Y_pred) ** 2)
    if output: print(f"\nTask(3) with lmd = {lmd}, sigma = {sigma}:\nc = {c.flatten()}\nloss = {loss}\n")
    return c, loss

def Task4():
    return 'task(4) 涉及高维的spline regression,  课堂上没有教。这一个task可以不用做。\n'

def Task5(lmd, *, num_iter = 100000, lr = 1e-4, output = True):
    beta = np.random.random((X_train.shape[1], 1))
    for _ in range(num_iter):
        grad = lmd * np.sign(beta) - X_train.T @ (Y_train - X_train @ beta)
        grad /= np.sum(np.abs(grad))
        beta -= lr * grad
    loss = np.sum((Y_test - X_test @ beta) ** 2)
    if output: print(f"\nTask(5) with lmd = {lmd}:\nbeta = {beta.flatten()}\nloss = {loss}\n")
    return beta, loss

if __name__ == '__main__':
    task1, task2, task3, task4, task5 = 1,1,1,1,1
    if task1:
        plt.cla()
        beta, loss = Task1()
        beta = beta.flatten()
        plt.bar(
            range(len(beta)),
            sorted(beta.flatten(), reverse=True)
        )
        plt.title("sorted $\\beta_i$ of Task(1)")
        plt.savefig("./assets/task1.svg")

    if task2:
        plt.cla()
        print("\nTask(2):")
        best_beta, min_loss = None, float('inf')
        for _lmd in range(1, 7):
            lmd = 10 ** _lmd
            beta, loss = Task2(lmd = lmd, output=False)
            if min_loss > loss:
                best_beta, min_loss = beta, loss
            print('lambda = %-20s loss = %-20s' %(lmd, loss))
            plt.plot(
                sorted(beta.flatten(), reverse=True),
                marker = 'o',
                label = f"$\\lambda = 10^{_lmd}$"
            )
        print(f"\nbest_beta = {best_beta.flatten()}\nbest_loss = {min_loss}\n")
        plt.legend()
        plt.title("sorted $\\beta_i$ of Task(2)")
        plt.savefig("./assets/task2.svg")

    if task3:
        plt.cla()
        print("\nTask(3):")
        best_beta, min_loss = None, float('inf')
        for _lmd in range(1, 10, 2):
            lmd = _lmd / 10
            beta, loss = Task3(lmd = lmd, sigma = 1e5, output=False)
            if min_loss > loss:
                best_beta, min_loss = beta, loss
            print('lambda = %-20s loss = %-20s' %(lmd, loss))
            plt.plot(
                sorted(beta.flatten(), reverse=True),
                # marker = 'o',
                label = f"$\\lambda = {lmd}$"
            )
        print(f"\nbest_c = {best_beta.flatten()}\nbest_loss = {min_loss}\n")
        plt.legend()
        plt.title("sorted $\\beta_i$ of Task(3) with $\\sigma = 10^5$")
        plt.savefig("./assets/task3.svg")

    if task4:
        print("\nTask(4):")
        print(Task4())

    if task5:
        plt.cla()
        print("\nTask(5):")
        best_beta, min_loss = None, float('inf')
        for _lmd in range(2000, 3000, 200):
            lmd = _lmd
            beta, loss = Task5(lmd = lmd, output=False)
            if min_loss > loss:
                best_beta, min_loss = beta, loss
            print('lambda = %-20s loss = %-20s' %(lmd, loss))
            plt.plot(
                sorted(beta.flatten(), reverse=True),
                marker = 'o',
                label = f"$\\lambda = {_lmd}$"
            )
        print(f"\nbest_beta = {best_beta.flatten()}\nbest_loss = {min_loss}\n")
        plt.legend()
        plt.title("sorted $\\beta_i$ of Task(5)")
        plt.savefig("./assets/task5.svg")