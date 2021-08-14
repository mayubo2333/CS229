# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad


def logistic_regression(X, Y, name):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    init_learning_rate = 10
    learning_rate = 10

    i = 0
    i_list = []
    theta_list = []
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        if i % 100 == 0:
            Y_pred = pred_result(theta, X)
            acc = sum(Y_pred==Y)/len(Y)
            print('Finished {} iterations, theta {}, grad {}, acc {}'.format(i, theta, grad, acc))
            i_list.append(i)
            theta_list.append(np.linalg.norm(theta))

            if i % 10000==0:
                learning_rate = init_learning_rate/(i/5000)**2
            if i > 500000:
                break

        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
        
    plt.figure()
    plt.plot(i_list, theta_list)
    plt.xlabel("training step"); plt.ylabel("norm of theta")
    plt.savefig("theta_{}.png".format(name))
    plt.close()

    return


def pred_result(theta, X):
    theta = theta.reshape(-1, 1)
    score = X@theta
    y_pred = 2*(score>0)-1
    return y_pred.reshape(-1)


def plot_data(x, y, file_name):
    x_pos = x[y == 1, :]
    x_neg = x[y == -1, :]
    
    plt.figure()
    plt.scatter(x_pos[:,1], x_pos[:,2], marker='x', color='red', label='Y=1')
    plt.scatter(x_neg[:,1], x_neg[:,2], marker='o', color='blue', label='Y=-1')
    plt.legend()
    plt.savefig(file_name)
    plt.close()


def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    plot_data(Xa, Ya, "./data_a.png")
    logistic_regression(Xa, Ya, 'a')

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    plot_data(Xb, Yb, "./data_b.png")
    logistic_regression(Xb, Yb, 'b')


if __name__ == '__main__':
    main()
