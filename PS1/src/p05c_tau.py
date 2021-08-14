import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    best_tau, best_mse = None, np.float('inf')
    for tau in tau_values:
        clf = LocallyWeightedLinearRegression(tau=tau)
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_valid)
        mse = np.mean((y_pred-y_valid)**2)
        print("MSE on valid set: {}\ttau: {}".format(mse, tau))

        plt.figure()
        plt.plot(x_train[:,1], y_train, 'bx')
        plt.plot(x_valid[:,1], y_pred, 'ro')
        plt.savefig("lwr_{}.png".format(tau)) 

        if mse<best_mse:
            best_mse = mse
            best_tau = tau
    print("Best tau: {}".format(best_tau))
    
    clf = LocallyWeightedLinearRegression(tau=best_tau)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    mse = np.mean((y_pred-y_test)**2)
    print("MSE on test set: {}\ttau: {}".format(mse, best_tau))
    
    # *** END CODE HERE ***
