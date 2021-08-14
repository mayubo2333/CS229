import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    clf = LocallyWeightedLinearRegression(tau=tau)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_eval)
    mse = np.mean((y_pred-y_eval)**2)
    print("MSE: {}".format(mse))

    plt.figure()
    plt.plot(x_train[:,1], y_train, 'bx')
    plt.plot(x_eval[:,1], y_pred, 'ro')
    plt.savefig("lwr_{}.png".format(tau)) 
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        y_pred = []
        for single_x in x:
            single_x = single_x.reshape(1, -1)
            dist = np.sqrt(np.sum((single_x - self.x)**2, axis=1))
            W = np.diag(
                np.exp(-dist/(2*self.tau**2))
            )
            theta = np.linalg.solve(self.x.T@W@self.x, self.x.T@W@self.y).reshape(-1, 1)
            y_pred.append(single_x@theta)

        y_pred = np.array(y_pred).reshape(-1)
        return y_pred
        # *** END CODE HERE ***
