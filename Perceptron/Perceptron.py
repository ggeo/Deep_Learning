import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class Perceptron():
    """
    The Perceptron classifier.

    Parameters
    ----------
    nb_iterations : int
                    Number of passes over the training dataset.
    eta : float
          Learning rate 0 < eta < 1.
    """
    def __init__(self, nb_iterations, eta):
        self.nb_iterations = nb_iterations
        self.eta = eta

    def activation(self, x):
        """
        The activation function that the neuron will
        use in order to fire or not (1 or -1).

        Parameters
        ----------
        x : array
            The array of values to normalize.
        
        Returns
        -------
        array
        The normalized values.
        """
        return np.where(x >= 0, 1, -1)
        
    def fit(self, X, y):
        """
        Fit the training data.
        
        Parameters
        -----------
        X : array [samples, features]
            Training array.
        y : array [1, samples]
            Target array.
        
        Returns
        -------
        None 
        """
        eta = self.eta
        nb_iterations = self.nb_iterations
        # The weights size is the same as X.shape[1] (features)
        # plus one for counting the bias.
        # The bias is used in order to be able to shift the line 
        # y = ax + b (b is the bias term).
        self.weights = np.zeros(1 + X.shape[1])
        self.bias = self.weights[0]
        self.errors = []
        for i in range(nb_iterations):
            # The predicted output
            output = self.predict(X)
            # Calculate the error/cost (desired output - computed output) in order to
            # find out how far you are from the desired output.
            # We are using the sum of squared errors method (1/2m * Sum(y-yp)^2)
            # in order to learn the weights
            # Basically, the cost function (a function that we want to minimize)
            #learns the weights as the sum of squared errors
            error = y - output
            # Adjust weights
            # If input is zero, the weights do not change.
            # The ideal values for weight are the ones where the error is zero.
            # If the expected value (training_outputs) is bigger than the
            # predicted (output) we need to increase the weights.
            # In the opposite case we need to decrease the weights.
            # To do so, we upadte the weights by minimizing the cost function by
            # gradient descent.
            self.weights[1:] += eta * np.dot(X.T, error)
            self.bias += eta * error.sum()
            cost = 0.5 * np.mean(error**2)
            self.errors.append(cost)
            

    def predict(self, X):
        """
        Predict the class label using the activation function.
        
        Parameters
        ----------
        X : array [samples, features]
            Training array.
            
        Returns
        -------
        Class label (1 or -1)
        """
        return self.activation(np.dot(X, self.weights[1:]) + self.bias)    

if __name__ == "__main__":
    # Train data (implement and gate)
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y_train = np.array([-1, -1, -1, 1])

    # Train perceptron
    per = Perceptron(1000, 0.02)
    per.fit(X_train, Y_train)

    # Create x,y limits and a meshgrid.
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    # Step size in the mesh.
    h_step = 0.02 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_step), np.arange(y_min, y_max, h_step))

    # Plot the decision boundary (predict).
    # Assign a color to each point in the mesh.
    fig, ax = plt.subplots()
    Z = per.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Make contourf plot
    ax.contourf(xx, yy, Z, cmap='RdGy')
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap='ocean')
    ax.set_title('Perceptron')
    ax.axis('off')
    plt.show()