import math, copy 
import numpy as np
import matplotlib.pyplot as plt


#### Linear regression functions ####

def compute_cost(X,y,w):
    """
    Compute the cost function for logistic regression.

    Parameters:
    X : numpy array (N,m). Data matrix where N is the number of samples and m is the number of features.
    y : numpy array (N,). Vector or target values.
    w : numpy array (M+1,). Weights vector including the bias term w_0.

    Returns:
        Cost value : float.
    """
    N,m = X.shape
    X = np.hstack((np.ones((N, 1)), X))  # Add dummy feature 1 term to X
    w = w.reshape(-1, 1)  # Ensure w is a column vector
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    cost  = (1/2)*np.sum((y - X@w)**2)
    return cost 

    
def compute_gradient(X,y,w):
    """
    Compute the gradient for linear regression.

    Parameters:
    X : numpy array (N,m). Data matrix where N is the number of samples and m is the number of features.
    y : numpy array (N,). Vector or target values.
    w : numpy array (m+1,). Weights vector including the bias term w_0.

    Returns:
        Gradient : numpy array (m+1,) containing at each entry the gradient with respect to the parameters w.
    """
    N,m = X.shape
    X = np.hstack((np.ones((N, 1)), X))  # Add dummy feature 1 term to X
    w = w.reshape(-1, 1)  # Ensure w is a column vector
    y = y.reshape(-1, 1)  # Ensure y is a column vector    
    gradient = - (y-X@w).T @ X 
    
    return gradient.flatten()  # Return as a flat array for consistency with w shape

def gradient_descent(X, y, w_in, cost_function, gradient_function,eta=0.01, num_iter=1000):
    """
    Batch gradient descent algorithm for linear regression. Updates the weights w by taking num_iters gradients steps with learning rate eta

    Parameters:
    X (ndarray (N,m))       :Data matrix with N samples and m features.
    y (ndarray (N,))        :Vector of target values.
    w_in (ndarray (m+1,))   :Initial weights vector including the bias term w_0.
    cost_function           :Function to compute the cost.
    gradient_function       :Function to compute the gradient. 
    eta (float)             :Learning rate.
    num_iter (int)          :Maximum number of iterations.
    tol (float)             :Tolerance for convergence.

    Returns:
        w (ndarray (m+1,))  : Final weights vector. 
        cost_history (list) : Cost history during training.
    """
    N,m = X.shape
    X = np.hstack((np.ones((N, 1)), X))  # Add dummy feature 1 term to X
    w = copy.deepcopy(w_in).reshape(-1, 1)  # Ensure w is a column vector
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    
    cost_history = [] # To store the cost at each iteration for plotting or analysis

    for i in range(num_iter):
        # Compute gradient using 
        gradient = gradient_function(X,y,w)
        w -= eta * gradient
        
        # Save the cost at each iteration
        cost_history.append(cost_function(X,y,w))
        
        # Print cost and gradient information every 10 times or as many as num_iters if less than 10
        if i% math.ceil(num_iter / 10) == 0: 
            print(f"Iteration {i:4d}: Cost = {cost_history[-1]:8.4f}, Gradient Norm = {np.linalg.norm(gradient):8.4f}") 
            
    return w, cost_history


#### Data management functions ####

def load_house_data(path):
    """Load house price data from a txt file."""
    try:
        data = np.loadtxt(path, delimiter=',')
        if data.shape[1] < 2:
            raise ValueError("Data must have at least 2 columns")
        X = data[:, :-1]
        y = data[:, -1]
        return X, y
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data file: {path}")
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")



##### Plotting functions #####




