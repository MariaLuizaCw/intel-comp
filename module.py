import numpy as np
import matplotlib.pyplot as plt


######################################################### Target Functions


class linear_target_function():
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
    
    def apply(self, x):
        return  np.sign(self.a * x[:, 0] + self.b * x[:, 1] + self.c)

class non_linear_target_function():
    def __init__(self) -> None:
        pass
    def apply(self, x):
        return np.sign(x[:, 0]**2 + x[:, 1]**2 - 0.6)


# def non_linear_target_function(x1, x2):
#     return np.sign(x1**2 + x2**2 - 0.6)

def generate_target_function():
    points = np.random.uniform(-1, 1, (2, 2))
    # Line parameters: ax + by + c = 0
    a = points[1, 1] - points[0, 1]
    b = points[0, 0] - points[1, 0]
    c = points[1, 0] * points[0, 1] - points[0, 0] * points[1, 1]
    return a, b, c


######################################################### Generate and tranform Data

def generate_data(N, target_function, noise_ratio=0.0):    
    X = np.random.uniform(-1, 1, (N, 2))
    y = target_function.apply(X)
        
    if noise_ratio > 0:
        num_noisy_points = int(N * noise_ratio)
        noisy_indices = np.random.choice(N, num_noisy_points, replace=False)
        y[noisy_indices] = -y[noisy_indices]
    return X, y


def transform_data(X):
    x1, x2 = X[:, 0], X[:, 1]
    X_transformed = np.c_[x1, x2, x1 * x2, x1**2, x2**2]
    return X_transformed




######################################################### Algorithms Implementation

# Perceptron
def pla(X, y, w):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    iterations = 0
    while True:
        predictions = np.sign(X_b.dot(w))
        misclassified = np.where(predictions != y)[0]
        if len(misclassified) == 0:
            break
        idx = np.random.choice(misclassified)
        w += y[idx] * X_b[idx]
        iterations += 1
    return w, iterations

# Regressão Linear
def linear_regression(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    w = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return w


# Pocket PLA
def pocket_pla(X, y, w_init, max_iterations):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    w_pocket = w_init
    best_error = np.mean(np.sign(X_b.dot(w_init)) != y)
    w = w_init.copy()
    for _ in range(max_iterations):
        predictions = np.sign(X_b.dot(w))
        misclassified = np.where(predictions != y)[0]
        if len(misclassified) == 0:
            break
        idx = np.random.choice(misclassified)
        w += y[idx] * X_b[idx]
        current_error = np.mean(np.sign(X_b.dot(w)) != y)
        if current_error < best_error:
            best_error = current_error
            w_pocket = w.copy()
    return w_pocket, best_error


######################################################### Error Function


# Calcular erro dentro e fora da amostra
def calculate_error(X, y, w):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    predictions = np.sign(X_b.dot(w))
    return np.mean(predictions != y)



