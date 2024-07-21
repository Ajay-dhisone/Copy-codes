import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

def backpropagation(X, y, hidden_size, epochs, lr):
    input_size, output_size = X.shape[1], y.shape[1]
    W1, W2 = np.random.randn(input_size, hidden_size), np.random.randn(hidden_size, output_size)
    for _ in range(epochs):
        L1, L2 = sigmoid(np.dot(X, W1)), sigmoid(np.dot(L1, W2))
        L2_error, L1_error = y - L2, np.dot(L2_error, W2.T)
        W2 += lr * np.dot(L1.T, L2_error * sigmoid_derivative(L2))
        W1 += lr * np.dot(X.T, L1_error * sigmoid_derivative(L1))
    return W1, W2￼Enter
