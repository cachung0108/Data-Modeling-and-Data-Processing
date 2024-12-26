import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

def generate_xor_dataset(num_samples=10000):
    X = np.zeros((num_samples, 2))
    y = np.zeros((num_samples, 1))
    
    for i in range(num_samples):
        x1 = np.random.choice([np.random.uniform(-0.5, 0.2), np.random.uniform(0.8, 1.5)])
        x2 = np.random.choice([np.random.uniform(-0.5, 0.2), np.random.uniform(0.8, 1.5)])
        X[i] = [x1, x2]
        x1_bin = 0 if x1 < 0.5 else 1
        x2_bin = 0 if x2 < 0.5 else 1
        y[i] = x1_bin ^ x2_bin
        
    return X, y

# Tangent Sigmoid activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Initialize weights
def initialize_network(input_size, hidden_size, output_size):
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

# Forward propagation with Tanh and Linear activation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X.T) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = Z2  # Linear activation for output layer
    return Z1, A1, Z2, A2

# Compute loss (Mean Squared Error)
def compute_loss(A2, Y):
    m = Y.shape[0]
    return (1/m) * np.sum((A2.T - Y)**2)

# Backpropagation with Tanh and Linear activation
def backward_propagation(X, Y, Z1, A1, Z2, A2, W2):
    m = X.shape[0]
    dZ2 = A2 - Y.T
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * tanh_derivative(Z1)
    dW1 = (1/m) * np.dot(dZ1, X)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# Update parameters
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Plot decision boundary
def plot_decision_boundary(X, W1, b1, W2, b2, iteration):
    x_min, x_max = -1, 2  # Set fixed x-axis limits
    y_min, y_max = -1, 2     # Set fixed y-axis limits
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    _, _, _, Z = forward_propagation(grid, W1, b1, W2, b2)
    Z = Z.reshape(xx.shape)
    
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], edgecolors='k', s=20)
    
    # Plot hidden layer decision boundaries
    x_vals = np.linspace(x_min, x_max, 200)
    for i in range(W1.shape[0]):
        w = W1[i]
        b = b1[i]
        y_vals = -(w[0] * x_vals + b) / w[1]
        plt.plot(x_vals, y_vals, '--', label=f'Hidden neuron {i+1}')
    
    plt.title(f"Decision Boundary at iteration {iteration}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.savefig(f"frame_{iteration}.png")
    plt.close()

# Training model and visualizing
def train_and_visualize(X, y, hidden_size=2, output_size=1, learning_rate=0.1, num_iterations=10001, every_n_iter=1000):
    input_size = X.shape[1]
    W1, b1, W2, b2 = initialize_network(input_size, hidden_size, output_size)
    frames = []
    
    for i in range(num_iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        loss = compute_loss(A2, y)
        dW1, db1, dW2, db2 = backward_propagation(X, y, Z1, A1, Z2, A2, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        if i % every_n_iter == 0:
            print(f"Iteration {i}, loss: {loss}")
            plot_decision_boundary(X, W1, b1, W2, b2, i)
            frames.append(imageio.imread(f"frame_{i}.png"))
    
    # Create a GIF
    imageio.mimsave('./training_process.gif', frames, fps=1)
    return W1, b1, W2, b2

# Example usage
X, y = generate_xor_dataset(1000)  # Using a smaller dataset for quicker visualization
train_and_visualize(X, y, learning_rate=0.1, num_iterations=10001, every_n_iter=1000)
