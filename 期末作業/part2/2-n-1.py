import numpy as np
import matplotlib.pyplot as plt

# Generate dataset
def generate_dataset(num_samples=10000):
    X = np.random.randint(1000, 10000, (num_samples, 2))
    y = np.sum(X, axis=1, keepdims=True)
    return X, y

# Split dataset
def split_dataset(X, y, train_ratio=0.8):
    num_train = int(len(X) * train_ratio)
    X_train, y_train = X[:num_train], y[:num_train]
    X_test, y_test = X[num_train:], y[num_train:]
    return X_train, y_train, X_test, y_test

# Define activation functions and derivatives
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Initialize weights and biases
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X.T) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    return Z1, A1, Z2

# Compute loss
def compute_loss(Z2, Y):
    m = Y.shape[0]
    loss = (1/(2*m)) * np.sum((Z2 - Y.T)**2)
    return loss

# Backward propagation
def backward_propagation(X, Y, Z1, A1, Z2, W2):
    m = X.shape[0]
    dZ2 = Z2 - Y.T
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

# Train model
def train(X_train, y_train, hidden_size, learning_rate, num_iterations):
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    losses = []

    for i in range(num_iterations):
        Z1, A1, Z2 = forward_propagation(X_train, W1, b1, W2, b2)
        loss = compute_loss(Z2, y_train)
        losses.append(loss)
        dW1, db1, dW2, db2 = backward_propagation(X_train, y_train, Z1, A1, Z2, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

    return W1, b1, W2, b2, losses

# Evaluate model
def evaluate(X, Y, W1, b1, W2, b2):
    Z1, A1, Z2 = forward_propagation(X, W1, b1, W2, b2)
    loss = np.mean((Z2 - Y.T) ** 2)
    return loss, Z2.T

# Main program
if __name__ == "__main__":
    X, y = generate_dataset(num_samples=12000)
    X_train, y_train, X_test, y_test = split_dataset(X, y)

    learning_rate = 0.02
    num_iterations = 30
    hidden_sizes = range(1, 21)  # Exploring different hidden layer sizes

    test_losses = {h: [] for h in hidden_sizes}
    all_losses = {h: [] for h in hidden_sizes}
    all_predictions = {h: [] for h in hidden_sizes}

    for hidden_size in hidden_sizes:
        for i in range(30):  # Repeating training 30 times for each hidden layer size
            W1, b1, W2, b2, losses = train(X_train, y_train, hidden_size, learning_rate, num_iterations)
            train_loss, _ = evaluate(X_train, y_train, W1, b1, W2, b2)
            test_loss, pred = evaluate(X_test, y_test, W1, b1, W2, b2)
            test_losses[hidden_size].append(test_loss)
            all_losses[hidden_size].append(losses)
            all_predictions[hidden_size].append(pred)

    # Plotting boxplot for test losses
    plt.figure(figsize=(12, 6))
    plt.boxplot([test_losses[h] for h in hidden_sizes], labels=[str(h) for h in hidden_sizes])
    plt.title('Boxplot of Test Losses for Different Hidden Layer Sizes')
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Test Loss')
    plt.savefig('./boxplot_test_losses.png')
    plt.close()

    # Determining the best hidden layer size based on median test loss
    median_losses = {h: np.median(test_losses[h]) for h in hidden_sizes}
    best_hidden_size = min(median_losses, key=median_losses.get)
    best_test_losses = test_losses[best_hidden_size]
    best_index = np.argmin(best_test_losses)
    best_predictions = all_predictions[best_hidden_size][best_index]
    best_losses = all_losses[best_hidden_size][best_index]

    print("Hidden Size=" + str(best_hidden_size))

    # Plotting the best model's true vs predicted values
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='True Values')
    plt.plot(best_predictions, label='Predicted Values')
    plt.legend()
    plt.title(f'True vs Predicted Values (Hidden Size={best_hidden_size})')
    plt.xlabel('Sample Index')
    plt.ylabel('Sum')
    plt.savefig('./true_vs_predicted.png')
    plt.close()

    # Plotting learning curve for the best model
    plt.figure(figsize=(10, 5))
    plt.plot(best_losses)
    plt.title(f'Learning Curve (Hidden Size={best_hidden_size})')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('./learning_curve.png')
    plt.close()

    # Plotting histogram of errors for the best model
    errors = np.abs(best_predictions - y_test)
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=50, alpha=0.75)
    plt.title(f'Error Histogram (Hidden Size={best_hidden_size})')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.savefig('./error_histogram.png')
    plt.close()
