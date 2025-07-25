# Complete implementation of neural network, AI-generated, for inspiration
# Uses definitions from neural_networks_fundamentals.ipynb
class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu', output_activation='sigmoid'):
        """
        Initialize a neural network with specified architecture
        
        Args:
            layer_sizes: List of integers specifying the size of each layer
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation = activation
        self.output_activation = output_activation
        
        # Initialize weights and biases
        self.weights = {}
        self.biases = {}
        self.initialize_parameters()
        
        # For storing activations during forward pass
        self.cache = {}
        
    def initialize_parameters(self):
        """Initialize weights and biases using Xavier initialization"""
        for i in range(1, self.num_layers):
            # Xavier initialization
            self.weights[i] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) * np.sqrt(1.0 / self.layer_sizes[i-1])
            self.biases[i] = np.zeros((self.layer_sizes[i], 1))
    
    def forward_propagation(self, X):
        """
        Perform forward propagation through the network
        
        Args:
            X: Input data of shape (features, samples)
            
        Returns:
            A: Output of the network
        """
        A = X
        self.cache['A0'] = A
        
        # Forward through hidden layers
        for i in range(1, self.num_layers - 1):
            Z = np.dot(self.weights[i], A) + self.biases[i]
            A = self._activate(Z, self.activation)
            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i}'] = A
        
        # Output layer
        i = self.num_layers - 1
        Z = np.dot(self.weights[i], A) + self.biases[i]
        A = self._activate(Z, self.output_activation)
        self.cache[f'Z{i}'] = Z
        self.cache[f'A{i}'] = A
        
        return A
    
    def backward_propagation(self, X, Y, output):
        """
        Perform backward propagation to compute gradients
        
        Args:
            X: Input data
            Y: True labels
            output: Network output from forward pass
            
        Returns:
            grads: Dictionary containing gradients for weights and biases
        """
        m = X.shape[1]  # Number of samples
        grads = {}
        
        # Start with output layer gradient
        if self.output_activation == 'sigmoid':
            dZ = output - Y  # For sigmoid + binary cross-entropy
        elif self.output_activation == 'softmax':
            dZ = output - Y  # For softmax + categorical cross-entropy
        else:
            # For other cases, compute derivative
            dA = -(Y / output) + (1 - Y) / (1 - output)  # BCE derivative
            dZ = dA * self._activate_derivative(self.cache[f'Z{self.num_layers-1}'], self.output_activation)
        
        # Gradients for output layer
        L = self.num_layers - 1
        grads[f'dW{L}'] = (1/m) * np.dot(dZ, self.cache[f'A{L-1}'].T)
        grads[f'db{L}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Backward through hidden layers
        for i in range(self.num_layers - 2, 0, -1):
            dA = np.dot(self.weights[i+1].T, dZ)
            dZ = dA * self._activate_derivative(self.cache[f'Z{i}'], self.activation)
            
            grads[f'dW{i}'] = (1/m) * np.dot(dZ, self.cache[f'A{i-1}'].T)
            grads[f'db{i}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        return grads
    
    def update_parameters(self, grads, learning_rate):
        """Update parameters using computed gradients"""
        for i in range(1, self.num_layers):
            self.weights[i] -= learning_rate * grads[f'dW{i}']
            self.biases[i] -= learning_rate * grads[f'db{i}']
    
    def compute_loss(self, Y, output):
        """Compute loss based on output activation"""
        if self.output_activation == 'sigmoid':
            return LossFunctions.binary_crossentropy(Y.T, output.T)
        elif self.output_activation == 'linear':
            return LossFunctions.mse(Y.T, output.T)
        else:
            return LossFunctions.binary_crossentropy(Y.T, output.T)
    
    def train(self, X, Y, epochs=1000, learning_rate=0.01, print_cost=True):
        """
        Train the neural network
        
        Args:
            X: Training input data
            Y: Training labels
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            print_cost: Whether to print cost during training
        """
        costs = []
        
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward_propagation(X)
            
            # Compute cost
            cost = self.compute_loss(Y, output)
            costs.append(cost)
            
            # Backward propagation
            grads = self.backward_propagation(X, Y, output)
            
            # Update parameters
            self.update_parameters(grads, learning_rate)
            
            # Print cost
            if print_cost and epoch % 100 == 0:
                print(f"Epoch {epoch}: Cost = {cost:.6f}")
        
        return costs
    
    def predict(self, X):
        """Make predictions on new data"""
        output = self.forward_propagation(X)
        if self.output_activation == 'sigmoid':
            return (output > 0.5).astype(int)
        else:
            return output
    
    def _activate(self, Z, activation):
        """Apply activation function"""
        if activation == 'relu':
            return ActivationFunctions.relu(Z)
        elif activation == 'sigmoid':
            return ActivationFunctions.sigmoid(Z)
        elif activation == 'tanh':
            return ActivationFunctions.tanh(Z)
        elif activation == 'softmax':
            return self._softmax(Z)
        elif activation == 'linear':
            return Z
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _activate_derivative(self, Z, activation):
        """Compute derivative of activation function"""
        if activation == 'relu':
            return ActivationFunctions.relu_derivative(Z)
        elif activation == 'sigmoid':
            return ActivationFunctions.sigmoid_derivative(Z)
        elif activation == 'tanh':
            return ActivationFunctions.tanh_derivative(Z)
        elif activation == 'linear':
            return np.ones_like(Z)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _softmax(self, Z):
        """Compute softmax activation"""
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Numerical stability
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

# Test our neural network implementation
print("TESTING NEURAL NETWORK IMPLEMENTATION")
print("="*50)

# Create synthetic dataset for binary classification
np.random.seed(42)
m = 1000  # number of samples
X_test = np.random.randn(2, m)
Y_test = ((X_test[0]**2 + X_test[1]**2) > 1).astype(int).reshape(1, m)

print(f"Dataset: {m} samples, {X_test.shape[0]} features")
print(f"Positive class: {np.sum(Y_test)}/{m} ({100*np.mean(Y_test):.1f}%)")

# Create and train neural network
nn = NeuralNetwork([2, 5, 3, 1], activation='relu', output_activation='sigmoid')

print(f"\nNetwork architecture: {nn.layer_sizes}")
print("Hidden activation: ReLU")
print("Output activation: Sigmoid")
print("Loss function: Binary Cross-Entropy")

# Train the network
print("\nTraining neural network...")
costs = nn.train(X_test, Y_test, epochs=1000, learning_rate=0.1, print_cost=True)

# Make predictions
predictions = nn.predict(X_test)
accuracy = np.mean(predictions == Y_test) * 100

print(f"\nFinal training accuracy: {accuracy:.2f}%")

# Visualize training progress and decision boundary
plt.figure(figsize=(15, 5))

# Plot 1: Training loss
plt.subplot(1, 3, 1)
plt.plot(costs)
plt.title('Training Loss Over Time')
plt.xlabel('Epochs')
plt.ylabel('Binary Cross-Entropy Loss')
plt.grid(True, alpha=0.3)

# Plot 2: Data distribution
plt.subplot(1, 3, 2)
positive_mask = Y_test[0] == 1
negative_mask = Y_test[0] == 0

plt.scatter(X_test[0, positive_mask], X_test[1, positive_mask], 
           c='red', alpha=0.6, label='Positive (outside circle)', s=20)
plt.scatter(X_test[0, negative_mask], X_test[1, negative_mask], 
           c='blue', alpha=0.6, label='Negative (inside circle)', s=20)

# Draw true decision boundary (circle)
theta = np.linspace(0, 2*np.pi, 100)
circle_x = np.cos(theta)
circle_y = np.sin(theta)
plt.plot(circle_x, circle_y, 'k--', linewidth=2, label='True boundary')

plt.title('Data Distribution and True Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.axis('equal')
plt.grid(True, alpha=0.3)

# Plot 3: Learned decision boundary
plt.subplot(1, 3, 3)
x1_range = np.linspace(-3, 3, 50)
x2_range = np.linspace(-3, 3, 50)
xx1, xx2 = np.meshgrid(x1_range, x2_range)
mesh_points = np.c_[xx1.ravel(), xx2.ravel()].T

# Get network predictions for mesh
mesh_predictions = nn.forward_propagation(mesh_points)
mesh_predictions = mesh_predictions.reshape(xx1.shape)

plt.contourf(xx1, xx2, mesh_predictions, levels=50, alpha=0.6, cmap='RdYlBu')
plt.colorbar(label='Predicted Probability')
plt.contour(xx1, xx2, mesh_predictions, levels=[0.5], colors='black', linewidths=2)

plt.scatter(X_test[0, positive_mask], X_test[1, positive_mask], 
           c='red', alpha=0.8, label='Positive', s=20)
plt.scatter(X_test[0, negative_mask], X_test[1, negative_mask], 
           c='blue', alpha=0.8, label='Negative', s=20)

plt.title('Learned Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstrate gradient checking
def gradient_check(nn, X, Y, epsilon=1e-7):
    """
    Verify backpropagation implementation using finite differences
    """
    print("\nGRADIENT CHECKING")
    print("="*30)
    
    # Forward pass to get initial cost
    output = nn.forward_propagation(X)
    grads = nn.backward_propagation(X, Y, output)
    
    # Check gradients for first layer weights only (for brevity)
    W = nn.weights[1]
    dW_backprop = grads['dW1']
    
    # Compute numerical gradients
    dW_numerical = np.zeros_like(W)
    
    for i in range(min(3, W.shape[0])):  # Check only first 3 rows for speed
        for j in range(min(3, W.shape[1])):  # Check only first 3 columns
            # W + epsilon
            W[i, j] += epsilon
            output_plus = nn.forward_propagation(X)
            cost_plus = nn.compute_loss(Y, output_plus)
            
            # W - epsilon  
            W[i, j] -= 2 * epsilon
            output_minus = nn.forward_propagation(X)
            cost_minus = nn.compute_loss(Y, output_minus)
            
            # Restore original value
            W[i, j] += epsilon
            
            # Numerical gradient
            dW_numerical[i, j] = (cost_plus - cost_minus) / (2 * epsilon)
    
    # Compare gradients
    difference = np.linalg.norm(dW_backprop[:3, :3] - dW_numerical[:3, :3])
    relative_error = difference / (np.linalg.norm(dW_backprop[:3, :3]) + np.linalg.norm(dW_numerical[:3, :3]))
    
    print(f"Backprop gradient (sample): \\n{dW_backprop[:3, :3]}")
    print(f"Numerical gradient (sample): \\n{dW_numerical[:3, :3]}")
    print(f"Relative error: {relative_error:.2e}")
    
    if relative_error < 1e-7:
        print("✓ Gradient check PASSED! Backpropagation is correctly implemented.")
    else:
        print("✗ Gradient check FAILED! There may be a bug in backpropagation.")

# Perform gradient check on a small sample
X_small = X_test[:, :10]
Y_small = Y_test[:, :10]
nn_check = NeuralNetwork([2, 3, 1], activation='sigmoid', output_activation='sigmoid')
gradient_check(nn_check, X_small, Y_small)

print("\n" + "="*60)
print("BACKPROPAGATION SUMMARY")
print("="*60)
print("\n✓ Forward Pass: Compute predictions layer by layer")
print("✓ Loss Calculation: Measure prediction error")  
print("✓ Backward Pass: Compute gradients using chain rule")
print("✓ Parameter Update: Adjust weights and biases")
print("✓ Repeat: Iterate until convergence")
print("\n🎯 Key Insight: Backpropagation enables efficient learning")
print("   by computing gradients in O(parameters) time complexity")
print("\n🔧 Implementation: Our neural network successfully learned")
print(f"   a non-linear decision boundary with {accuracy:.1f}% accuracy")
