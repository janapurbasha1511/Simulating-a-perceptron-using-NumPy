Code with explanation
# NumPy stands for numerical python, it is used for numerical computations in python
import numpy as np

# Step activation function is the decision making function in this program, it checks if the prediction is correct (1) or wrong (0)
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron class initialises the weights and the bias as 0, and the learning rate is initliased as 0.1 to specify how quickly the weights change, if the epoch is increased then the perceptron learns in fewer epochs
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = learning_rate
#predict class calculated the dot product of the data point and the weight of that point and adds the bias and this is passed to the step function class which dmakes the decision, if z>=0, it outputs 1, if z<0, it outputs 0.
    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return step_function(z)
#this is the training class, here epochs are taken as 10, epochs should be taken larger for larger datasets to stabilise the weights, since our dataset is so small the weights stabilise after 4-5epochs
#this class also updates the weights and the bias after calculating the error term that is target value - prediction value
#this whole thing runs within a for loop that runs according to the number of epochs, since our dataet has 4 rows and 10 epochs, it makes 40 data points
    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi) #jumps to prediction class where the dot product is calculated and sent to step function
                error = target - prediction
                self.weights += self.lr * error * xi #weights are updated for wach data point
                self.bias += self.lr * error #bias gets updated for all data points

# Training data for AND gate
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

# Train the perceptron
p = Perceptron(input_size=2) #each input size will have two features
p.train(X, y, epochs=10)

# Test predictions
print("Predictions:")
for xi in X:
    print(f"Input: {xi}, Output: {p.predict(xi)}")
