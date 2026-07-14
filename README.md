# Simulating-a-perceptron-using-NumPy
Code explanation :
This script implements a basic Single-Layer Perceptron using NumPy to learn the behavior of an AND logic gate. 
step_function(x) : 
Which serves as the network's activation gatekeeper by returning a binary 1 if the incoming value is zero or positive, and a 0 otherwise. 

The class Perceptron is structured with three core methods:
__init__ :
Sets up the initial model state by initializing weights as a zero array matching the data's dimensions, setting the bias to zero, and establishing the learning rate (lr); predict performs a forward mathematical pass by calculating the vector dot product of the input features and current weights, adding the bias shifting factor (self.bias), and passing the total sum into the step function; and train executes the iterative Perceptron Learning Rule over a designated number of epochs, evaluating the current error (target - prediction) row-by-row to structurally update both the weights and the bias parameters whenever a prediction is incorrect. Finally, the main execution block defines the classic binary matrices for the training inputs (X) and ground-truth targets (y) matching an AND gate configuration, instantiates a two-input perceptron, calls the training routine over 10 epochs until the weights converge on a linear boundary line, and prints the resolved evaluation loop to verify that the model correctly predicts the true outputs for all four coordinates.
