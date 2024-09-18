import numpy as np

inputs = [1,2,3, 2.5]
weights = [0.2,0.8,-0.5, 1.0]
biases = 2

#order matters, weights come first
value = np.dot(weights, inputs) + biases
print(value)