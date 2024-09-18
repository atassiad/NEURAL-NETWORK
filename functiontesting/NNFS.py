import numpy as np

#layer 1
#sample inputs
inputs = [[1,2,3, 2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]

#weights & biases for each input
weights = [[0.2,0.8,-0.5, 1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

biases = [2, 3, 0.5]

#layer 2
weights2 = [[0.1,-0.14,0.5],
            [-0.5,0.12,-0.33],
            [-0.44,0.73,-0.13]]
biases2 = [-1, 2, -0.5]

""""
Hard-coded output calculation
output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights3[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
"""

#simpler output calculation (w/out numpy)- inputs*weights+bias
"""""
for neuron_weight, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weight):
        neuron_output += weight*n_input
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)
"""

#output calculation w/ numpy on a singular vector of inputs
"""""
neuron_output = 0
for neuron_weight, neuron_bias in zip(weights, biases):
    neuron_output = 0
    neuron_output += np.dot(neuron_weight, inputs) + neuron_bias
    layer_outputs.append(neuron_output)
"""

#output calculation w/ numpy on a batch of inputs
#transposes of weights allows the rows*column matrice calculation to work, biases is just addition
"""""
outputs = np.dot(inputs, np.array(weights).T) + biases
print(outputs)
"""

#multiple layer batch output calculations
outputs = np.dot(inputs, np.array(weights).T) + biases #layer 1
outputs = np.dot(outputs, np.array(weights2).T) + biases2 #layer 2
print(outputs)