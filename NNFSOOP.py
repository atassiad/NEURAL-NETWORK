##
## Neural Network from Scratch Implementation
##
## Created w/ guidance from Neural Network From Scratch (NNFS) YouTube series by SentDex/Harrison Kinsley
## and the NNFS book by Harrison Kinsley

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

#neural network from scratch library which contains datasets
nnfs.init()

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1, n_neurons)) #first param is the shape

    #forward pass
    def forward(self, inputs):
        self.inputs = inputs #track input values

        #calculate output values using inputs and adjusting with weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        #gradients on params
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        #gradient on the values
        self.dinputs = np.dot(dvalues, self.weights.T)

class ActivationReLU:
    def __init__(self):
        #initalize output numpy array
        self.output = 0

    #forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    #backward pass
    def backward(self, dvalues):
        #take copy to modify original variables
        self.dinputs = dvalues.copy()

        #take zero gradient
        self.dinputs[self.inputs <= 0] = 0

class ActivationSoftMax:
    def __init__(self):
        self.output = 0
    def forward(self, inputs):
        self.inputs = inputs #remember original inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        #outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1) #flatten array

            #calculate matrix of output
            jacobianMatrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobianMatrix, single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) #clips values to range close to zero and 1

        if (len(y_true.shape) == 1):
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: #values are passed as one hot encoded vectors
                correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues) #num of samples

        #num of labels
        labels = len(dvalues[0])

        #if labels are sparse, turn to one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        #calculate gradient
        self.dinputs = -y_true / dvalues

        #normalize gradient
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():
        def __init__(self):
            self.activation = ActivationSoftMax()
            self.loss = Loss_CategoricalCrossentropy()

        #forward pass
        def forward(self, inputs, y_true):
            self.activation.forward(inputs)
            self.output = self.activation.output
            return self.loss.calculate(self.output, y_true)

        def backward(self, dvalues, y_true):
            samples = len(dvalues)

            if len(y_true.shape) == 2:
                y_true = np.argmax(y_true, axis=1)

            self.dinputs = dvalues.copy()
            self.dinputs[range(samples), y_true] -= 1
            self.dinputs = self.dinputs / samples

#create dataset from NNFS library
X, y = spiral_data(samples=100, classes=3)

#first dense layer
dense1 = LayerDense(2,3)
activation1 = ActivationReLU()

#create second dense layer which takes output of previous layer
dense2 = LayerDense(3, 3)

#softmax combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

#forward pass for dense1
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)

#output of samples
print("Loss-activation output: ")
print(loss_activation.output[:10])

#print loss
print("Loss: ", loss)

#calculate accuracy
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y=np.argmax(y,axis=1)
accuracy = np.mean(predictions==y)

#print accuracy
print("Accuracy: ", accuracy)

#backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

#print gradients
print("Gradients: ")
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)