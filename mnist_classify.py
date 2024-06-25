import load_mnist

import copy
import math
import numpy as np
import random
import sys


def rectified_linear_unit(z):
    relu = np.maximum(z, 0)
    return relu

def rectified_linear_unit_back_propagate(loss_gradient, activation_vals):
    out = np.copy(loss_gradient)
    for i, activation_val in enumerate(activation_vals):
        if activation_val <= 0:
            out[i] = 0
    return out

def softmax_stable(z):
    # constant to stabilize softmax
    c = -1 * np.max(z)
    softmax_denom = np.sum(np.exp(z + c))
    s = np.exp(z + c) / softmax_denom
    return s

def softmax_stable_back_propagate(loss_gradient, activation_vals):
    n = len(loss_gradient)
    ds_dz = np.reshape(activation_vals, (n,1)) * (np.eye(n, n) - activation_vals)
    return np.matmul(loss_gradient, ds_dz)

def affine_transformation(a, weights, biases):
    activation = np.add(np.matmul(a, weights), biases)
    return activation

def negative_log_likelihood_loss(model_output, label):
    # if probability saturates to zero, nudge up to avoid a math domain error
    loss = -1 * math.log(max(model_output[label], sys.float_info.min))
    gradient = np.zeros(model_output.shape)
    gradient[label] = -1 / max(model_output[label], sys.float_info.min)
    return loss, gradient


class Layer:
    def __init__(
        self,
        input_width,
        output_width,
        activation_fn,
        activation_back_propagate,
    ):
        self.input_width = input_width
        self.output_width = output_width
        self.activation_fn = activation_fn
        self.activation_back_propagate = activation_back_propagate
        self.weights = np.random.rand(input_width, output_width) / 10
        self.biases = np.random.rand(output_width) / 10
        self.prev_activation_input = None
        self.prev_activation_output = None
        self.weights_gradient = np.zeros((input_width, output_width))
        self.biases_gradient = np.zeros((output_width))
        self.weights_acc_gradient = np.zeros((input_width, output_width))
        self.biases_acc_gradient = np.zeros((output_width))

    def compute_activation(self, activation_input):
        """
        As a side effect, this method stores the activation input and output,
        for later use to compute parameter gradients.
        """
        z = affine_transformation(activation_input, self.weights, self.biases)
        activation_output = self.activation_fn(z)
        self.prev_activation_input = activation_input
        self.prev_activation_output = activation_output
        return activation_output


    def back_propagate(self, loss_gradient):
        """
        As a side effect, this method stores the weights and biases gradients,
        for later use to update the weight and bias parameters.
        """
        assert(loss_gradient.shape == (self.output_width,))
        assert(self.prev_activation_output.shape == (self.output_width,))
        assert(self.prev_activation_input.shape == (self.input_width,))
        dL_dz = self.activation_back_propagate(loss_gradient, self.prev_activation_output)
        dL_da = np.matmul(dL_dz, np.transpose(self.weights))
        dL_dW = np.matmul(self.prev_activation_input.reshape((self.input_width,1)), dL_dz.reshape((1, self.output_width)))
        np.add(self.weights_gradient, dL_dW, out=self.weights_gradient)
        np.add(self.biases_gradient, dL_dz, out=self.biases_gradient)
        return dL_da

    def apply_parameter_gradients(self, count):
        learning_rate = 0.001
        np.add(self.weights, self.weights_gradient * (-1 * learning_rate / count), out=self.weights)
        np.add(self.biases, self.biases_gradient * (-1 * learning_rate / count), out=self.biases)
        np.multiply(self.weights_gradient, 0, out=self.weights_gradient)
        np.multiply(self.biases_gradient, 0, out=self.biases_gradient)

    def apply_parameter_gradients_rmsprop(self, count):
        learning_rate = 0.001
        decay_rate = 0.5
        delta = 0.000001
        self.weights_gradient /= count
        self.weights_acc_gradient = (decay_rate * self.weights_acc_gradient) + ((1 - decay_rate) * self.weights_gradient**2)
        weights_gradient_delta = (-1 * learning_rate / np.sqrt(delta + self.weights_acc_gradient)) * self.weights_gradient 
        np.add(self.weights, weights_gradient_delta, out=self.weights)
        self.biases_gradient /= count
        self.biases_acc_gradient = (decay_rate * self.biases_acc_gradient) + ((1 - decay_rate) * self.biases_gradient**2)
        biases_gradient_delta = (-1 * learning_rate / np.sqrt(delta + self.biases_acc_gradient)) * self.biases_gradient 
        np.add(self.biases, biases_gradient_delta, out=self.biases)


class Model:
    def __init__(
        self,
        layers,
    ):
        self.layers = layers

    def predict(self, activation_input):
        activation = activation_input
        for layer in self.layers:
            activation = layer.compute_activation(activation)
        return activation

    def predict_batch(self, activation_inputs):
        return np.array([self.predict(a) for a in activation_inputs])

    def fit(self, images, labels):
        count = 0
        loss_sum = 0
        for img, label in zip(images, labels):
            activation = self.predict(img)
            loss, loss_gradient = negative_log_likelihood_loss(activation, label)
            loss_sum += loss
            for layer in reversed(self.layers):
                # the layer stores parameter deltas
                loss_gradient = layer.back_propagate(loss_gradient)
            count +=1
        for layer in self.layers:
            layer.apply_parameter_gradients_rmsprop(count)
        return loss_sum / count


print("loading MNIST image data")
train_images = load_mnist.read_train_images()
train_labels = load_mnist.read_train_labels()
test_images = load_mnist.read_test_images()
test_labels = load_mnist.read_test_labels()

print("normalizing data")
np.divide(train_images, 255, out=train_images)
np.divide(test_images, 255, out=test_images)

print("creating model")
layer_1_width = 512
layer_2_width = 10
model = Model([
    Layer(
        input_width=len(train_images[0]),
        output_width=layer_1_width,
        activation_fn=rectified_linear_unit,
        activation_back_propagate=rectified_linear_unit_back_propagate,
    ),
    Layer(
        input_width=layer_1_width,
        output_width=layer_2_width,
        activation_fn=softmax_stable,
        activation_back_propagate=softmax_stable_back_propagate,
    ),
])

print("train batches...")
epochs = 5
batch_size = 128
max_batch_index = len(train_images)
for epoch in range(epochs):
    batch_index = 0
    while batch_index < max_batch_index:
        images_batch = train_images[batch_index : batch_index + batch_size]
        labels_batch = train_labels[batch_index : batch_index + batch_size]
        avg_training_loss = model.fit(images_batch, labels_batch)
        if (batch_index % (50 * batch_size)) == 0:
            print(f"epoch: {epoch} batch_index: {batch_index} avg_training_loss: {avg_training_loss}")
        batch_index += batch_size

print(f"compute accuracy...")
test_examples = 1000
predictions = model.predict_batch(test_images[:test_examples])
predicted_labels = np.argmax(predictions, axis=1)
print(f"predicted_labels {predicted_labels}")
accuracy = sum(1 if a==b else 0 for a,b in zip(predicted_labels, test_labels[:test_examples])) / len(predictions)
print(f"accuracy: {accuracy}")

