import numpy as np

class Layer:
    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_error):
        raise NotImplementedError


class ConvolutionalLayer(Layer):
    def __init__(self, input_shape, num_filters, filter_size, stride=1, padding=0, learning_rate=0.15):
        self.input_depth, self.input_height, self.input_width = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.learning_rate = learning_rate

        self.filters = 0.1 * np.random.randn(num_filters, self.input_depth, filter_size, filter_size)
        self.biases = np.zeros((num_filters, 1))

        self.input = None

    def _pad(self, X):
        if self.padding == 0:
            return X
        return np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')

    def forward(self, input_data):
        self.input = input_data
        batch_size, depth, height, width = input_data.shape
        assert depth == self.input_depth

        out_height = int((height + 2 * self.padding - self.filter_size) / self.stride) + 1
        out_width = int((width + 2 * self.padding - self.filter_size) / self.stride) + 1

        output = np.zeros((batch_size, self.num_filters, out_height, out_width))

        padded_input = self._pad(input_data)

        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(out_height):
                    for j in range(out_width):
                        vert_start = i * self.stride
                        horiz_start = j * self.stride
                        region = padded_input[b, :, vert_start:vert_start + self.filter_size, horiz_start:horiz_start + self.filter_size]
                        output[b, f, i, j] = np.sum(region * self.filters[f]) + self.biases[f, 0]

        return output

    def backward(self, output_error):
        batch_size = self.input.shape[0]
        _, _, height, width = self.input.shape

        padded_input = self._pad(self.input)
        padded_input_grad = np.zeros_like(padded_input)

        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)

        out_height = output_error.shape[2]
        out_width = output_error.shape[3]

        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(out_height):
                    for j in range(out_width):
                        vert_start = i * self.stride
                        horiz_start = j * self.stride

                        region = padded_input[b, :, vert_start:vert_start + self.filter_size, horiz_start:horiz_start + self.filter_size]
                        d_filters[f] += output_error[b, f, i, j] * region
                        padded_input_grad[b, :, vert_start:vert_start + self.filter_size, horiz_start:horiz_start + self.filter_size] += output_error[b, f, i, j] * self.filters[f]

                d_biases[f] += np.sum(output_error[b, f])

        if self.padding != 0:
            d_input = padded_input_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            d_input = padded_input_grad

        # SGD parameter update
        self.filters -= self.learning_rate * (d_filters / batch_size)
        self.biases -= self.learning_rate * (d_biases / batch_size)

        return d_input


class PoolingLayer(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.max_indices = None

    def forward(self, input_data):
        self.input = input_data
        batch_size, depth, height, width = input_data.shape

        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1

        output = np.zeros((batch_size, depth, out_height, out_width))
        self.max_indices = {}

        for b in range(batch_size):
            for d in range(depth):
                for i in range(out_height):
                    for j in range(out_width):
                        vert_start = i * self.stride
                        horiz_start = j * self.stride
                        region = input_data[b, d, vert_start:vert_start + self.pool_size, horiz_start:horiz_start + self.pool_size]
                        max_val = np.max(region)
                        output[b, d, i, j] = max_val

                        max_pos = np.argwhere(region == max_val)[0]
                        self.max_indices[(b, d, i, j)] = (vert_start + max_pos[0], horiz_start + max_pos[1])

        return output

    def backward(self, output_error):
        d_input = np.zeros_like(self.input)
        batch_size, depth, out_h, out_w = output_error.shape

        for b in range(batch_size):
            for d in range(depth):
                for i in range(out_h):
                    for j in range(out_w):
                        max_i, max_j = self.max_indices[(b, d, i, j)]
                        d_input[b, d, max_i, max_j] += output_error[b, d, i, j]

        return d_input


class DenseLayer(Layer):
    def __init__(self, input_size, output_size, learning_rate=0.15):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights = 0.1 * np.random.randn(output_size, input_size)
        self.biases = np.zeros((output_size, 1))

        self.input = None
        self.input_shape = None

    def forward(self, input_data):
        # input_data shape: (batch, input_size) or (batch, d, h, w)
        self.input_shape = input_data.shape
        if input_data.ndim > 2:
            self.input = input_data.reshape(input_data.shape[0], -1)
        else:
            self.input = input_data

        output = np.dot(self.input, self.weights.T) + self.biases.T
        return output

    def backward(self, output_error):
        batch_size = output_error.shape[0]

        dW = np.dot(output_error.T, self.input) / batch_size
        dB = np.sum(output_error, axis=0, keepdims=True).T / batch_size
        d_input = np.dot(output_error, self.weights)

        self.weights -= self.learning_rate * dW
        self.biases -= self.learning_rate * dB

        if len(self.input_shape) > 2:
            return d_input.reshape(self.input_shape)

        return d_input
