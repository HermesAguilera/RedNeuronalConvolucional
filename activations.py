import numpy as np

class Activation:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, output_error):
        raise NotImplementedError


class ReLU(Activation):
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, output_error):
        relu_grad = np.where(self.input > 0, 1.0, 0.0)
        return output_error * relu_grad


class Softmax(Activation):
    def __init__(self):
        self.output = None

    def forward(self, x):
        # Numerical stability
        expo = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = expo / np.sum(expo, axis=1, keepdims=True)
        return self.output

    def backward(self, output_error):
        # output_error is dL/dy, where y is softmax output.
        # Use Jacobian times error for each sample.
        batch_size = output_error.shape[0]
        dx = np.zeros_like(output_error)

        for i in range(batch_size):
            y = self.output[i].reshape(-1, 1)
            jacobian = np.diagflat(y) - np.dot(y, y.T)
            dx[i] = np.dot(jacobian, output_error[i])

        return dx
