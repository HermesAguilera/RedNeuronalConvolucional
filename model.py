import numpy as np


class CNN:
    def __init__(self, num_classes=5, confidence_threshold=0.5):
        self.layers = []
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, output_error):
        grad = output_error
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    @staticmethod
    def _to_one_hot(y, num_classes):
        y = y.astype(int)
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y] = 1.0
        return one_hot

    @staticmethod
    def _cross_entropy_loss(y_true_one_hot, y_pred, eps=1e-12):
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(np.sum(y_true_one_hot * np.log(y_pred), axis=1))

    @staticmethod
    def _cross_entropy_grad_wrt_softmax_output(y_true_one_hot, y_pred, eps=1e-12):
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        batch_size = y_true_one_hot.shape[0]
        return -(y_true_one_hot / y_pred) / batch_size

    def fit(self, x_train, y_train, epochs=10, batch_size=32, verbose=True, shuffle=True):
        n_samples = x_train.shape[0]
        y_train = y_train.astype(int)

        for epoch in range(1, epochs + 1):
            if shuffle:
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                x_train = x_train[indices]
                y_train = y_train[indices]

            epoch_loss = 0.0
            correct = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                xb = x_train[start:end]
                yb = y_train[start:end]

                y_one_hot = self._to_one_hot(yb, self.num_classes)

                probs = self.forward(xb)
                loss = self._cross_entropy_loss(y_one_hot, probs)
                epoch_loss += loss * (end - start)

                pred_labels = np.argmax(probs, axis=1)
                correct += np.sum(pred_labels == yb)

                grad = self._cross_entropy_grad_wrt_softmax_output(y_one_hot, probs)
                self.backward(grad)

            epoch_loss /= n_samples
            epoch_acc = correct / n_samples

            if verbose:
                print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")

    def predict_proba(self, x):
        return self.forward(x)

    def predict(self, x, confidence_threshold=None, none_label="Ninguna de las anteriores"):
        threshold = self.confidence_threshold if confidence_threshold is None else confidence_threshold
        probs = self.predict_proba(x)

        max_probs = np.max(probs, axis=1)
        labels = np.argmax(probs, axis=1)

        outputs = []
        for i in range(len(labels)):
            if max_probs[i] < threshold:
                outputs.append(none_label)
            else:
                outputs.append(int(labels[i]))

        return outputs, probs
