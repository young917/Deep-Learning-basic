from layers import *

'''
Conv1D
def __init__(self, in_channel, out_channel,
                 kernel_size, stride):
'''
class CNN_B():
    def __init__(self):
        # Your initialization code goes here
        self.layers = [Conv1D(24, 8, 8, 4), ReLU(), Conv1D(8, 16, 1, 1), ReLU(), Conv1D(16, 4, 1, 1), Flatten()]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        weight = weights[0]
        weight = weight.T.reshape((8, 8, 24))
        self.layers[0].W = np.transpose(weight, (0, 2, 1))

        for level in range(1, 3):
            weight = weights[level]
            layer = self.layers[2*level]
            layer.W = weight.T.reshape(layer.W.shape)

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta




class CNN_C():
    def __init__(self):
        # Your initialization code goes here
        self.layers = [Conv1D(24, 2, 2, 2), ReLU(), Conv1D(2, 8, 2, 2), ReLU(), Conv1D(8, 4, 2, 1), Flatten()]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        channels = [24, 2, 8, 4]
        for w, idx in zip(weights, range(0, 3)):
            arr = np.array([w.T[c, : 2*channels[idx]] for c in range(0, channels[idx+1])])
            arr = arr.reshape((channels[idx+1], 2, channels[idx]))
            self.layers[2*idx].W = np.transpose(arr, (0, 2, 1))


    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
