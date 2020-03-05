import numpy as np
import math


class Linear():
    # DO NOT DELETE
    def __init__(self, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.W = np.random.randn(out_feature, in_feature)
        self.b = np.zeros(out_feature)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        self.out = x.dot(self.W.T) + self.b
        return self.out

    def backward(self, delta):
        self.db = delta
        self.dW = np.dot(self.x.T, delta)
        dx = np.dot(delta, self.W.T)
        return dx


class Conv1D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        self.W = np.random.randn(out_channel, in_channel, kernel_size)
        self.b = np.zeros(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        ## Your codes here
        self.batch, __, self.width = x.shape
        self.x = x
        assert __ == self.in_channel, 'Expected the inputs to have {} channels'.format(self.in_channel)

        out_width = math.floor((self.width - self.kernel_size) / self.stride + 1)
        output = np.zeros((self.batch, self.out_channel, out_width))

        for x_, output_ in zip(self.x, output):
            for w_, output_chn, b_chn in zip(self.W, output_, self.b):
                for out_idx in range(out_width):
                    x_idx = self.stride * out_idx
                    output_chn[out_idx] += np.sum(x_[:, x_idx:x_idx + self.kernel_size] * w_)
                    output_chn[out_idx] += b_chn

        return output

    def backward(self, delta):

        ## Your codes here
        assert delta.shape[1] == self.out_channel
        # self.db = self.delta.sum(axis=0) maybe possible...
        dx = np.zeros((self.batch, self.in_channel, self.width))

        for x_, dx_, d_ in zip(self.x, dx, delta):
            for w_, dw_, d_chn in zip(self.W, self.dW, d_):
                for out_idx in range(d_chn.shape[0]):
                    x_idx = self.stride * out_idx
                    dw_ += d_chn[out_idx] * x_[:, x_idx:x_idx + self.kernel_size]
                    dx_[:, x_idx:x_idx + self.kernel_size] += d_chn[out_idx] * w_
            self.db += d_.sum(axis=1)


        return dx


class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # Your codes here
        self.channel = x.shape[1]
        return np.reshape(x, (x.shape[0], -1))

    def backward(self, x):
        # Your codes here
        raise np.reshape(x, (x.shape[0], self.channel, -1))


class ReLU():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.dy = (x >= 0).astype(x.dtype)
        return x * self.dy

    def backward(self, delta):
        return self.dy * delta
