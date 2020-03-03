import numpy as np
import matplotlib.pyplot as plt

from layers import Linear, Sequential, Conv2D, Flatten, MaxPooling2D
from optimizers import SGDOptimizer, RmsPropOptimizer, AdaGradOptimizer, AdamOptimizer
from losses import mse_loss, ce_loss


class Learner():
    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def fit_batch(self, X, Y):
        Y_pred, backward = self.model.forward(X)
        L, D = self.loss(Y_pred, Y)  # compute loss and dL/dy
        backward(D)  # compute w,b'gradient and back propagation
        self.model.update(self.optimizer)  # get w,b's gradient, update w,b
        return L

    def fit(self, X, Y, epochs, bs):
        losses = []
        for epoch in range(epochs):
            p = np.random.permutation(len(X))
            L = 0
            for i in range(0, len(X), bs):
                X_batch = X[p[i:i + bs]]
                Y_batch = Y[p[i:i + bs]]
                L += self.fit_batch(X_batch, Y_batch)
            print('epoch:{} Loss:{}'.format(epoch+1, L))
            losses.append(L)
        return losses


'''
if __name__ == '__main__':
    from keras.utils import to_categorical
    from keras.datasets import mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    y_train = to_categorical(y_train,10)
    y_test = to_categorical(y_test,10)
    
    X = x_train[:100,:,:]
    X = X.reshape((100,1,28,28))
    
    Y = y_train[:100]
    
    model = Sequential(
        Conv2D(1, 32, 5, 1),
        ReLu(),
        Conv2D(32, 64, 5, 1),
        ReLu(),
        MaxPooling2D(),
        Flatten(),
        Linear(10*10*64, 128),
        ReLu(),
        Linear(128, 10),
    )
    loss = Learner(model, ce_loss, SGDOptimizer(lr=0.01)).fit(X, Y, epochs=10, bs=64)  # 100 training data, can learn something
    
    plt.plot(loss)
    plt.show()
    
    ret, _ = model.forward(x_test[:1000].reshape((1000,1,28,28)))
    correct = 0
    for i in range(ret.shape[0]):
        if np.argmax(y_test[i]) == np.argmax(ret[i]):
            correct += 1
    print(correct/ret.shape[0])
'''

'''
if __name__ == '__main__':
    x_train_1D = x_train.reshape(60000,784)
    x_test_1D  = x_test.reshape(10000,784)
    model = Sequential(
        Linear(28*28, 512),
        ReLu(),
        Linear(512, 256),
        ReLu(),
        Linear(256, 10),
    )
    
    loss = Learner(model, ce_loss, SGDOptimizer(lr=0.01)).fit(x_train_1D, y_train, epochs=10, bs=256)  # test acc: 0.9179
    #loss = Learner(model, ce_loss, AdaGradOptimizer(lr=0.01)).fit(x_train_1D, y_train, epochs=10, bs=256) # test acc: 0.9816
    #loss = Learner(model, ce_loss, RmsPropOptimizer(lr=0.01)).fit(x_train_1D, y_train, epochs=10, bs=256)  # test acc: 0.9764
    #loss = Learner(model, ce_loss, AdamOptimizer(lr=0.001).fit(x_train_1D[:100], y_train[:100], epochs=10, bs=256) # test acc: 0.9421
    
    plt.plot(loss)
    plt.show()
    
    ret, _ = model.forward(x_test_1D)
    correct = 0
    for i in range(ret.shape[0]):
        if np.argmax(y_test[i]) == np.argmax(ret[i]):
            correct += 1
    print(correct/ret.shape[0])
'''


if __name__ == '__main__':
    # simple linear data: y = w1*x1 + w2*x2 + ... wi*xi + b
    in_features = 5
    num_samples = 1000
    X = np.random.randn(num_samples, in_features)
    W = np.random.randn(in_features, 1)
    B = np.random.randn(1)
    Y = X @ W + B + 0.01 * np.random.randn(num_samples, 1)

    m = Linear(in_features, 1)
    model = Sequential(m)
    loss = Learner(model, mse_loss, SGDOptimizer(lr=0.01)).fit(X, Y, epochs=100, bs=100)

    plt.plot(loss)
    plt.show()
