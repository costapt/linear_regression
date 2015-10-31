import theano
import numpy as np
import theano.tensor as T
from theano import function
import matplotlib.pyplot as plt
from theano.tensor.shared_randomstreams import RandomStreams

def add_bias(X):
    return np.insert(X,0,1,axis=1)

def add_features(X):
    (num_points, num_features) = X.shape
    for f in range(num_features):
        X = np.insert(X,num_features+f,np.square(X[:,f]),axis=1)

    return X

class LinearRegression:

    def __init__(self,add_features=True,learning_rate=0.001,threshold=0.001,debug=False):
        self.add_features = add_features
        self.debug = debug
        self.learning_rate = learning_rate
        self.threshold = threshold

    def train(self,X_train,Y_train):
        # Add new features
        if self.add_features:
            X_train = add_features(X_train)

        # Add bias to training examples in order to make the mathematics more elegant
        X_train = add_bias(X_train)
        (num_points, num_features) = X_train.shape

        # Define theano variables to represent the training examples (X) and
        # respective values (Y)
        X = T.dmatrix('X')
        Y = T.dvector('Y')

        # Define the weight of the model
        w = theano.shared(np.asarray([0]*num_features,dtype=theano.config.floatX))
        # Prediction
        y = T.dot(w,X.T)

        # Cost function: Square the difference between the real label and the
        # predicted label. Then take the mean of all these differences
        cost = T.mean(T.sqr(Y-y))
        grad = T.grad(cost=cost, wrt=w)
        updates = [[w, w - grad*self.learning_rate]]

        # Train function: Inputs X and Y. Computes the cost function.
        # X is a matrix of dimensions [num_points,num_features]
        # Y is a vector of dimension [num_points]
        # cost is a function that returns how well the model fits the reality
        # everytime train is called, w is updated
        train = function([X,Y],outputs=cost,updates=updates,allow_input_downcast=True)

        # Run the training function untill the reduction on the error is small enough
        error_before, iterations = 0, 0
        while True:
            error = train(X_train,Y_train)
            if self.debug:
                iterations = iterations + 1
                print("[iteration {0}] error = {1}".format(iterations,error))

            if abs(error_before - error) < self.threshold:
                break
            error_before = error

        self.w = w.get_value()
        # return the learned weights
        return w.get_value()

    def predict(self,X):
        if self.add_features:
            X = add_features(X)
        X = add_bias(X)
        return np.dot(self.w,np.transpose(X))


def random_noise_generator(num_points,seed=234):
    srng = RandomStreams(seed)
    rv_n = srng.normal((num_points,))
    f = function([],rv_n)
    return f()

def points_generator(f,num_points,start=0,end=4):
    x = np.linspace(start,end,num_points)
    return x,f(x)+random_noise_generator(num_points)


def main():
    num_points = 150

    x = T.dvector('x')
    y = x**2
    f = function([x],y)

    X,y = points_generator(f,num_points,start=-5,end=5)
    X = np.array([[xi] for xi in X])

    model = LinearRegression()
    w = model.train(X,y)
    y_pred = model.predict(X)

    plt.plot(X,y,'ro')
    plt.plot(X,y_pred)
    plt.show()

if __name__ == '__main__':
    main()
