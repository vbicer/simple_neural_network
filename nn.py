import numpy as np
import matplotlib.pyplot as pyplot
import os

# Network
#       o  flower type
#      / \  w1, w2, b
#     o   o  length, width

'''
Samle data for flowers
Length of leaf
Width of leaf
Color of grap 0,1 whic are red or blue flower
''''
data = [[3,1.5,1],[2,1,0],[4,1.5,1],[3,1,0],[3.5,0.5,1],[2,0.5,0],[5.5,1,1],[1,1,0]]
flowers = {1:'r', 0:'b'}

def predict(m1,m2,w1,w2,b, derrivative = False):
    x = (m1*w1) + (m2*w2) + b
    if derrivative:
        return sigmoid_d(x)
    return sigmoid(x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derrivative of sigmoid function
def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))

def show_data_on_graph(data):
    pyplot.grid()
    # pyplot.axis([0, 6, 0, 6])
    for i in range(len(data)):
        x = data[i][0]
        y = data[i][1]
        color = flowers[data[i][2]]
        pyplot.scatter(x, y, c= color)
    pyplot.show()

def show_sigmoid_on_graph():
    x = np.linspace(-10, 10, 100)
    y = sigmoid(x)
    y_d = sigmoid_d(x)
    pyplot.plot(x, y)
    pyplot.plot(x, y_d, c='r')
    pyplot.show()

def show_costs_on_graph(costs):
    pyplot.plot(costs)
    pyplot.show()

def fit(data, step = 0.1, iteration = 10000):
    #random init of weights
    w1 = np.random.randn()
    w2 = np.random.randn()
    b = np.random.randn()
    costs = [] # keep costs during training, see if they go down

    for i in range(iteration):
        random_index = np.random.randint(len(data))
        point = data[random_index]
        prediction = predict(point[0], point[1], w1, w2, b)
        expectaion = point[2]
        cost = (prediction - expectaion) ** 2

        # Just for cost visualization
        if i % 100 == 0:
            total_cost = 0
            for d in data:
                d_prediction = predict(d[0], d[1], w1, w2, b)
                d_expectation = d[2]
                total_cost += np.square(d_prediction - d_expectation)
            costs.append(total_cost)

        # Derrivatives
        cost_d = 2 * (prediction - expectaion)
        prediction_d = predict(point[0], point[1], w1, w2, b, True)
        w1_d = point[0]
        w2_d = point[1]
        b_d = 1

        w1 = w1 - step * cost_d * prediction_d * w1_d
        w2 = w2 - step * cost_d * prediction_d * w2_d
        b = b - step * cost_d * prediction_d * b_d
    return costs, w1, w2, b

costs, w1, w2, b = fit(data)

def test(data, w1, w2, b):
    for x in np.linspace(0, 6, 20):
        for y in np.linspace(0, 3, 20):
            prediction = predict(x, y, w1, w2, b)
            # print('Data: ',d, 'Prediction: ', prediction)
            color = 'b'
            if prediction > 0.5:
                color = 'r'
            pyplot.scatter([x], [y], c= color, alpha=0.2)
    show_data_on_graph(data)

test(data, w1, w2, b)
