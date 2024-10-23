import time
import numpy as np
import matplotlib.pyplot as plt
from neuralNetworkFunctions import *

class NeuralNetwork:
    def __init__(self):
        pass

    def initNn(self, xLength, l1, outLength):
        self.w1 = np.random.random((l1, xLength))
        self.w2 = np.random.random((outLength, l1))

        self.shape = {
            "inputs": xLength,
            "hidden layer neurons": l1,
            "outputs": outLength
        }

    def plotTraining(self):
        plt.plot(self.generations, self.errors)
        plt.xlabel("Generations")
        plt.ylabel("Error")
        plt.show()


    def forward(self, x):
        z1 = np.dot(self.w1, x)

        a1 = relu(z1)

        z2 = np.dot(self.w2, a1)

        a2 = sigmoid(z2)

        return z1, z2, a1, a2


    def getGradients(self, x, y, z1, z2, a1, a2):
        delta2 = squaredErrorDerivative(a2, y) * sigmoidDerivative(a2)  # sigmoid derivative takes sigmoid input, so a2

        a1 = a1.reshape(4, 1)

        gradient2 = np.dot(delta2, a1.T)
        a1_grad = np.dot(self.w2.T, delta2)

        delta1 = a1_grad * reluDerivative(z1)
        delta1 = delta1.reshape(-1, 1)

        gradient1 = np.dot(delta1, x.reshape(1, -1))

        return gradient2, gradient1


    def updateWeights(self, gradient1, gradient2, lr):
        self.w1 -= gradient1 * self.lr
        self.w2 -= gradient2 * self.lr


    def fit(self, lr gens):
        print(f"Training for {self.gens} generations...")

        x = np.random.random((self.shape["inputs"],))

        y = np.array([1])

        self.errors = []

        self.generations = list(range(self.gens))

        startTime = time.time()

        for _ in range(gens):
            z1, z2, a1, a2 = self.forward(x)

            gradient2, gradient1 = self.getGradients(x, y, z1, z2, a1, a2)

            self.updateWeights(gradient1, gradient2, np.array([lr]))

            self.errors.append(squaredError(a2, y))

        self.fitTime = round(time.time() - startTime, 4)
