from neuralNetwork import NeuralNetwork

nn = NeuralNetwork(0.1, 10 ** 5)

nn.initNn(10, 4, 1)

nn.fit()

print(f"Time taken: {nn.fitTime} seconds")

nn.plotTraining()
