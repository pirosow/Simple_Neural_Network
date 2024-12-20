from neuralNetwork import NeuralNetwork

nn = NeuralNetwork()

nn.initNn(10, 4, 2)

nn.fit(lr=0.1, gens=10000)

print(f"Time taken: {nn.fitTime} seconds")

nn.plotTraining()
