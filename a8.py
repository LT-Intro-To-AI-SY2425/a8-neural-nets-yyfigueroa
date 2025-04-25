from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_training_data = [([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]


xorn = NeuralNet(2, 2, 1)
xorn.train(xor_training_data, iters=10000, print_interval=1000)
print("Final XOR test with 2 hidden nodes:")
print(xorn.test_with_expected(xor_training_data))


xorn8 = NeuralNet(2, 8, 1)
xorn8.train(xor_training_data, iters=10000, print_interval=1000)
print("Final XOR test with 8 hidden nodes:")
print(xorn8.test_with_expected(xor_training_data))


xorn1 = NeuralNet(2, 1, 1)
xorn1.train(xor_training_data, iters=10000, print_interval=1000)
print("Final XOR test with 1 hidden node:")
print(xorn1.test_with_expected(xor_training_data))

# politics
party_training_data = [([0.9, 0.6,0.8,0.3,0.1], [1.0]), ([0.8 , 0.8, 0.4, 0.6, 0.4], [1.0]), ([0.7, 0.2, 0.4, 0.6, 0.3], [1.0]), ([0.5, 0.5, 0.8, 0.4, 0.8], [0.0]), ([0.3, 0.1, 0.6, 0.8, 0.8], [0.0]), ([0.6, 0.3, 0.4, 0.3, 0.6], [0.0])]
party_testing_data = [([1.0, 1.0, 1.0, 0.1, 0.1], []), ([0.5, 0.2, 0.1, 0.7, 0.7], []), ([0.8,0.3,0.3,0.3,0.8], []), ([0.8,0.3,0.3,0.8,0.3], []), ([0.9,0.8,0.8,0.3,0.6], [])]

PoliticalNet = NeuralNet(5, 2, 1)