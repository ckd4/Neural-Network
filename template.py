import numpy as np
np.set_printoptions(suppress=True)

test_input = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1], [0, 1, 0, 0]])
test_output = np.array([[1, 0, 1, 0, 1, 0]]).T
synaptic_weight = 2 * np.random.random((4, 1)) - 1

def activation(x):
    # Sigmoid activate function
    return 1 / (1 + np.exp(-x))

input_layer = test_input
output_layer = activation(np.dot(input_layer, synaptic_weight))

print(f"Start weights:\n{synaptic_weight}")

# Learning...
for i in range(100000):
    input_layer = test_input
    output_layer = activation(np.dot(input_layer, synaptic_weight))
    err = test_output - output_layer
    adj = np.dot(test_input.T, err * (output_layer * (1 - output_layer)))
    synaptic_weight += adj

print(f"Trained weights:\n{synaptic_weight}")
print(f"Training result:\n{output_layer}")
print("\n--------------------------------------------------\n\n")

while True:
    data = np.asarray(input("New data: ").split(" "), dtype=float)
    print(activation(np.dot(data, synaptic_weight)))
