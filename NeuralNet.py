import autograd.numpy as np
from autograd import grad
from time import perf_counter
from matplotlib import pyplot as plt

#Christian Eid

# Sample Code at the bottom

# Structure of code

# 1. generate_net creates list the net in list structure
# 2. Sigmoid neural net functions
# activation function, and derivative
# Activate layer & forward propogate
# Backpropogate
# Run the neural net
# 3. ReLU neural net functions
# 4. Gaussian Kernelized Soft-SVM
# 5. Testing
# 6. Sample Code


def generate_net(inputs, hidden_length):
    hidden_1 = []
    hidden_2 = []
    output = [0, 1]
    weights_1 = []
    weights_2 = []
    weights_3 = []

    for i in range(hidden_length):
        list = []
        for j in range(len(inputs[0])):
            list.append(np.random.uniform(-0.3, 0.3))
        weights_1.append(list)

    for i in range(hidden_length):
        list = []
        for j in range(hidden_length + 1):
            list.append(np.random.uniform(-0.3, 0.3))
        weights_2.append(list)

    for i in range(hidden_length + 1):
        hidden_1.append(1.0)
        hidden_2.append(1.0)
        weights_3.append(np.random.uniform(-0.3, 0.3))

    return hidden_1, weights_1, hidden_2, weights_2, weights_3, output


# SIGMOID NEURAL NET
def sigmoid(x):
    return (1 / (1 + np.power(np.e, -x)))


def sigmoid_dx(x):
    return (sigmoid(x) * (1 - sigmoid(x)))


def activate_layer(layer, weights, inputs):
    layer_net = []
    for i in range(len(layer) - 1):
        if isinstance(weights[0], float):
            layer_net.append(np.dot(inputs, weights))
            layer[i] = sigmoid(np.dot(inputs, weights))
        else:
            layer_net.append(np.dot(inputs, weights[i]))
            layer[i] = sigmoid(np.dot(inputs, weights[i]))
    return layer, layer_net


def propogate_forward(layer1, layer2, output, w1, w2, w3, input):
    layer1, layer1_net = activate_layer(layer1, w1, input)
    layer2, layer2_net = activate_layer(layer2, w2, layer1)
    output, output_net = activate_layer(output, w3, layer2)

    return layer1, layer1_net, layer2, layer2_net, output, output_net


def delta_output(y, a):
    return -(y - a)


def backpropogate(instance, y, layer1, layer2, output, weights1, weights2, weights3, layer1_net, layer2_net, step_size):
    # TOTAL LOSS AND UPDATES TO WEIGHTS LEADING TO OUTPUT
    output_error = delta_output(y, output[0])

    layer2_error = []
    for i in range(len(weights3) - 1):
        layer2_error.append(weights3[i] * output_error * sigmoid_dx(layer2_net[i]))
        weights3[i] -= step_size * output_error * layer2[i]
    weights3[len(weights3) - 1] -= step_size * output_error

    # UPDATES TO 2ND LAYER WEIGHTS
    for i in range(len(weights2)):
        for j in range(len(weights2[0])):
            weights2[i][j] -= step_size * layer2_error[i] * layer1[j]

    layer1_error = []
    for i in range(len(layer2) - 1):
        layer1_error.append(0)
        for j in range(len(weights2)):
            layer1_error[i] += layer2_error[j] * weights2[j][i]
        layer1_error[i] *= sigmoid_dx(layer1_net[i])

    # LAYER 1 Weights
    for i in range(len(layer1) - 1):
        for j in range(len(instance)):
            weights1[i][j] -= step_size * layer1_error[i] * instance[j]

    return weights1, weights2, weights3, output_error


def run_neural_net(instances, y, size, step_size, stop_limit):
    # Stop limit is the number of times to go through all training instances
    # The function creates a neural net
    layer1, weights_1, layer2, weights_2, weights_3, output = generate_net(instances, size)

    count = 0
    i = 0

    while i <= (len(instances) - 1):
        # For each training instance ,it passes the instance forward
        # Then backpropogates and updates the weights
        layer1, layer1_net, layer2, layer2_net, output, output_net = propogate_forward(layer1, layer2, output,
                                                                                       weights_1, weights_2, weights_3,
                                                                                       instances[i])
        weights_1, weights_2, weights_3, loss = backpropogate(instances[i], y[i], layer1, layer2, output, weights_1,
                                                              weights_2, weights_3, layer1_net, layer2_net, step_size)

        # The following tells the network to keep going after going through all training data
        if i == (len(instances) - 1):
            count += 1
            if count != stop_limit:
                i = 0
        i += 1

    # It ends by printing the network
    print_network(instances[0], weights_1, weights_2, weights_3, layer1, layer2, output)

    return layer1, layer2, weights_1, weights_2, weights_3, count


# HELPERS

# Get ouptut runs the neural net on a single instance, and returns the output
def get_output(instance, layer1, layer2, weights1, weights2, weights3):
    output = [0, 0]
    layer1, layer1_net, layer2, layer2_net, output, output_net = propogate_forward(layer1, layer2, output, weights1,
                                                                                   weights2, weights3, instance)
    return output[0]


def get_output_ReLU(instance, layer1, layer2, weights1, weights2, weights3):
    output = [0, 0]
    layer1, layer1_net, layer2, layer2_net, output, output_net = propogate_ReLU(layer1, layer2, output, weights1,
                                                                                weights2, weights3, instance)
    return output[0]


# Test network uses get_output to test all training instances
def test_network_ReLU(instances, layer1, layer2, w1, w2, w3, y):
    success_rate = 0
    for i in range(len(instances)):
        a_o = get_output_ReLU(instances[i], layer1, layer2, w1, w2, w3)
        if a_o > 0.5:
            res = 1
        else:
            res = 0
        if res == y[i]:
            success_rate += 1
    success_rate *= 1 / len(instances) * 100
    return success_rate


def test_network(instances, layer1, layer2, w1, w2, w3, y):
    success_rate = 0
    for i in range(len(instances)):
        a_o = get_output(instances[i], layer1, layer2, w1, w2, w3)
        if a_o > 0.5:
            res = 1
        else:
            res = 0
        if res == y[i]:
            success_rate += 1
    success_rate *= 1 / len(instances) * 100
    return success_rate


def print_network(instances, w1, w2, w3, layer1, layer2, output):
    print("OUTPUT: ", output[0])
    print("w3: ", w3)
    print("layer 2: ", layer2)
    print("w2: ", w2)
    print("layer 1: ", layer1)
    print("w1: ", w1)
    print("instances: ", instances)


# RELU NET FUNCTIONS
def ReLU(x):
    return max(0, x)


def ReLU_dx(x):
    if x <= 0:
        return 0
    else:
        return 1


def activate_ReLU(layer, weights, inputs):
    layer_net = []
    for i in range(len(layer) - 1):
        # if isinstance(weights[0], float):
        #     layer_net.append(np.dot(inputs, weights))
        #     layer[i] = ReLU(np.dot(inputs, weights))
        #     print(layer[i])
        # else:
        layer_net.append(np.dot(inputs, weights[i]))
        layer[i] = ReLU(np.dot(inputs, weights[i]))

    return layer, layer_net


def propogate_ReLU(layer1, layer2, output, w1, w2, w3, input):
    layer1, layer1_net = activate_ReLU(layer1, w1, input)
    layer2, layer2_net = activate_ReLU(layer2, w2, layer1)
    output, output_net = activate_layer(output, w3, layer2)
    return layer1, layer1_net, layer2, layer2_net, output, output_net


def backpropogate_ReLU(instance, y, layer1, layer2, output, weights1, weights2, weights3, layer1_net, layer2_net,
                       step_size):
    # TOTAL LOSS AND UPDATES TO WEIGHTS LEADING TO OUTPUT
    output_error = delta_output(y, output[0])
    layer2_error = []
    for i in range(len(weights3) - 1):
        layer2_error.append(weights3[i] * output_error * ReLU_dx(layer2[i]))

        weights3[i] -= step_size * output_error * layer2[i]
    weights3[len(weights3) - 1] -= step_size * output_error

    # UPDATES TO 2ND LAYER WEIGHTS
    for i in range(len(weights2)):
        for j in range(len(weights2[0])):
            weights2[i][j] -= step_size * layer2_error[i] * layer1[j]

    layer1_error = []
    for i in range(len(layer2) - 1):
        layer1_error.append(0)

        for j in range(len(weights2)):
            layer1_error[i] += layer2_error[j] * weights2[j][i]
        layer1_error[i] *= ReLU_dx(layer1_net[i])

    # LAYER 1 Weights
    for i in range(len(layer1) - 1):
        for j in range(len(instance)):
            weights1[i][j] -= step_size * layer1_error[i] * instance[j]

    return weights1, weights2, weights3, output_error


def run_ReLU_net(instances, y, size, step_size, stop_limit):
    # Creates net
    layer1, weights_1, layer2, weights_2, weights_3, output = generate_net(instances, size)

    count = 0
    i = 0
    while i <= (len(instances) - 1):

        layer1, layer1_net, layer2, layer2_net, output, output_net = propogate_ReLU(layer1, layer2, output, weights_1,
                                                                                    weights_2, weights_3, instances[i])

        weights_1, weights_2, weights_3, loss = backpropogate_ReLU(instances[i], y[i], layer1, layer2, output,
                                                                   weights_1, weights_2, weights_3, layer1_net,
                                                                   layer2_net, step_size)

        if i == (len(instances) - 1):
            count += 1
            if count != stop_limit:
                i = 0
        i += 1

    print_network(instances[0], weights_1, weights_2, weights_3, layer1, layer2, output)

    return layer1, layer2, weights_1, weights_2, weights_3, count


# GAUSSIAN SVM

def norm(w):
    return np.sqrt(np.dot(w, w))


def Gaussian(x, z):
    scale = 10
    return np.power(np.e, (norm(x-z)**2)/(2*scale))

def gaussian_sum(w, x, i):
    res = 0.0
    for j in range(100):
        res += w[j] * Gaussian(x[j], x[i])
    return res


def gaussian_hinge(w, x, y):
    res = 0.0
    for i in range(100):
        res += max(0, 1 - y[i]*(gaussian_sum(w, x, i) )+ 0*(norm(w))**2)
    res *= 1/100
    return res

def generate_gaussian_loss(x, y):
    LS_gaussian_loss = lambda w : gaussian_hinge(w, x, y)
    return grad(LS_gaussian_loss)

def gradient_descent_gaussian(x, y, step_size):
    gradient_gaussian = generate_gaussian_loss(x, y)

    #NEEDS TO BE 100 long
    w = []
    for i in range(100):
        w.append(np.random.uniform(-2, 2))
    alpha = np.array(w)
    i = 0
    while i < 100:
        alpha = alpha - step_size * gradient_gaussian(alpha)
        i += 1

    return alpha
#Implementing Classification using Alpha vector



#Testing the success rate of Kernalized functions


def test_kernel_Gaussian(alpha, instances, y):
    success_rate = 0
    test_y = []
    for i in range(100):
        test_y.append(classify_Gaussian(alpha, instances, instances[i]))


    for i in range(100):
        if test_y[i] == y[i]:
            success_rate += 1
    return success_rate

def classify_Gaussian(alpha, instances, instance):
    res = 0
    for i in range(100):
        res += alpha[i]*Gaussian(instances[i], instance)
    if res > 0:
        return 1
    else:
        return -1


# TRAINING DATA
def generate_x(n):
    x = []
    for i in range(n):
        x.append([1, np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
    return x


def generate_c():
    return np.array(
        [np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2),
         np.random.uniform(-2, 2), np.random.uniform(-2, 2)])


def generate_w():
    return np.array([np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2)])


def generate_y(x, c):
    y = []
    for i in range(len(x)):
        if np.dot(c, x[i]) > 0:
            y.append(1)
        else:
            y.append(0)
    return y


# Generate nonlinear y uses the feature space function
def generate_nonlinear_y(x, c):
    fs = generate_feature_space(x)
    return generate_y(fs, c)


def generate_feature_space(x):
    res = []
    for i in range(len(x)):
        list = [x[i][0], x[i][1], x[i][2], x[i][1] * x[i][2], x[i][1] ** 2, x[i][2] ** 2]
        res.append(list)
    return res


def make_predictions(layer1, layer2, w1, w2, w3, number_of_predictions, w):
    success_rate = 0

    x = generate_x(number_of_predictions)
    y = generate_y(x, w)

    for i in range(len(x)):
        a_o = get_output(x[i], layer1, layer2, w1, w2, w3)
        if a_o > 0.5:
            res = 1
        else:
            res = 0
        if res == y[i]:
            success_rate += 1
    success_rate *= 1 / len(x) * 100
    print('Success rate of predicting new', number_of_predictions, ' instances: ', success_rate)
    return success_rate


# TEST 1 - Constant Iterations Rate
def test_1():
    sigmoid_success = 0
    sigmoid_time = 0

    ReLU_success = 0
    ReLU_time = 0

    gaussian_success = 0
    gaussian_time = 0

    cycles = 10
    for i in range(cycles):
        training_set = np.array(generate_x(100))
        c_weight_vector = generate_c()

        # the following creates non-linearly classifyable data with feature space in the nonlinear_y function.
        non_linear_labels = generate_nonlinear_y(training_set, c_weight_vector)

        start1 = perf_counter()
        l1, l2, w1, w2, w3, sigmoid_i = run_neural_net(training_set, non_linear_labels, 4, 0.1, 10)
        stop1 = perf_counter()
        sigmoid_success += test_network(training_set, l1, l2, w1, w2, w3, non_linear_labels)
        sigmoid_time += stop1 - start1

        start2 = perf_counter()
        rl1, rl2, rw1, rw2, rw3, ReLU_i = run_ReLU_net(training_set, non_linear_labels, 4, 0.1, 10)
        stop2 = perf_counter()
        ReLU_success += test_network_ReLU(training_set, rl1, rl2, rw1, rw2, rw3, non_linear_labels)
        ReLU_time += stop2 - start1

        start3 = perf_counter()
        gaussian_alpha, gaussian_i = gradient_descent_gaussian(training_set, non_linear_labels, 50.0)
        stop3 = perf_counter()
        gaussian_success += test_kernel_Gaussian(gaussian_alpha, training_set, non_linear_labels)
        gaussian_time += stop3 - start3

    sigmoid_success *= 1 / cycles
    sigmoid_time *= 1 / cycles
    ReLU_success *= 1 / cycles
    ReLU_time *= 1 / cycles

    gaussian_success *= 1 / cycles
    gaussian_time *= 1 / cycles
    print('sigmoid success average : ', sigmoid_success)
    print('ReLU success average : ', ReLU_success)
    print('gaussian success average : ', gaussian_success)

    data = [sigmoid_success, ReLU_success, gaussian_success, sigmoid_time, ReLU_time, gaussian_time]
    for el in data:
        print(el)


# test_1()


def test_len():
    sigmoid_successes = []
    sigmoid_times = []
    relu_successes = []
    relu_times = []
    length_range = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    cycles = 25

    for i in range(9):
        sigmoid_successes.append(0)
        relu_successes.append(0)

    for j in range(cycles):
        training_set = np.array(generate_x(100))
        weight_vector = generate_w()
        c_weight_vector = generate_c()
        non_linear_labels = generate_nonlinear_y(training_set, c_weight_vector)

        for i in range(len(length_range)):
            start1 = perf_counter()
            l1, l2, w1, w2, w3, sigmoid_i = run_neural_net(training_set, non_linear_labels, i, 0.1, 10)
            stop1 = perf_counter()
            sigmoid_successes[i] += stop1 - start1
            start2 = perf_counter()
            rl1, rl2, rw1, rw2, rw3, ReLU_i = run_ReLU_net(training_set, non_linear_labels, i, 0.1, 10)
            stop2 = perf_counter()
            relu_successes[i] += stop2 - start2

    for i in range(9):
        print('len: ', length_range[i], 'sigmoid: ', sigmoid_successes[i], 'relu: ', relu_successes[i])

    for i in range(9):
        sigmoid_successes[i] *= 1 / cycles
        relu_successes[i] *= 1 / cycles

    plt.plot(length_range, sigmoid_successes, c='blue')
    plt.plot(length_range, relu_successes, c='green')
    plt.title("Comparing Success of Hidden Lengths")

    plt.show()


# test_len()


def sample():
    x = np.array(generate_x(100))
    c = generate_c()

    y = generate_nonlinear_y(x, c)
    print("SIGMOID NEURAL NET: ")
    # FORM:                          run_neural_net(instances, lables, hidden_layer_length, step_size, iterations)
    l1, l2, w1, w2, w3, sigmoid_i = run_neural_net(x, y, 7, 1, 100)
    print('**************')
    print("ReLU NEURAL NET: ")
    rl1, rl2, rw1, rw2, rw3, relu_i = run_ReLU_net(x, y, 7, 0.1, 100)


    print('Success of sigmoid neural net: ', test_network(x, l1, l2, w1, w2, w3, y))
    print('Success of ReLU neural net: ', test_network_ReLU(x, rl1, rl2, rw1, rw2, rw3, y))


sample()
