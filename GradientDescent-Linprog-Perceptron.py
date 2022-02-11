from autograd import grad
import autograd.numpy as np
from time import perf_counter

from matplotlib import pyplot as plt
from scipy.optimize import linprog

#Christian Eid

#SAMPLE CODE AT THE END THAT RUNS EVERYTHING ONCE

#PART 1: Creating Training Data

def generate_w():
    return np.array([np.random.uniform(-1, 1), np.random.uniform(-5, 5), np.random.uniform(-5, 5)])

def generate_x():
    x = []
    for i in range(100):
        x.append([1, np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
    return x

def generate_y(x, w):
    y = []
    for i in range(100):
        if np.dot(w, x[i])> 0:
            y.append(1)
        else:
            y.append(0)
    return y

def generate_y_negative(x, w):
    y = []
    for i in range(100):
        if np.dot(w, x[i])> 0:
            y.append(1)
        else:
            y.append(-1)
    return y


#Tests to determine the Success of Algorithms
#Returns percent of training samples the learned weight vector correctly classifies

def test_algorithm(x, y, w):
    success_rate = 0
    test_y = np.array(generate_y(x, w))
    for i in range(100):
        if test_y[i] == y[i]:
            success_rate += 1
    return success_rate

def test_algorithm_softmax(x, y, w):
    success_rate = 0
    test_y = np.array(generate_y_negative(x, w))
    for i in range(100):
        if test_y[i] == y[i]:
            success_rate += 1
    return success_rate




#PART 2: Loss Functions & Logistic Function

#Logistic
def a(x):
    t = 1
    return (1 / (1 + np.power(np.e, t*-x)))

#Least Squares Loss
def least_squares(w, x, y):
    result = 0
    t = 1
    for i in range(len(x)):
        result += (a(np.dot(x[i], w))-y[i])**2
    result = result/len(x)
    return result

# #Cross Entropy
def cross_entropy(w, x, y):
    result = 0
    for i in range(len(x)):
        result += y[i] * np.log(a(np.dot(x[i], w))) + (1 - y[i]) * np.log(1 - a(np.dot(x[i], w)))
    result = result / -len(x)
    return result

#Softmax
def softmax(w, x, y):
    result = 0.0
    for i in range(100):
        result += np.log(1 + np.power(np.e, -y[i] * np.dot(x[i], w)))
    result = result / 100
    return result

#Getting the gradients
def get_gradient_ls(x, y):
    LS_least_squares = lambda w: least_squares(w, x, y)
    return grad(LS_least_squares)

def get_gradient_ce(x, y):
    LS_cross_entropy = lambda w: cross_entropy(w, x, y)
    return grad(LS_cross_entropy)

def get_gradient_sm(x, y):
    LS_soft_max = lambda w: softmax(w, x, y)
    return grad(LS_soft_max)





#Gradient Descents Part 1 -- No Stopping Criteria, each algorithm runs 100 iterations

def gradient_descent_least_squares(x, y, step_size):
    gradient_least_squares = get_gradient_ls(x, y)
    w = np.array([np.random.uniform(-1, 1), np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
    i = 0
    while i < 100:
        w = w - step_size*gradient_least_squares(w)
        i+=1
    return w

def gradient_descent_cross_entropy(x, y, step_size):
    gradient_cross_entropy = get_gradient_ce(x, y)
    w = np.array([np.random.uniform(-1, 1),np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
    i = 0
    while i < 100:
        w = w - step_size*gradient_cross_entropy(w)
        i += 1
    return w

def gradient_descent_soft_max(x, y, step_size):
    gradient_soft_max = get_gradient_sm(x, y)
    w = np.array([np.random.uniform(-1, 1),np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
    i = 0
    while i < 100:
        w = w - step_size*gradient_soft_max(w)
        i += 1

    return w



#PART 3 : Perceptron Algorithm
def perceptron(x, y):
    w = np.array([0.0, 0.0, 0.0])
    i = 0
    iteration_count = 0
    fire_count = 0
    while i < 100:
        iteration_count+=1
        if y[i] * np.dot(x[i], w) <= 0:
            fire_count += 1
            w += y[i]*x[i]
            i = 0
        i+=1
    return w, iteration_count, fire_count



#Part 4: Linprog

def run_linprog(x, y):
    objective = [0, 0, 0]
    inequalities, inequality_rhs = make_inequalities(x, y)
    start = perf_counter()
    result = linprog(objective, inequalities, inequality_rhs, bounds=(None, None))
    print(result)
    stop = perf_counter()

    return (stop-start, test_algorithm_softmax(x, y, result.x), result.nit)

def make_inequalities(x, y):
    res = []
    rhs = []
    for i in range(100):
        if y[i] == 1.0:
            res.append([-x[i][0], -x[i][1], -x[i][2]])
        else:
            res.append([x[i][0], x[i][1], x[i][2]])
        rhs.append(-1)
    return res, rhs


#Part 5: TESTING THE ALGORITHMS

#The first test keeps track of the number of training samples missed after 100 iterations of gradient descent,
#and the amount of time it took.
def time_misclassify_test():
    #INITIALIZING DATA TO COLLECT
    cycles = 100
    print("Number of Cycles: ", cycles)


    least_squares_time = 0
    least_squares_success = 0

    cross_entropy_time = 0
    cross_entropy_success = 0

    soft_max_time = 0
    soft_max_success = 0

    perceptron_time = 0
    perceptron_success = 0

    linprog_time = 0
    linprog_success = 0

    for i in range(cycles):
        print("At cycle", i)

        #ESTABLISHING A NEW TRAINING SET
        w = np.array([np.random.uniform(-1, 1), np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
        x = np.array(generate_x())
        y = np.array(generate_y(x, w))
        y_negative = np.array(generate_y_negative(x, w))
        step_size = 0.1

        start_1 = perf_counter()
        least_squares_w = gradient_descent_least_squares(x, y, step_size)
        stop_1 = perf_counter()

        least_squares_time += (stop_1 - start_1)
        least_squares_success += test_algorithm(x, y, least_squares_w)

        #Cross Entropy Test
        start_2 = perf_counter()
        cross_entropy_w = gradient_descent_cross_entropy(x, y, step_size)
        stop_2 = perf_counter()

        cross_entropy_time += (stop_2 - start_2)
        cross_entropy_success += test_algorithm(x, y, cross_entropy_w)

        #Soft Max Test
        start_3 = perf_counter()
        softmax_w = gradient_descent_soft_max(x, y_negative, step_size)
        stop_3 = perf_counter()

        soft_max_time += (stop_3-start_3)
        soft_max_success += test_algorithm_softmax(x, y_negative, softmax_w)

        #Perceptron Test
        start_4 = perf_counter()
        perceptron_w, count, fire = perceptron(x, y_negative)
        stop_4 = perf_counter()
        perceptron_time += (stop_4-start_4)
        perceptron_success += test_algorithm_softmax(x, y_negative, perceptron_w)

        lp_time, lp_success, lp_iterations = run_linprog(x, y_negative)
        linprog_time += lp_time
        linprog_success += lp_success


    least_squares_time *= 1/cycles
    least_squares_success *= 1/cycles
    cross_entropy_time *= 1/cycles
    cross_entropy_success *= 1/cycles
    soft_max_time *= 1/cycles
    soft_max_success *= 1/cycles
    perceptron_time *= 1/cycles
    perceptron_success *= 1/cycles
    linprog_time *= 1/cycles
    linprog_success *= 1/cycles

    return least_squares_time, least_squares_success, cross_entropy_time, cross_entropy_success, soft_max_time, soft_max_success, perceptron_time, perceptron_success, linprog_time, linprog_success


#Returning the data
def print_data():
    least_squares_time, least_squares_success, cross_entropy_time, cross_entropy_success, soft_max_time, soft_max_success, perceptron_time, perceptron_success, linprog_time, linprog_success = time_misclassify_test()

    print("Average Time")
    print("Least Squares: ", least_squares_time)
    print("Cross Entropy: ", cross_entropy_time)
    print("Soft Max: ", soft_max_time)
    print("Perceptron: ", perceptron_time)
    print("Linprog: ", linprog_time)

    print("*****************")
    print("Average Success Rate")
    print("Least Squares: ", least_squares_success)
    print("Cross Entropy: ", cross_entropy_success)
    print("Soft Max: ", soft_max_success)
    print("Perceptron: ", perceptron_success)
    print("Linprog: ", linprog_success)


    data = [least_squares_time,cross_entropy_time,soft_max_time, perceptron_time,linprog_time, least_squares_success, cross_entropy_success,  soft_max_success, perceptron_success, linprog_success]
    for val in data:
        print(val)







# Test 2 - Changing the Stopping Criteria & Testing time / number of iterations
# Each Gradient Descent algorithm runs until it is 100% accurate
# If it reaches 500 iterations without being 100%, it stops
# # Step size is 1 here


#Creating new gradient descents with stopping condition
def gradient_descent_least_squares_100(x, y, step_size):
    gradient_least_squares = get_gradient_ls(x, y)
    w = np.array([np.random.uniform(-1, 1), np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
    i = 0
    while test_algorithm(x, y, w) != 100:
        w = w - step_size*gradient_least_squares(w)
        i += 1
        if i == 500:
            print('least squares at 500', test_algorithm(x, y, w))
            return w, i
    return w, i

def gradient_descent_cross_entropy_100(x, y, step_size):
    gradient_cross_entropy = get_gradient_ce(x, y)
    w = np.array([np.random.uniform(-1, 1),np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
    i = 0
    while test_algorithm(x, y, w) != 100:

        w = w - step_size * gradient_cross_entropy(w)
        i += 1
        if i == 500:
            print('cross entropy at 500', test_algorithm(x, y, w))
            return w, i
    return w, i

def gradient_descent_soft_max_100(x, y, step_size):
    gradient_soft_max = get_gradient_sm(x, y)
    w = np.array([np.random.uniform(-1, 1),np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
    i = 0
    while test_algorithm_softmax(x, y, w) != 100:

        w = w - step_size * gradient_soft_max(w)
        i += 1
        if i == 500:
            print('soft max at 500', test_algorithm_softmax(x, y, w))

            return w, i
    return w, i

#Here is the test
def time_misclassify_test():
    #INITIALIZING DATA TO COLLECT
    cycles = 50
    print("Number of Cycles: ", cycles)


    least_squares_time = 0
    least_squares_iterations = 0

    cross_entropy_time = 0
    cross_entropy_iterations = 0

    soft_max_time = 0
    soft_max_iterations= 0

    perceptron_time = 0
    perceptron_iterations = 0
    perceptron_fire  = 0

    linprog_time = 0
    linprog_iterations  = 0

    for i in range(cycles):
        print("At cycle", i)

        #ESTABLISHING A NEW TRAINING SET
        w = np.array([np.random.uniform(-1, 1), np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
        x = np.array(generate_x())
        y = np.array(generate_y(x, w))
        y_negative = np.array(generate_y_negative(x, w))
        step_size = 1.0

        #Least Squares Test
        start_1 = perf_counter()
        least_squares_w, ls_i = gradient_descent_least_squares_100(x, y, step_size)
        stop_1 = perf_counter()

        least_squares_time += (stop_1 - start_1)
        least_squares_iterations += ls_i

        #Cross Entropy Test
        start_2 = perf_counter()
        cross_entropy_w, ce_i = gradient_descent_cross_entropy_100(x, y, step_size)
        stop_2 = perf_counter()

        cross_entropy_time += (stop_2 - start_2)
        cross_entropy_iterations += ce_i

        #Soft Max Test
        start_3 = perf_counter()
        softmax_w, sm_i = gradient_descent_soft_max_100(x, y_negative, step_size)
        stop_3 = perf_counter()

        soft_max_time += (stop_3-start_3)
        soft_max_iterations += sm_i

        #Perceptron Test
        start_4 = perf_counter()
        perceptron_w, count, fire = perceptron(x, y_negative)
        stop_4 = perf_counter()
        perceptron_time += (stop_4-start_4)
        perceptron_iterations += count
        perceptron_fire += fire

        lp_time, lp_success, lp_iterations = run_linprog(x, y_negative)
        linprog_time += lp_time
        linprog_iterations+= lp_iterations

    least_squares_time *= 1/cycles
    least_squares_iterations *= 1/cycles
    cross_entropy_time *= 1/cycles
    cross_entropy_iterations *= 1/cycles
    soft_max_time *= 1/cycles
    soft_max_iterations *= 1/cycles
    perceptron_time *= 1/cycles
    perceptron_iterations *= 1/cycles
    perceptron_fire *= 1/cycles
    linprog_time *= 1 / cycles
    linprog_iterations *= 1 / cycles

    print('perceptron updates weights an average of ', perceptron_fire, 'times')



    return least_squares_time, least_squares_iterations, cross_entropy_time, cross_entropy_iterations, soft_max_time, soft_max_iterations, perceptron_time, perceptron_iterations, linprog_time, linprog_iterations


def print_classify_data():
    least_squares_time, least_squares_iterations, cross_entropy_time, cross_entropy_iterations, soft_max_time, soft_max_iterations, perceptron_time, perceptron_iterations, linprog_time, linprog_iterations = time_misclassify_test()

    print("Average Time")
    print("Least Squares: ", least_squares_time)
    print("Cross Entropy: ", cross_entropy_time)
    print("Soft Max: ", soft_max_time)
    print("Perceptron: ", perceptron_time)
    print("Linprog: ", linprog_time)
    print("*****************")
    print("Average Iterations ")
    print("Least Squares: ", least_squares_iterations)
    print("Cross Entropy: ", cross_entropy_iterations)
    print("Soft Max: ", soft_max_iterations)
    print("Perceptron: ", perceptron_iterations)
    print("Linprog: ", linprog_iterations)
    data = [least_squares_time,cross_entropy_time,soft_max_time, perceptron_time, linprog_time,least_squares_iterations, cross_entropy_iterations, soft_max_iterations, perceptron_iterations, linprog_iterations ]
    for val in data:
        print(val)




#Part 6: Plotting


def divide_x(x,y):
    blue_x = []
    blue_y = []
    green_x = []
    green_y =[]
    for i in range(100):
        if y[i] == 1:
            blue_x.append(x[i][1])
            blue_y.append(x[i][2])
        else:
            green_x.append(x[i][1])
            green_y.append(x[i][2])
    return blue_x, blue_y, green_x, green_y

def make_plot():
    x = generate_x()
    w = generate_w()
    y = generate_y(x, w)
    y_negative = generate_y_negative(x, w)
    x1, y1, x2, y2 = divide_x(x,y_negative)
    plt.scatter(x1, y1, c = 'blue')
    plt.scatter(x2, y2, c = 'green')
    plt.quiver(0, 0, w[1], w[2], units = 'xy', scale = 1 )
    plt.title("Linearly Seperable Data")

    #point (0, -b/w2)
    plt.show()

#SAMPLE CODE TO TEST THE PROGRAM
def run_sample():
    x = np.array(generate_x())
    w = np.array(generate_w())
    y = np.array(generate_y(x, w))
    y_negative = np.array(generate_y_negative(x, w))
    step_size = 0.1


    least_squares_w = gradient_descent_least_squares(x, y, step_size)
    print('Success of least squares', test_algorithm(x, y, least_squares_w))


    cross_entropy_w = gradient_descent_cross_entropy(x, y, step_size)


    print('Success of cross entropy', test_algorithm(x, y, cross_entropy_w))


    softmax_w = gradient_descent_soft_max(x, y_negative, step_size)
    print('Success of softmax',test_algorithm_softmax(x, y_negative, softmax_w))


    perceptron_w, count, fire = perceptron(x, y_negative)
    print('Success of perceptron', test_algorithm_softmax(x, y_negative, perceptron_w))

    lp_time, lp_success, lp_iterations = run_linprog(x, y_negative)

run_sample()
