from autograd import grad
import autograd.numpy as np
from scipy.optimize import linprog
from time import perf_counter
from cvxopt.solvers import qp
from cvxopt import matrix
from matplotlib import pyplot as plt

#CHRISTIAN EID
#Sample Code that runs all algorithms at the bottom.

#PART 1: Training Data

def generate_w():
    return [np.random.uniform(-1, 1), np.random.uniform(-2, 2), np.random.uniform(-2, 2)]

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
            y.append(-1)
    return y


#For testing

def test_algorithm(x, y, w):
    success_rate = 0
    test_y = np.array(generate_y(x, w))
    for i in range(100):
        if test_y[i] == y[i]:
            success_rate += 1
    return success_rate



#PART 2 : Linearly Seperable Data Algorithms

#PART 2(a) Perceptron
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

##PART 2(b) Linprog
def run_linprog(x, y):
    objective = [0, 0, 0]
    inequalities, inequality_rhs = make_inequalities(x, y)
    start = perf_counter()
    result = linprog(objective, inequalities, inequality_rhs, bounds=(None, None))
    stop = perf_counter()

    return (stop-start, test_algorithm(x, y, result.x), result.nit)

def make_inequalities(x, y):
    res = []
    rhs = []
    for i in range(100):
        if y[i] == 1.0:
            res.append([-x[i][0], -x[i][1], -x[i][2]])
        else:
            res.append([x[i][0], x[i][1], x[i][2]])
        rhs.append(-1.0)
    return res, rhs

##PART 2(c) HARD SVM
def hard_SVM(x, y):

    inequalities, inequality_rhs = make_inequalities(x, y)

    G = matrix(inequalities, tc='d')
    h = matrix(inequality_rhs, tc='d')
    P = matrix(np.identity(3))
    q = matrix([0.0, 0.0, 0.0])
    start = perf_counter()
    sol = qp(P, q, G.trans(), h)
    stop=perf_counter()
    print(sol)
    qp_w = [sol['x'][0], sol['x'][1], sol['x'][2]]

    print('success of quadratic programming', test_algorithm(x, y, qp_w))
    return (stop-start), sol['iterations']




#PART 2(d) : SoftSVM

def norm(w):
    return np.sqrt(np.dot(w, w))

#Hinge loss with regularizer
def hinge_loss(w, x, y):
    res = 0.0

    for i in range(100):
        res += ( max(0, 1 - y[i]* np.dot(x[i], w)) + 0.000001* (norm(w))**2)
    res *= 1/100
    return res


#There are two soft svm
#Each has a different stopping condition
def get_gradient_soft_svm(x, y):
    LS_hinge = lambda w: hinge_loss(w, x, y)
    return grad(LS_hinge)


def gradient_descent_soft_SVM(x, y, step_size):
    gradient_softSVM = get_gradient_soft_svm(x, y)

    w = np.array([np.random.uniform(-1, 1), np.random.uniform(-2, 2), np.random.uniform(-2, 2)])

    i = 0

    while i < 100:
        w = w - step_size * gradient_softSVM(w)
        i += 1
    return w, i

def gradient_descent_soft_SVM_100(x, y, step_size):
    gradient_softSVM = get_gradient_soft_svm(x, y)

    w = np.array([np.random.uniform(-1, 1), np.random.uniform(-2, 2), np.random.uniform(-2, 2)])

    i = 0

    while test_algorithm(x, y, w) != 100:
        w = w - step_size * gradient_softSVM(w)
        i += 1
        if i == 500:
            print('Soft SVM reached 500 with ', test_algorithm(x, y, w))
            return w, i
    return w, i



#PART 3: NON LINEAR TRAINING DATA
#Using X, we can generate a feature space and classify the data with it




def generate_feature_space(x):
    res = []
    for i in range(100):
        list = [x[i][0], x[i][1], x[i][2], x[i][1] * x[i][2], x[i][1]**2, x[i][2]**2]
        res.append(list)
    return res

def generate_nonlinear_y(x, c):
    y = []
    for i in range(100):
        if np.dot(c, x[i])> 0:
            y.append(1)
        else:
            y.append(-1)
    return y

def generate_c():
    return [np.random.uniform(-1, 1),np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2)]

#PART 3(a): Perceptron using non_linear data

def perceptron_nonlinear(x, y):
    w = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    i = 0
    iteration_count = 0
    fire_count = 0
    feature_space = np.array(generate_feature_space(x))
    while i < 100:
        iteration_count+=1
        if y[i] * np.dot(feature_space[i], w) <= 0:
            fire_count += 1
            w += np.dot(y[i],feature_space[i])
            i = 0
        i+=1
    return w, iteration_count, fire_count


def test_nonlinear(x, y, c):
    success_rate = 0
    test_y = generate_nonlinear_y(x, c)
    for i in range(100):
        if test_y[i] == y[i]:
            success_rate += 1
    return success_rate





#PART 3(b): Linear Programming using the feature space
def linprog_featurespace(x, y):
    right = []
    left = []
    for i in range(100):
        if y[i] == 1.0:
            left.append([-x[i][0], -x[i][1], -x[i][2], -x[i][3], -x[i][4], -x[i][5]])
        else:
            left.append([x[i][0], x[i][1], x[i][2], x[i][3], x[i][4], x[i][5]])
        right.append(-1.0)


    objective = [0, 0, 0, 0, 0, 0]
    start = perf_counter()
    result = linprog(objective, left, right, bounds=(None, None))
    stop = perf_counter()
    print(result)
    print('nonlinear linprog success = ', test_nonlinear(x, y, result.x))
    return (stop-start, test_algorithm(x, y, result.x), result.nit)


#PART 3(c): Hard SVM of Nonlinear Data
def hard_svm_nonlinear(x, y):
    right = []
    left = []
    for i in range(100):
        if y[i] == 1.0:
            left.append([-x[i][0], -x[i][1], -x[i][2], -x[i][3], -x[i][4], -x[i][5]])
        else:
            left.append([x[i][0], x[i][1], x[i][2], x[i][3], x[i][4], x[i][5]])
        right.append(-1.0)

    # print(inequality_rhs)
    G = matrix(left, tc='d')
    h = matrix(right, tc='d')
    # print(h)

    P = matrix(np.identity(6))
    q = matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    start = perf_counter()
    sol = qp(P, q, G.trans(), h)
    stop = perf_counter()
    # print(sol)
    qp_w = [sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]]
    # print(qp_w)
    print('success of quadratic nonlinear', test_nonlinear(x, y, qp_w))
    return (stop - start), sol['iterations']
# hard_svm_nonlinear(feature_space, y)

#PART 3(d): Soft SVM on nonlinear data
def hinge_loss_nonlinear(s, x, y):
    c = 0.1
    res = 0.0
    w =[]
    for i in range(6):
        w.append(s[i])
    for i in range(100):
        res += (max(0, 1 - y[i] * np.dot(x[i], w)) + 0.001*(norm(w))**2)
    res *= 1/100
    return res

def generate_hinge_nonlinear(feature_space, y):
    LS_hinge_nonlinear = lambda w: hinge_loss_nonlinear(w, feature_space, y)
    return grad(LS_hinge_nonlinear)

def gradient_descent_soft_SVM_feature_space(x, y, step_size):

    gradient_softSVM_nonlinear = generate_hinge_nonlinear(x, y)
    w = np.array([np.random.uniform(-1, 1), np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
    i = 0

    while i < 100:
        w = w - step_size * gradient_softSVM_nonlinear(w)
        i += 1
    return w, i

def gradient_descent_soft_SVM_feature_space_100(x, y, step_size):

    gradient_softSVM_nonlinear = generate_hinge_nonlinear(x, y)
    w = np.array([np.random.uniform(-1, 1), np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
    i = 0

    while test_nonlinear(x, y, w) != 100:
        w = w - step_size * gradient_softSVM_nonlinear(w)
        i += 1
        if i == 500:
            print('Soft SVM reached 500 with ', test_nonlinear(x, y, w))
            return w, i
    return w, i



#


##PART 4: IMPLEMENTING KERNALS


##PART 4(a): Implementing the Polynomial Kernalized Soft SVM

def Kernel(u, v):
    return (1 + np.dot(u, v))**2-1

def kernal_sum(w, x, i):
    res = 0.0
    for j in range(100):
        res += w[j] * Kernel(x[j], x[i])
    return res


def kernelized_hinge_loss(w, x, y):
    res = 0.0
    for i in range(100):
        res += max(0, 1 - y[i]*(kernal_sum(w, x, i)) + 0.0001*(norm(w))**2)
    res *= 1/100
    return res

def generate_kernelized_hinge_loss(x, y):
    LS_kernelized_loss = lambda w : kernelized_hinge_loss(w, x, y)
    return grad(LS_kernelized_loss)

def gradient_descent_kernalized(x, y, step_size):
    gradient_kernalized = generate_kernelized_hinge_loss(x, y)

    #NEEDS TO BE 100 long
    w = []
    for i in range(100):
        w.append(np.random.uniform(-2, 2))
    alpha = np.array(w)
    i = 0
    while i < 15:

        alpha = alpha - step_size * gradient_kernalized(alpha)
        i += 1
        if test_kernel(alpha, x, y)==100:
            return w, i
    return alpha, i




##PART 4(b): Gaussian Kernel
def Gaussian(x, z):
    scale = 1
    return np.power(np.e, (-norm(x-z)**2/(2*scale)))

def gaussian_sum(w, x, i):
    res = 0.0
    for j in range(100):
        res += w[j] * Gaussian(x[j], x[i])
    return res


def gaussian_hinge(w, x, y):
    res = 0.0
    for i in range(100):
        res += max(0, 1 - y[i]*(gaussian_sum(w, x, i) )+ 0.001*(norm(w))**2)
    res *= 1/100
    return res

def generate_gaussian_loss(x, y):
    LS_gaussian_loss = lambda w : gaussian_hinge(w, x, y)
    gradient_gaussian = grad(LS_gaussian_loss)

def gradient_descent_gaussian(x, y, step_size):
    gradient_gaussian = generate_gaussian_loss(x, y)

    #NEEDS TO BE 100 long
    w = []
    for i in range(100):
        w.append(np.random.uniform(-2, 2))
    alpha = np.array(w)
    i = 0
    while i < 10:
        alpha = alpha - step_size * gradient_gaussian(alpha)
        i += 1
        if test_kernel(alpha, x, y)==100:
            return w, i
    return alpha
#Implementing Classification using Alpha vector

def classify_Kernalized(alpha, instances, instance):
    res = 0
    for i in range(100):
        res += alpha[i]*Kernel(instances[i], instance)
    if res > 0:
        return 1
    else:
        return -1

#Testing the success rate of Kernalized functions
def test_kernel(alpha, instances, y):
    success_rate = 0
    test_y = []
    for i in range(100):
        test_y.append(classify_Kernalized(alpha, instances, instances[i]))


    for i in range(100):
        if test_y[i] == y[i]:
            success_rate += 1
    return success_rate

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
#PART 5 TESTS

#Test 1 Linearly Classified Data

def linear_test():

    # INITIALIZING DATA TO COLLECT
    cycles = 1
    print("Number of Cycles: ", cycles)

    perceptron_time = 0
    perceptron_iterations = 0
    perceptron_fire = 0

    linprog_time = 0
    linprog_iterations = 0

    hard_svm_time = 0
    hard_svm_iterations = 0

    soft_svm_time = 0
    soft_svm_iterations = 0
    soft_svm_success = 0

    for i in range(cycles):
        print("At cycle", i)

        # ESTABLISHING A NEW TRAINING SET
        w = np.array([np.random.uniform(-1, 1), np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
        x = np.array(generate_x())
        y = np.array(generate_y(x, w))

        step_size = 1.0

        # Perceptron Test
        start_1 = perf_counter()
        perceptron_w, count, fire = perceptron(x, y)
        stop_1 = perf_counter()

        perceptron_time += (stop_1 - start_1)
        perceptron_iterations += count
        perceptron_fire += fire

        #Linprog Test
        lp_time, lp_success, lp_iterations = run_linprog(x, y)
        linprog_time += lp_time
        linprog_iterations += lp_iterations

        hard_time, hard_iterations = hard_SVM(x, y)
        hard_svm_time += hard_time
        hard_svm_iterations += hard_iterations


        soft_w, soft_i = gradient_descent_soft_SVM(x, y, 10)

        start4 = perf_counter()
        soft_w2, soft_i2 = gradient_descent_soft_SVM_100(x, y, 10)
        stop4 = perf_counter()
        soft_svm_time += stop4 - start4
        soft_svm_iterations += soft_i2
        soft_svm_success += test_algorithm(x, y, soft_w)

    perceptron_time *= 1 / cycles
    perceptron_iterations *= 1 / cycles
    perceptron_fire *= 1 / cycles
    linprog_time *= 1 / cycles
    linprog_iterations *= 1 / cycles
    hard_svm_time *= 1 / cycles
    hard_svm_iterations *= 1 / cycles
    soft_svm_time *= 1 / cycles
    soft_svm_iterations *= 1 / cycles
    soft_svm_success *= 1 / cycles
    print('Perceptron iterations: ', perceptron_iterations)
    print('soft svm succes:', soft_svm_success)
    data = [perceptron_time, linprog_time, hard_svm_time, soft_svm_time, perceptron_fire, linprog_iterations, hard_svm_iterations, soft_svm_iterations]
    for val in data:
        print(val)

# linear_test()

#Test 2: Nonlinearly seperable data
def non_linear_test():

    # INITIALIZING DATA TO COLLECT
    cycles = 25
    print("Number of Cycles: ", cycles)

    perceptron_time = 0
    perceptron_iterations = 0
    perceptron_fire = 0

    linprog_time = 0
    linprog_iterations = 0

    hard_svm_time = 0
    hard_svm_iterations = 0

    soft_svm_time = 0
    soft_svm_iterations = 0
    soft_svm_success = 0

    for i in range(cycles):
        print("At cycle", i)

        # ESTABLISHING A NEW TRAINING SET
        c = np.array(generate_c())
        x = np.array(generate_x())
        feature_space = np.array(generate_feature_space(x))
        y = np.array(generate_nonlinear_y(feature_space, c))



        # Perceptron Test
        start_1 = perf_counter()
        perceptron_w, count, fire = perceptron_nonlinear(feature_space, y)
        stop_1 = perf_counter()


        perceptron_time += (stop_1 - start_1)
        perceptron_iterations += count
        perceptron_fire += fire

        #Linprog Test
        lp_time, lp_success, lp_iterations = linprog_featurespace(feature_space, y)
        linprog_time += lp_time
        linprog_iterations += lp_iterations
        #
        hard_time, hard_iterations = hard_svm_nonlinear(feature_space, y)
        hard_svm_time += hard_time
        hard_svm_iterations += hard_iterations
        #
        #
        soft_w, soft_i = gradient_descent_soft_SVM_feature_space(feature_space, y, 10)
        #
        print(soft_w, soft_i)
        start4 = perf_counter()
        soft_w2, soft_i2 = gradient_descent_soft_SVM_feature_space_100(feature_space, y, 10)
        stop4 = perf_counter()
        soft_svm_time += stop4 - start4
        soft_svm_iterations += soft_i2
        soft_svm_success += test_algorithm(feature_space, y, soft_w)

    perceptron_time *= 1 / cycles
    perceptron_iterations *= 1 / cycles
    perceptron_fire *= 1 / cycles
    linprog_time *= 1 / cycles
    linprog_iterations *= 1 / cycles
    hard_svm_time *= 1 / cycles
    hard_svm_iterations *= 1 / cycles
    soft_svm_time *= 1 / cycles
    soft_svm_iterations *= 1 / cycles
    soft_svm_success *= 1 / cycles
    print('Perceptron iterations: ', perceptron_iterations)
    print('soft svm succes:', soft_svm_success)
    data = [perceptron_time, linprog_time, hard_svm_time, soft_svm_time, perceptron_fire, linprog_iterations, hard_svm_iterations, soft_svm_iterations]
    for val in data:
        print(val)

# non_linear_test()

#Test 3: Kernelized Soft SVM & Hard SVM feature mapping
def kernel_test():

    # INITIALIZING DATA TO COLLECT
    cycles = 25
    print("Number of Cycles: ", cycles)


    hard_svm_time = 0
    hard_svm_iterations = 0

    soft_polynomial_time = 0
    soft_polynomial_iterations = 0
    soft_polynomial_success = 0

    soft_gaussian_time = 0
    soft_gaussian_iterations = 0
    soft_gaussian_success = 0

    step_size = 10

    for i in range(cycles):
        print("At cycle", i)

        # ESTABLISHING A NEW TRAINING SET
        c = np.array(generate_c())
        x = np.array(generate_x())
        feature_space = np.array(generate_feature_space(x))
        y = np.array(generate_nonlinear_y(feature_space, c))


        #HARD SVM with Feature Mapping


        hard_time, hard_iterations = hard_svm_nonlinear(feature_space, y)
        hard_svm_time += hard_time
        hard_svm_iterations += hard_iterations
        #
        start1 = perf_counter()
        polynomial_alpha, polynomial_i = gradient_descent_kernalized(x, y, 10)
        print('Sucess of polynomial: ', test_kernel(polynomial_alpha, x, y))
        stop1=perf_counter()
        soft_polynomial_time += stop1 - start1
        soft_polynomial_success += test_kernel(polynomial_alpha, x, y)

        soft_polynomial_iterations += polynomial_i

        start2 = perf_counter()
        gaussian_alpha, gaussian_i = gradient_descent_kernalized(x, y, 10)
        print('Sucess of gaussian: ', test_kernel_Gaussian(gaussian_alpha, x, y))
        stop2 = perf_counter()
        soft_gaussian_time += stop2 - start2
        soft_gaussian_success += test_kernel_Gaussian(gaussian_alpha, x, y)
        soft_gaussian_iterations += gaussian_i

    hard_svm_time *= 1/cycles
    hard_svm_iterations *= 1/cycles
    hard_svm_success = 100

    soft_polynomial_time *= 1/cycles
    soft_polynomial_iterations *= 1/cycles
    soft_polynomial_success *= 1/cycles

    soft_gaussian_time *= 1/cycles
    soft_gaussian_iterations *= 1/cycles
    soft_gaussian_success *= 1/cycles


    data = [hard_svm_time, soft_polynomial_time, soft_gaussian_time, hard_svm_iterations, soft_polynomial_iterations, soft_gaussian_iterations, hard_svm_success, soft_polynomial_success, soft_gaussian_success]
    for val in data:
        print(val)

# kernel_test()


#PLOTTING
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
    x = np.array(generate_x())
    feature_space = np.array(generate_feature_space(x))
    c = np.array(generate_c())
    y = np.array(generate_nonlinear_y(feature_space, c))

    x1, y1, x2, y2 = divide_x(x,y)
    plt.scatter(x1, y1, c = 'blue')
    plt.scatter(x2, y2, c = 'green')
    # plt.quiver(0, 0, w[1], w[2], units = 'xy', scale = 1 )
    plt.title("Nonlinearly Seperable Data")

    #point (0, -b/w2)
    plt.show()

# make_plot()
def run_sample():
    w = np.array([np.random.uniform(-1, 1), np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
    x = np.array(generate_x())
    feature_space = np.array(generate_feature_space(x))
    y = np.array(generate_y(x, w))

    c = np.array(generate_c())
    nonlinear_y = np.array(generate_nonlinear_y(feature_space, c))
    # Linear
    perceptron_w, count, fire = perceptron(x, y)
    print('Success of perceptron ', test_algorithm(x, y, perceptron_w))
    run_linprog(x, y)
    hard_SVM(x, y)
    soft_w, soft_i = gradient_descent_soft_SVM(x, y, 10)
    print('Success of soft SVM', test_algorithm(x, y, soft_w))

    # Nonlinear
    perceptron_w, count, fire = perceptron_nonlinear(feature_space, nonlinear_y)
    print('Success of perceptron nonlinear', test_algorithm(feature_space, nonlinear_y, perceptron_w))
    lp_time, lp_success, lp_iterations = linprog_featurespace(feature_space, nonlinear_y)
    hard_time, hard_iterations = hard_svm_nonlinear(feature_space, nonlinear_y)
    soft_w, soft_i = gradient_descent_soft_SVM_feature_space(feature_space, nonlinear_y, 10)
    print('Success of soft SVM nonlinear ', test_algorithm(feature_space, nonlinear_y, soft_w))

    #Kernelized
    hard_time, hard_iterations = hard_svm_nonlinear(feature_space, nonlinear_y)
    polynomial_alpha, polynomial_i = gradient_descent_kernalized(x, nonlinear_y, 10)
    gaussian_alpha, gaussian_i = gradient_descent_kernalized(x, nonlinear_y, 10)
    print('Sucess of polynomial: ', test_kernel(polynomial_alpha, x, nonlinear_y))
    print('Sucess of gaussian: ', test_kernel_Gaussian(gaussian_alpha, x, nonlinear_y))

run_sample()
