import copy
import math

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model

data = pd.read_csv('.\\Inputs\\aliasing.csv')


def part_a():
    plt.title("Part a")
    plt.ylabel("target")
    plt.xlabel("Variable")
    plt.scatter(data['Variable'], data['Target'])
    plt.show()


def part_b():
    target_list = data['Target'].to_numpy()

    if target_list[0] >= target_list[1]:
        # incremental part
        for i in range(target_list.shape[0]):
            if target_list[i] < target_list[i + 1]:
                print("its not sorted")
                return

    else:
        for i in range(1, target_list.shape[0]):
            if target_list[i] > target_list[i + 1]:
                print("its not sorted")
                return

    print("its sorted")

def part_c():
    train_dataset = data.sample(frac=0.8, random_state=0)
    test_dataset = data.drop(train_dataset.index)

    plt.title("Part b")
    plt.ylabel("target")
    plt.xlabel("Variable")
    plt.scatter(train_dataset['Variable'], train_dataset['Target'], alpha=0.5, label="train", color="yellow")
    plt.scatter(test_dataset['Variable'], test_dataset['Target'], alpha=0.5, label="test", color="red")
    plt.legend()
    plt.show()


# part d
def part_d(X=data['Variable'], Y=data['Target'], learning_rate=0.99, iterations=1000, degree=5):
    X_train = data['Variable']
    Y_train = data['Target']

    predictions, hist = RegressionTrainerGD(X, Y, learning_rate, iterations, degree)

    # plot one
    plt.scatter(X_train, Y_train, label="original")
    plt.scatter(X_train, predictions, label="test")
    plt.title(
        f'iteration : {iterations}, start learning rate :{learning_rate}, complexity : {degree} \n'
        f' error : {hist[-1]}')
    plt.xlabel("variable")
    plt.ylabel("target")
    plt.legend()
    plt.show()

    # plot 2
    plt.title(
        f'iteration : {iterations}, start learning rate :{learning_rate}, complexity : {degree} \n'
        f' error : {hist[-1]}')
    series = np.arange(0, iterations, 1)
    plt.plot(series, hist)
    plt.show()


def NormalizeData(data):
    return (data - np.mean(data)) / (np.max(data) - np.min(data))


def RegressionTrainerGD(X=data['Variable'], Y=data['Target'], learning_rate=0.99, iterations=1000, degree=1,
                        x_test=None):
    if not isinstance(X, np.ndarray):  # handle pandas dataframe
        X_train_base = X.to_numpy()
        y_train_base = Y.to_numpy()
    else:
        X_train_base = X
        y_train_base = Y
    X_train_base *= 0.1
    X_train = X_train_base.copy()
    Y_train = y_train_base

    for i in range(degree):
        if i == 0:
            continue

        if i == 1:
            # X_train = np.array([X_train, np.power(X_train_base,2)])
            temp = np.power(X_train_base, 2)
            temp = NormalizeData(temp)
            X_train = numpy.vstack([X_train, temp])
            continue
        temp = np.power(X_train_base, i + 1)
        temp = NormalizeData(temp)
        X_train = numpy.vstack([X_train, temp])

        # print("----")
        # print(X_train.shape)
        # print(X_train)
        # print("----")

    X_train = X_train.transpose()
    # print(X_train.shape)
    # print(X_train)

    initial_w = np.random.rand(degree)
    initial_b = 0.1

    w_final, b_final, J_hist = gradient_descent(X_train, Y_train, initial_w, initial_b,
                                                compute_cost, compute_gradient,
                                                learning_rate, iterations)

    predict_list = []

    # it can handle better :)
    if x_test is None:
        for x in X_train:
            predicted = np.dot(x, w_final) + b_final
            predict_list.append(predicted)
    else:
        baseColumn = x_test

        final = x_test * 0.1

        for i in range(2, degree + 1):
            temp = np.power(baseColumn, i)
            temp = NormalizeData(temp)
            temp = np.reshape(temp, (temp.shape[0], -1))
            final = np.reshape(final, (final.shape[0], -1))
            final = np.append(final, temp, axis=1)

        b_final_list = np.ones(x_test.shape[0]) * b_final
        predicted = np.matmul(final, w_final) + b_final_list
        return predicted, J_hist

    return predict_list, J_hist


def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b  # (n,)(n,) = scalar
        cost = cost + (f_wb_i - y[i]) ** 2  # scalar
    cost = cost / (2 * m)  # scalar
    return cost


def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    try:
        m, n = X.shape  # (number of examples, number of features)

        dj_dw = np.zeros((n,))
        dj_db = 0.

        for i in range(m):
            err = (np.dot(X[i], w) + b) - y[i]
            for j in range(n):
                dj_dw[j] = dj_dw[j] + err * X[i, j]
            dj_db = dj_db + err
        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_db, dj_dw
    except:
        m = X.shape[0]
        dj_dw = 0
        dj_db = 0

        for i in range(m):
            f_wb = w * X[i] + b
            dj_dw_i = (f_wb - y[i]) * X[i]
            dj_db_i = f_wb - y[i]
            dj_db += dj_db_i
            dj_dw += dj_dw_i
        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    first_alpha = alpha
    cost_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw  ##None
        b = b - alpha * dj_db  ##None

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            cost_history.append(cost_function(X, y, w, b))

        if (i % 10 == 0):
            print(cost_history[-1])

        if (i > num_iters / 2):
            alpha = 0.75 * first_alpha
        elif (i > num_iters * 0.75):
            alpha = 0.4 * first_alpha
        elif (i > 0.9 * num_iters):
            alpha = 0.1 * first_alpha

    return w, b, cost_history  # return final w,b and J history for graphing


# part e
def part_e(X=data['Variable'], Y=data['Target'], learning_rate=0.5):
    X_train = data['Variable'].copy()
    Y_train = data['Target'].copy()

    iteration_list = [100, 1000, 5000]
    degrees = [1, 2, 3, 5, 10]

    fig, axis = plt.subplots(3, 5, figsize=(30, 25), sharey=False)

    for i in range(len(iteration_list)):
        for j in range(len(degrees)):
            if j == 0:
                linear_learning_rate = 0.0001
                predictions, hist = RegressionTrainerGD(X, Y, linear_learning_rate, iteration_list[i], degrees[j])

                axis[i, j].scatter(X_train * 10, Y_train, label="original")
                axis[i, j].scatter(X_train * 10, predictions, label="test")
                axis[i, j].title.set_text(
                    f'iteration : {iteration_list[i]}, start learning rate :{linear_learning_rate}, complexity : {degrees[j]} \n'
                    f' error : {hist[-1]}')
                axis[i, j].set_xlabel("variable")
                axis[i, j].set_ylabel("target")
                axis[i, j].legend()
            else:
                predictions, hist = RegressionTrainerGD(X, Y, learning_rate, iteration_list[i], degrees[j])

                axis[i, j].scatter(X_train * 10, Y_train, label="original")
                axis[i, j].scatter(X_train * 10, predictions, label="test")
                axis[i, j].title.set_text(
                    f'iteration : {iteration_list[i]}, start learning rate :{learning_rate}, complexity : {degrees[j]} \n'
                    f' error : {hist[-1]}')
                axis[i, j].set_xlabel("variable")
                axis[i, j].set_ylabel("target")
                axis[i, j].legend()
    plt.show()
    # predictions, hist = RegressionTrainerGD(X, Y, learning_rate, iterations, degree)
    #
    # # plot one
    # plt.scatter(X_train, Y_train, label="original")
    # plt.scatter(X_train, predictions, label="test")
    # plt.title(
    #     f'iteration : {iterations}, start learning rate :{learning_rate}, complexity : {degree} \n'
    #     f' error : {hist[-1]}')
    # plt.xlabel("variable")
    # plt.ylabel("target")
    # plt.legend()
    # plt.show()

    # # plot 2
    # plt.title(
    #     f'iteration : {iterations}, start learning rate :{learning_rate}, complexity : {degree} \n'
    #     f' error : {hist[-1]}')
    # series = np.arange(0, iterations, 1)
    # plt.plot(series, hist)
    # plt.show()


def part_e2(X=data['Variable'], Y=data['Target'], learning_rate=0.5):
    iteration_list = [100, 1000, 5000]
    degrees = [1, 2, 3, 5, 10]

    # iteration_list = [10 ,30]
    # degrees = [1, 2, 3, 5, 10]

    fig, axis = plt.subplots(3, 5, figsize=(30, 25), sharey=False)

    for i in range(len(iteration_list)):
        for j in range(len(degrees)):
            if j == 0:
                linear_learning_rate = 0.0001
                predictions, hist = RegressionTrainerGD(X, Y, linear_learning_rate, iteration_list[i], degrees[j])

                axis[i, j].title.set_text(
                    f'iteration : {iteration_list[i]}, start learning rate :{linear_learning_rate}, complexity : {degrees[j]} \n'
                    f' error : {hist[-1]}')
                series = np.arange(0, iteration_list[i], 1)
                axis[i, j].plot(series, hist)

                axis[i, j].set_xlabel("iteration")
                axis[i, j].set_ylabel("error")
            else:
                predictions, hist = RegressionTrainerGD(X, Y, learning_rate, iteration_list[i], degrees[j])

                axis[i, j].title.set_text(
                    f'iteration : {iteration_list[i]}, start learning rate :{learning_rate}, complexity : {degrees[j]} \n'
                    f' error : {hist[-1]}')
                series = np.arange(0, iteration_list[i], 1)
                axis[i, j].plot(series, hist)

                axis[i, j].set_xlabel("iteration")
                axis[i, j].set_ylabel("error")
    plt.show()


def part_f(X=data['Variable'], Y=data['Target'], learning_rate=0.95, iterations=12000, degree=8):
    X_train = data['Variable']
    Y_train = data['Target']

    predictions, hist = RegressionTrainerGD(X, Y, learning_rate, iterations, degree)

    # plot one
    plt.scatter(X_train, Y_train, label="original")
    plt.scatter(X_train, predictions, label="test")
    plt.title(
        f'iteration : {iterations}, start learning rate :{learning_rate}, complexity : {degree} \n'
        f' error : {hist[-1]}')
    plt.xlabel("variable")
    plt.ylabel("target")
    plt.legend()
    plt.show()

    # plot 2
    plt.title(
        f'iteration : {iterations}, start learning rate :{learning_rate}, complexity : {degree} \n'
        f' error : {hist[-1]}')
    series = np.arange(0, iterations, 1)
    plt.plot(series, hist)
    plt.show()


def part_g(degree=10):
    x = data.iloc[:, 0].values  # Features
    y = data.iloc[:, 1].values  # results
    # print(x.shape)

    bias = np.ones((x.shape[0], 1))

    x = np.reshape(x, (x.shape[0], 1))  # this will convert (2000,) => (2000,1)

    temp = x
    for i in range(1, degree):
        x = np.append(x, np.power(temp, i + 1), axis=1)

    final_x = np.append(bias, x, axis=1)

    # t_1 : (Xt * X)^-1
    x_transpose = np.transpose(final_x)  # calculating transpose
    x_transpose_dot_x = x_transpose.dot(final_x)  # calculating dot product
    t_1 = np.linalg.inv(x_transpose_dot_x)  # calculating inverse

    # t_2 : (Xt * y)
    t_2 = x_transpose.dot(y)  # its equal to matmul

    coefficents = t_1.dot(t_2)
    coefficents = np.reshape(coefficents, (coefficents.shape[0], 1))

    predictions = np.matmul(final_x, coefficents)

    # plot
    plt.scatter(data['Variable'], data['Target'], label="original")
    plt.scatter(data['Variable'], predictions, label="test")
    plt.title(f'normal equation regression  \n error {calculate_err(predictions, y)} \n degree = {degree} ')
    plt.xlabel("variable")
    plt.ylabel("target")
    plt.legend()
    plt.show()

    print("")


def calculate_err(target, predicted):
    if not isinstance(target, np.ndarray):
        target = np.array(target)

    if not isinstance(predicted, np.ndarray):
        predicted = np.array(predicted)

    m = target.shape[0]
    target = np.reshape(target, (target.shape[0], 1))
    predicted = np.reshape(predicted, (predicted.shape[0], 1))

    distance = target - predicted
    distance = np.power(distance, 2)
    sum_of_square = np.sum(distance)
    mse = sum_of_square / m
    return mse


# part h
def part_h(K=5, degree=10, learning_rate=0.85, iterations=250):
    H_data = data.copy()
    m = data.shape[0]
    slice_size = int(m / K)

    k_parts = []

    for j in range(K):
        k_parts.append(H_data.iloc[j * slice_size: slice_size + j * slice_size, :].values)  # Features

    errors = 0

    for i in range(K):
        train = None
        test = None
        for j in range(K):
            if i == j:
                test = k_parts[j]
                continue
            if train is None:
                train = k_parts[j]
            else:
                train = np.append(train, k_parts[j], axis=0)

        train_x = train[:, 0]  # Features
        train_y = train[:, 1]  # value
        test_x = test[:, 0]  # Features
        test_y = test[:, 1]  # value
        pred, hist = RegressionTrainerGD(train_x, train_y, x_test=test_x, learning_rate=learning_rate,
                                         iterations=iterations,
                                         degree=degree)

        err = calculate_err(test_y, pred)
        print(f'iteration err {err}')
        plt.title(f'k fold test  \n error {err} \n degree = {degree}  learning_rate {learning_rate} ')
        errors += err

        plt.scatter(test_x, test_y)
        plt.scatter(test_x, pred)
        plt.show()
        # print(err)

    errors /= K
    print(f'avg err is  : {errors}')


if __name__ == '__main__':
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    part_e2()
    part_f()
    part_g()
    part_h()

