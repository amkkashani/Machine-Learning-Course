import itertools

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    print("main is called")
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    part_f()
    part_g()

    print("***")
    return


def part_a():
    print("------ a -------")
    raw_dataset = pd.read_csv('.\Inputs\Fish.csv',
                              comment='\t',
                              sep=',', skipinitialspace=True)

    train_x = raw_dataset[['Length1', 'Length2', 'Length3', 'Height', 'Width']].to_numpy()
    train_y = raw_dataset['Weight'].to_numpy()

    train_x = (train_x - train_x.min()) / (train_x.max() - train_x.min())
    train_y = train_y

    regression = Regression(5, 0.01)
    regression.train_by_data(train_x, train_y, 200)
    precision = regression.precision_R2(train_x, train_y)
    print(f'R2 erro :{precision}')
    print(f'R2 prediction {regression.predicted_r2(train_y, train_x)}')


def powerset(elems):
    if len(elems) <= 1:
        yield elems
        yield []
    else:
        for item in powerset(elems[1:]):
            yield [elems[0]] + item
            yield item


def part_b():
    print("------ b -------")
    column_names = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
    raw_dataset = pd.read_csv('.\Inputs\Fish.csv', comment='\t', sep=',', skipinitialspace=True)
    all_comb = list(powerset(column_names))
    predictions_scores = []

    for i in range(len(all_comb)):
        if len(all_comb[i]) == 0:
            predictions_scores.append(0)
            continue

        print("---")
        print(all_comb[i])

        train_x = raw_dataset[all_comb[i]].to_numpy()
        train_y = raw_dataset['Weight'].to_numpy()

        train_x = (train_x - train_x.min()) / (train_x.max() - train_x.min())
        train_y = train_y

        regression = Regression(len(all_comb[i]), 0.1)
        regression.train_by_data(train_x, train_y, 100)
        precision = regression.precision_R2(train_x, train_y)
        predictions_scores.append(precision)

        print(f'R2 : {precision}')
        print(f'predicted R2 :  {regression.predicted_r2(train_y, train_x)}')
        print("---")

    index = 0
    max = 0
    for i in range(len(predictions_scores)):
        if predictions_scores[i] > max:
            max = predictions_scores[i]
            index = i

    print("****")
    print('best answer')
    print(all_comb[index])
    print(predictions_scores[index])


def part_c():
    print("------ c -------")
    raw_dataset = pd.read_csv('.\Inputs\Fish.csv', comment='\t', sep=',', skipinitialspace=True)
    model = forward_selected(raw_dataset, 'Weight')
    print(model.model.formula)
    print(model.rsquared_adj)


def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model


def part_d():
    print("------ d -------")
    raw_dataset = pd.read_csv('.\Inputs\Fish.csv', comment='\t', sep=',', skipinitialspace=True)
    column_names = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
    sns.pairplot(raw_dataset[column_names], diag_kind='kde')
    plt.show()


def part_e():  # no idea about results
    # https://www.kirenz.com/post/2021-11-14-linear-regression-diagnostics-in-python/linear-regression-diagnostics-in-python/

    print("------ e -------")
    # estimate the model and save it as lm (linear model)
    raw_dataset = pd.read_csv('.\Inputs\Fish.csv', comment='\t', sep=',', skipinitialspace=True)
    lm = ols("Weight ~ Length3 + Width + 1", data=raw_dataset).fit()
    fig = sm.graphics.influence_plot(lm, criterion="cooks")
    plt.title("coocks criterion")
    fig.tight_layout(pad=1)
    plt.show()

    fig = sm.graphics.influence_plot(lm, criterion="DFFITS")
    plt.title("DFFITS criterion")
    fig.tight_layout(pad=1)
    plt.show()


def part_f():
    print("------ f -------")
    return


def part_g():
    print("------ g -------")
    raw_dataset = pd.read_csv('.\Inputs\Fish.csv', comment='\t', sep=',', skipinitialspace=True)
    train_x = raw_dataset[['Length3', 'Width']].to_numpy()
    train_y = raw_dataset['Weight'].to_numpy()
    train_x = convert_x_train_to_multi_dimension(train_x)

    train_x =pd.DataFrame(train_x, columns = ['A','B','A2','B2','AB'])
    train_x = sm.add_constant(train_x)
    train_y =pd.DataFrame(train_y, columns = ['Weight'])

    result = pd.concat([train_x, train_y], axis=1)

    model = ols("Weight ~ A + B + B2 + A2 + AB + 1", data=result).fit()
    # model = ols(train_y , train_x)
    print(model.summary())


def convert_x_train_to_multi_dimension(X):
    if not isinstance(X, np.ndarray):
        X = X.to_numpy()  # convert pandas to numpy
    x0 = X[:, 0]
    x1 = X[:, 1]

    return np.array([x0, x1, np.power(x0, 2), np.power(x1, 2), np.multiply(x0, x1)]).T


class Regression:
    def __init__(self, degree, learning_rate):  # initial w , and b
        self.degree = degree
        self.W = np.random.rand(degree + 1) * 2 - np.ones(degree + 1)  # between (-1 and 1]   && +1 for bias
        self.learning_rate = learning_rate
        self.history_err_train = []
        self.history_err_test = []
        self.initial_learning_rate = learning_rate

    def train_by_data(self, X, y, epoch):
        ##
        # X is matrix of inputs
        # y is matrix of results
        ##
        for i in range(epoch):
            self.learning_rate -= self.learning_rate * (i / (epoch * 1.1))
            prev_W = self.W
            for j in range(X.shape[0]):
                x = X[j, :]
                y_exact = y[j]
                self.train_a_data(x, y_exact)

            self.history_err_train.append(self.precision_R2(X, y))

    def train_a_data(self, X, y):  # just for classification
        ###
        # X is a vector
        # y is a result value
        ###
        prediction = self.calculate_value_without_activision_function(X)
        X = np.append(np.ones(1), X)

        self.W += self.learning_rate * (y - prediction) * X

    def calculate_value_without_activision_function(self, X):
        if not isinstance(X, np.ndarray):  # not necessary and can remove
            print("its not a numpy matrix")
            return

        X = np.append(np.ones(1), X)
        z = np.dot(X, self.W)
        return z

    def calculate_value(self, X):
        if not isinstance(X, np.ndarray):  # not necessary and can remove
            print("its not a numpy matrix calculate value")
            return

        ones = np.ones(X.shape[0])
        ones = ones.reshape((ones.shape[0], 1))
        X = np.reshape(X, (X.shape[0], -1))
        X = np.append(ones, X, axis=1)

        z = np.dot(X, self.W)
        return z

    def precision_R2(self, X, y_true):
        ##
        # X matrix
        # y vector
        ##
        y_pred = self.calculate_value(X)
        sse = np.square(y_pred - y_true).sum()
        sst = np.square(y_true - y_true.mean()).sum()
        return 1 - sse / sst

    def press_statistic(self, y_true, y_pred, xs):
        res = y_pred - y_true
        hat = xs.dot(np.linalg.pinv(xs))
        den = (1 - np.diagonal(hat))
        sqr = np.square(res / den)
        return sqr.sum()

    def predicted_r2(self, y_true, X):
        y_pred = self.calculate_value(X)
        press = self.press_statistic(y_true=y_true, y_pred=y_pred, xs=X)
        sst = np.square(y_true - y_true.mean()).sum()
        return 1 - press / sst

    def plot_boundary_and_data(self, blues, reds, precision):
        ##
        # blues are pandas array
        # reds are pandas array
        ##
        x_axis = np.arange(-3, 9, 0.01)
        w = self.W
        y_axis = -w[0] / w[2] - x_axis * w[1] / w[2]

        plt.title(f'Adaline \n precision : {precision}  initial_learning_rate : {self.initial_learning_rate}')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(x_axis, y_axis)
        plt.scatter(blues['x'], blues['y'])
        plt.scatter(reds['x'], reds['y'])
        plt.show()

    def plot_curve_boundary_and_data(self, blues, reds, precision):
        ##
        # blues are pandas array
        # reds are pandas array
        ##
        x_axis = np.arange(-3, 9, 0.01)
        w = self.W

        x = np.linspace(-3, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        F = w[3] * X ** 2 + w[5] * X * Y + w[4] * Y ** 2 + w[1] * X + w[2] * Y + w[0]

        plt.title(f'Adaline \n  precision : {precision} initial_learning_rate : {self.initial_learning_rate}')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.contour(X, Y, F, levels=[0])  # take level set corresponding to 0
        plt.scatter(blues['x'], blues['y'])
        plt.scatter(reds['x'], reds['y'])
        plt.show()

    def plot_accuracy_history(self, precision):
        plt.title(
            f'Adaline \n train_precision : {self.history_err_train[-1]} test_precision : {self.history_err_test[-1]} \ninitial_learning_rate : {self.initial_learning_rate}')
        plt.xlabel("epoch")
        plt.ylabel("precision")
        x = np.arange(0, len(self.history_err_test), 1)
        plt.plot(x, self.history_err_train, label="train", alpha=0.5)
        plt.plot(x, self.history_err_test, label="test", alpha=0.5)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
