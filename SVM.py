import numpy as np
import pandas as pd
import cvxopt
import random
from sklearn.metrics import accuracy_score


def train_test_split(x, y, test_rate):
    num_x = x.shape[0]
    # 打乱
    index = list(range(num_x))
    random.shuffle(index)
    x = x[index]
    y = y[index]

    split = round(num_x * test_rate)
    x_test = x[:split, :]
    x_train = x[split:, :]
    y_test = y[:split]
    y_train = y[split:]
    return x_train, x_test, y_train, y_test


# Iris数据集
# 150个样本，4维特征，三种类别
data1 = pd.read_csv('iris.csv', header=None)
data1 = np.array(data1)
x1 = np.float64(data1[0:100, 0:4])
y1 = data1[0:100, 4]
for i in range(len(y1)):
    if y1[i] == 'Iris-setosa':
        y1[i] = -1
    else:
        y1[i] = 1
y1 = np.int64(y1)
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_rate=0.3)


def linear_kernal(**kwargs):
    def K(x1, x2):
        return np.inner(x1, x2)

    return K


class svm:
    # 输入参数
    # param kernal:核函数
    # param penaltyC:软间隔惩罚项C
    # param power, gamma,coef为核函数的一些参数
    def __init__(self, kernal=linear_kernal, penaltyC=1, power=1, gamma=1, coef=1):
        self.kernal = kernal
        self.penaltyC = penaltyC
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.kernal = self.kernal(power=self.power,gamma=self.gamma,coef=self.coef)

    def train(self, x, y):
        x_num = x.shape[0]
        kernal_matrix = self.kernal(x, x) + (1 / self.penaltyC) * np.eye(x_num)

        # 计算标准凸二次规划的几个参数
        p = cvxopt.matrix(kernal_matrix * np.outer(y, y))
        q = cvxopt.matrix(-np.ones([x_num, 1], np.float64))
        g = cvxopt.matrix(-np.eye(x_num))
        h = cvxopt.matrix(np.zeros([x_num, 1], np.float64))

        y = np.float64(y)
        a = cvxopt.matrix(y, (1, x_num))
        b = cvxopt.matrix(0.)

        # 使用凸规划工具包cvxopt求解SVM目标函数（算lagrange乘子）
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(p, q, g, h, a, b)
        alpha = np.float32(np.array(solution['x']))
        alpha[alpha <= 1e-4] = 0

        # 求权重w和截距b
        w = np.sum(np.reshape(y, [-1, 1]) * alpha * x, axis=0)
        b = np.mean(np.reshape(y, [-1, 1]) - np.reshape(np.dot(w, np.transpose(x)), [-1, 1]))
        self.w = w
        self.b = b

        return w, b, alpha

    def predict(self, x_test):
        y_predict = []
        for sample in x_test:
            predict1 = self.kernal(self.w, sample) + self.b
            predict1 = np.int64(np.sign(predict1))
            y_predict.append(predict1)
        #         y_predict.tolist()
        return y_predict


if __name__ == '__main__':
    svm1 = svm(linear_kernal, penaltyC=1)
    w1, b1, alpha1 = svm1.train(x1, y1)
    y1_predict = svm1.predict(x1_test)
    print('准确率为:', accuracy_score(y1_test, y1_predict))
