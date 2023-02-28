# 导入所需的库
import numpy as np
from sklearn.datasets import load_iris
from collections import Counter

# 加载数据集
iris = load_iris()

# 将数据集分成特征和目标变量
X = iris.data
y = iris.target

# 将数据集分成训练集和测试集
train_X = np.concatenate((X[:40], X[50:90], X[100:140]))
train_y = np.concatenate((y[:40], y[50:90], y[100:140]))
test_X = np.concatenate((X[40:50], X[90:100], X[140:]))
test_y = np.concatenate((y[40:50], y[90:100], y[140:]))

# 定义高斯分布函数
def gaussian(x, mean, std):
    exponent = np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

# 定义朴素贝叶斯分类器
class NaiveBayes:
    def fit(self, X, y): #训练模型
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # 计算每个类别的先验概率
        self.priors = np.zeros(n_classes)
        for c in self.classes:
            self.priors[c] = np.mean(y == c)

        # 计算每个类别的均值和方差
        self.means = np.zeros((n_classes, n_features))
        self.stds = np.zeros((n_classes, n_features))
        for c in self.classes:
            X_c = X[y == c]
            self.means[c, :] = X_c.mean(axis=0)
            self.stds[c, :] = X_c.std(axis=0)

    def predict(self, X):#预测类别
        y_pred = []
        for x in X:
            posteriors = []
            for i, c in enumerate(self.classes):
                # 计算后验概率
                prior = np.log(self.priors[i])
                likelihood = np.sum(np.log(gaussian(x, self.means[i, :], self.stds[i, :])))
                posterior = prior + likelihood
                posteriors.append(posterior)
            # 选择后验概率最大的类别作为预测结果
            y_pred.append(self.classes[np.argmax(posteriors)])
        return y_pred

if __name__ == '__main__':
    # 创建朴素贝叶斯分类器
    nb = NaiveBayes()

    # 使用训练集拟合分类器
    nb.fit(train_X, train_y)

    # 对测试集进行预测
    y_pred = nb.predict(test_X)

    # 计算分类器的准确率
    accuracy = np.mean(y_pred == test_y)
    print("准确率为:", accuracy)
