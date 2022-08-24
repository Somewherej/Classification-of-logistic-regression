import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
np.random.seed(5)




class LogisticRegression_Newton:
    def __init__(self,epoch,lr):
        self.epoch = epoch
        self.a = None
        self.lr = lr
        # 不同于梯度下降实现的逻辑回归,我们直接存loss,便于打印
        self.loss = pd.Series(np.arange(self.epoch, dtype=float))

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self,xTrain,yTrain):
        #进行梯度的计算
        def getGradient(xTrain, yTrain, hy):
            return np.dot(xTrain.T, yTrain[:, np.newaxis] - hy)
        #计算损失
        def loss_function(a, xTrain, yTrain):
            hy = self.sigmoid(xTrain.dot(a))
            return -1 / len(yTrain) * np.sum(yTrain * np.log(hy.flatten()) + (1 - yTrain) * np.log(1 - hy.flatten()))


        xTrain = np.hstack([np.ones((len(xTrain), 1)), xTrain])
        #特征数量
        n = xTrain.shape[1]
        #样本数量
        m = xTrain.shape[0]
        self.a = np.zeros((n,1))


        for i in range(self.epoch):
            hy = self.sigmoid(xTrain.dot(self.a))
            self.loss[i] = loss_function(self.a, xTrain, yTrain)
            A = (hy - 1) * hy * np.eye(len(xTrain))
            H = np.mat(xTrain.T) * A * np.mat(xTrain)
            g = getGradient(xTrain, yTrain, hy)
            self.a -= np.linalg.pinv(H) * g
            #如果达到条件就停止更新
            if np.linalg.norm(self.a) < 1e-8:
                break

        #直接在fit里画图
        plt.plot(self.loss.values,color='red')
        plt.xlabel(('epoch'))
        plt.ylabel('loss')
        plt.show()

    def predict(self,X):
        X = np.hstack([np.ones((len(X), 1)), X])
        yPredict = self.sigmoid(X.dot(self.a))
        # 概率>=0.5,类别为1   概率<0.5,类别为0
        yPredict = np.array(yPredict)
        for i in range(len(yPredict)):
            if yPredict[i] >= 0.5:
                yPredict[i] = 1
            else:
                yPredict[i] = 0
        return yPredict

    def calculate(self, xTrain, yTrain, xTest, yTest):
        def calculate(TP, FP, TN, FN):
            P = TP / (TP + FP)  # 查准率
            R = TP / (TP + FN)  # 召回率（查全率
            accuracy = (TP + TN) / (TP + FP + TN + FN)
            error_rate = (FN + FP) / (TP + FP + TN + FN)  # 错误率
            F1_Score = 2.0 * P * R / (P + R)
            print('Precision = ', P)
            print('Recall = ', R)
            print('Error_rate = ', error_rate)
            print('F1 = ', F1_Score)

        y_pre_train = self.predict(xTrain)
        y_pre_train = np.array(y_pre_train)
        yTrain = np.array(yTrain)
        y_pre_test = self.predict(xTest)
        y_pre_test = np.array(y_pre_test)
        yTest = np.array(yTest)
        Positive_TP = 0.0
        Positive_FP = 0.0
        Positive_TN = 0.0
        Positive_FN = 0.0

        Negative_TP = 0.0
        Negative_FP = 0.0
        Negative_TN = 0.0
        Negative_FN = 0.0

        for i in range(len(y_pre_train)):
            Positive_TP += y_pre_train[i] == yTrain[i] == 1
            Positive_FP += (y_pre_train[i] == 1 and yTrain[i] == 0)
            Positive_FN += (y_pre_train[i] == 0 and yTrain[i] == 1)
            Positive_TN += (y_pre_train[i] == yTrain[i] == 0)
            Negative_TP += y_pre_train[i] == yTrain[i] == 0
            Negative_FP += (y_pre_train[i] == 0 and yTrain[i] == 1)
            Negative_FN += (y_pre_train[i] == 1 and yTrain[i] == 0)
            Negative_TN += (y_pre_train[i] == yTrain[i] == 1)

        print("Train:")
        print("Positive")
        calculate(Positive_TP, Positive_FP, Positive_TN, Positive_FN)
        print("Negative")
        calculate(Negative_TP, Negative_FP, Negative_TN, Negative_FN)

        print("--------------")

        Positive_TP = 0.0
        Positive_FP = 0.0
        Positive_TN = 0.0
        Positive_FN = 0.0

        Negative_TP = 0.0
        Negative_FP = 0.0
        Negative_TN = 0.0
        Negative_FN = 0.0
        for i in range(len(y_pre_test)):
            Positive_TP += y_pre_test[i] == yTest[i] == 1
            Positive_FP += (y_pre_test[i] == 1 and yTest[i] == 0)
            Positive_FN += (y_pre_test[i] == 0 and yTest[i] == 1)
            Positive_TN += (y_pre_test[i] == yTest[i] == 0)
            Negative_TP += y_pre_test[i] == yTest[i] == 0
            Negative_FP += (y_pre_test[i] == 0 and yTest[i] == 1)
            Negative_FN += (y_pre_test[i] == 1 and yTest[i] == 0)
            Negative_TN += (y_pre_test[i] == yTest[i] == 1)

        print("Test:")
        print("Positive")
        calculate(Positive_TP, Positive_FP, Positive_TN, Positive_FN)
        print("Negative")
        calculate(Negative_TP, Negative_FP, Negative_TN, Negative_FN)




if __name__ == '__main__':
    data = pd.read_csv('./iris.csv', encoding='gbk')
    data = shuffle(data)   #必须打乱
    kf = KFold(n_splits=10)  # 10折
    X = data.iloc[:, :4]
    Y = data.iloc[:, 4]
    for train_index, test_index in kf.split(X, Y):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        xTrain = train.iloc[:, :4]
        yTrain = train.iloc[:, 4]
        xTest = test.iloc[:, :4]
        yTest = test.iloc[:, 4]

        xTrain = np.array(xTrain)
        yTrain = np.array(yTrain)
        xTest = np.array(xTest)
        yTest = np.array(yTest)

        model = LogisticRegression_Newton(epoch=1000, lr=0.1)
        model.fit(xTrain, yTrain)
        model.calculate(xTrain, yTrain, xTest, yTest)



