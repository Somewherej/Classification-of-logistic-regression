import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
np.random.seed(5)





class LogisticRegression_GradientDescent:
    def __init__(self, epoch, lr):
        self.a = None
        self.epoch = epoch
        self.lr = lr

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))


    def fit(self, xTrain, yTrain):

        # 通过梯度下降法求参数a的更新式
        def gradient_descent(a, X, yTrain):
            hy = self.sigmoid(X.dot(a))
            return X.T.dot(hy - yTrain) / len(yTrain)

        # 计算损失函数
        def loss_function(a, X, yTrain):
            hy = self.sigmoid(X.dot(a))
            # 利用最大似然估计, 并将函数取负,便可以利用梯度下降法
            # 损失函数的最终形式   -1 / len(yTrain) * np.sum(yTrain * np.log(hy) + (1 - yTrain) * np.log(1 - hy))
            return -1 / len(yTrain) * np.sum(yTrain * np.log(hy) + (1 - yTrain) * np.log(1 - hy))



        # 迭代次数初始化为0
        epoch_number = 0
        X = np.hstack([np.ones((len(xTrain), 1)), xTrain])
        # a服从正态分布
        self.a = np.random.normal(size=(X.shape[1]))
        #进行迭代
        for i in range(self.epoch):
            original_a = self.a
            #判断梯度下降的更新条件
            self.a = self.a - self.lr * gradient_descent(self.a, X, yTrain)
            if (abs(loss_function(self.a, X,yTrain) - loss_function(original_a, X, yTrain)) < 1e-8):
                break
            #print("epoch:", i + 1)
            #print("train_loss:", loss_function(self.a, X, yTrain))
            train_loss.append(loss_function(self.a, X, yTrain))

        return self


    """
    np.hstack将参数元组的元素数组按水平方向进行叠加
    """
    def predict(self, xTest):
        X_b = np.hstack([np.ones((len(xTest), 1)),xTest])
        yPredict = self.sigmoid(X_b.dot(self.a))
        #概率>=0.5,类别为1   概率<0.5,类别为0
        yPredict = np.array(yPredict)
        for i in range(len(yPredict)):
            if yPredict[i]>=0.5:
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
            print('Precision = ',P)
            print('Recall = ',R)
            print('Error_rate = ', error_rate)
            print('F1 = ',F1_Score)

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
        calculate(Negative_TP, Negative_FP,  Negative_TN,  Negative_FN)






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






data = pd.read_csv('./iris.csv',encoding='gbk')
kf = KFold(n_splits=10)  # 10折
from sklearn.utils import shuffle
data = shuffle(data)

X = data.iloc[:, :4]
Y = data.iloc[:, 4]
i = 1
for train_index, test_index in kf.split(X,Y):
    epoch_number = []
    for i in range(1000):
        epoch_number.append(i + 1)
    train = data.iloc[train_index]
    test = data.iloc[test_index]
    xTrain = train.iloc[:, :4]
    yTrain = train.iloc[:, 4]
    xTest = test.iloc[:, :4]
    yTest = test.iloc[:, 4]
    train_loss = []
    logisticregression = LogisticRegression_GradientDescent(epoch=1000,lr=0.001)
    model =logisticregression.fit(xTrain,yTrain)
    print(" ",i)
    i = i + 1
    model.calculate(xTrain, yTrain, xTest, yTest)
    plt.plot(epoch_number, train_loss, color='red', label="train loss")
    plt.xlabel("epoch")
    plt.ylabel("train_loss")
    plt.show()