from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#carrega o arquivo do disco
data_frame = pd.read_csv("Adult - Gilberto e Henrique.csv")
x = data_frame.iloc[:, 0:14].values
y = data_frame.iloc[:, 14].values

#divide a base em 50% para treino, 25% para teste e 25% para validação
def redo_ds_partitions():
    global train_x
    global train_y
    global validation_x
    global test_x
    global validation_y
    global test_y

    train_x, aux_x, train_y, aux_y = train_test_split(x, y, test_size=0.5)
    validation_x, test_x, validation_y, test_y = train_test_split(aux_x, aux_y, test_size=0.5)

#usando um classificador dst
def dst():
    decisionTree = DecisionTreeClassifier()
    decisionTree = decisionTree.fit(train_x, train_y)
    return accuracy(test_y, decisionTree.predict(test_x))

#usando um classificador KNN
def knn():
    kNearestNeighbors = KNeighborsClassifier(n_neighbors=7, algorithm='auto', weights='distance')
    kNearestNeighbors.fit(train_x, train_y)
    return accuracy(test_y, kNearestNeighbors.predict(test_x))

#usando um classificador nb
def nb():
    naiveBayes = BernoulliNB()
    naiveBayes.fit(train_x, train_y)
    return accuracy(test_y, naiveBayes.predict(test_x))

#usando um classificador mlp
def mlp():
    multiLayerPerceptron = MLPClassifier(hidden_layer_sizes=3, activation='relu', max_iter=200)
    multiLayerPerceptron.fit(train_x, train_y)
    return accuracy(test_y, multiLayerPerceptron.predict(test_x))

#usando um classificador svm
def svm():
    supVectorMachine = SVC()
    supVectorMachine.fit(train_x, train_y)
    return accuracy(test_y, supVectorMachine.predict(test_x))

#função para medir a acurácia da predição
def accuracy(y_true: object, y_pred: object) -> object:
    return np.mean(np.equal(y_true, y_pred))
