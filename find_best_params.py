from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# carrega o arquivo do disco
data_frame = pd.read_csv("Adult - Gilberto e Henrique.csv")
x = data_frame.iloc[:, 0:14].values
y = data_frame.iloc[:, 14].values


# divide a base em 50% para treino, 25% para teste e 25% para validação
def do_partitions():
    global train_x
    global train_y
    global validation_x
    global test_x
    global validation_y
    global test_y

    train_x, aux_x, train_y, aux_y = train_test_split(x, y, test_size=0.5)
    validation_x, test_x, validation_y, test_y = train_test_split(aux_x, aux_y, test_size=0.5)


# função para medir a acurácia da predição
def accuracy(y_true: object, y_pred: object) -> object:
    return np.mean(np.equal(y_true, y_pred))


# usando um classificador dt
def dt():
    print('Decision Tree\nCalculando profundidade\t')
    aux_depth = 1
    aux_accuracy = 0.00
    for _ in range(20):
        do_partitions()
        for depth in range(1, 10):
            decisionTree = DecisionTreeClassifier(max_depth=depth)
            decisionTree = decisionTree.fit(train_x, train_y)
            if (accuracy(test_y, decisionTree.predict(test_x)) > aux_accuracy):
                aux_accuracy = accuracy(test_y, decisionTree.predict(test_x))
                aux_depth = depth
    print('Profundidade máxima = ', aux_depth)
    return (aux_depth)


# usando um classificador KNN k=1-3-5-7-9-11-13-15 * (não ponderado, ponderado pelo inverso da distância, ponderado por 1-distância normalizada)
def knn():
    print('\nK Nearest Neighbors\nCalculando K, Métrica de Distância\t')
    aux_accuracy = 0.00
    aux_weights = ''
    aux_k = 0

    for _ in range(20):
        do_partitions()
        for k in range(1, 16, 2):
            kNearestNeighbors_uniform = KNeighborsClassifier(n_neighbors=k, algorithm='auto', weights='uniform')
            kNearestNeighbors_uniform.fit(train_x, train_y)
            if (accuracy(test_y, kNearestNeighbors_uniform.predict(test_x)) > aux_accuracy):
                # print('swap to uniform')
                aux_accuracy = accuracy(test_y, kNearestNeighbors_uniform.predict(test_x))
                aux_weights = 'uniform'
                aux_k = k

            kNearestNeighbors_distance = KNeighborsClassifier(n_neighbors=k, algorithm='auto',
                                                              weights='distance')
            kNearestNeighbors_distance.fit(train_x, train_y)
            if (accuracy(test_y, kNearestNeighbors_distance.predict(test_x)) > aux_accuracy):
                # print('swap to distance')
                aux_accuracy = accuracy(test_y, kNearestNeighbors_distance.predict(test_x))
                aux_weights = 'distance'
                aux_k = k
                # print(accuracy(test_y, kNearestNeighbors_uniform.predict(test_x)),
                #       accuracy(test_y, kNearestNeighbors_distance.predict(test_x)))
    print('K = ', aux_k, 'Métrica de Distância = ', aux_weights)
    return (aux_weights, aux_k)


# usando um classificador mlp
def mlp():
    print(
        '\nMulti Layer Perceptron\nCalculando Número de épocas de treino, taxa de aprendizagem, número de camadas escondidas\t')
    aux_accuracy = 0.00
    aux_hls = 0
    aux_mi = 0
    aux_learning_rate = ''

    for _ in range(20):
        do_partitions()
        for mi in range(50, 201, 25):
            for hls in range(1, 15):
                # print(_, mi, hls)

                multiLayerPerceptron_constant = MLPClassifier(hidden_layer_sizes=hls, activation='relu', max_iter=mi,
                                                              learning_rate='constant')
                multiLayerPerceptron_constant.fit(train_x, train_y)

                multiLayerPerceptron_invscaling = MLPClassifier(hidden_layer_sizes=hls, activation='relu', max_iter=mi,
                                                                learning_rate='invscaling')
                multiLayerPerceptron_invscaling.fit(train_x, train_y)

                multiLayerPerceptron_adaptive = MLPClassifier(hidden_layer_sizes=hls, activation='relu', max_iter=mi,
                                                              learning_rate='adaptive')
                multiLayerPerceptron_adaptive.fit(train_x, train_y)

                if (accuracy(test_y, multiLayerPerceptron_constant.predict(test_x)) > aux_accuracy):
                    aux_accuracy = accuracy(test_y, multiLayerPerceptron_constant.predict(test_x))
                    aux_learning_rate = 'constant'
                    aux_hls = hls
                    aux_mi = mi

                if (accuracy(test_y, multiLayerPerceptron_invscaling.predict(test_x)) > aux_accuracy):
                    aux_accuracy = accuracy(test_y, multiLayerPerceptron_invscaling.predict(test_x))
                    aux_learning_rate = 'invscaling'
                    aux_hls = hls
                    aux_mi = mi

                if (accuracy(test_y, multiLayerPerceptron_adaptive.predict(test_x)) > aux_accuracy):
                    aux_accuracy = accuracy(test_y, multiLayerPerceptron_adaptive.predict(test_x))
                    aux_learning_rate = 'adaptive'
                    aux_hls = hls
                    aux_mi = mi

    print('Número de épocas de treino = ', aux_mi, 'Taxa de aprendizagem = ', aux_learning_rate,
          'Número de camadas escondidas = ', aux_hls)
    return (aux_mi, aux_hls, aux_learning_rate)


# usando um classificador svm
def svm():
    print('\nSupport Vector Machine\nCalculando Kernel e Penalidade\t')
    aux_accuracy = 0.00
    aux_penalty = 0.00
    aux_kernel = ''

    for _ in range(20):
        do_partitions()
        for penalty in range(1, 51):
            supVectorMachine_poly = SVC(C=penalty / 10, kernel='poly')
            supVectorMachine_poly.fit(train_x, train_y)

            supVectorMachine_rbf = SVC(C=penalty / 10, kernel='rbf')
            supVectorMachine_rbf.fit(train_x, train_y)

            if (accuracy(test_y, supVectorMachine_poly.predict(test_x)) > aux_accuracy):
                aux_accuracy = accuracy(test_y, supVectorMachine_poly.predict(test_x))
                aux_kernel = 'poly'
                aux_penalty = penalty / 10

            if (accuracy(test_y, supVectorMachine_rbf.predict(test_x)) > aux_accuracy):
                aux_accuracy = accuracy(test_y, supVectorMachine_rbf.predict(test_x))
                aux_kernel = 'rbf'
                aux_penalty = penalty / 10
    print('Kernel = ', aux_kernel, 'Penalidade = ', aux_penalty)
    return (aux_penalty, aux_kernel)
