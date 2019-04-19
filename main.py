import inspect
import statistics

import classifiers
import numpy as np
from warnings import filterwarnings

filterwarnings('ignore')

dt = []
knn = []
mlp = []
nb = []
svm = []
params_dt = []
params_knn = []
params_mlp = []
params_svm = []

for index in range(20):
    print('\n===Iteração ', index, '===\n')
    classifiers.redo_ds_partitions()

    print('Calculando Melhores Parametros\n')
    bp_dt = classifiers.fbp_dt()
    params_dt.append(bp_dt)
    bp_knn = classifiers.fbp_knn()
    params_knn.append(bp_knn)
    bp_mlp = classifiers.fbp_mlp()
    params_mlp.append(bp_mlp)
    print('Naive Bayes\nUsando Configuração Padrão')
    bp_svm = classifiers.fbp_svm()
    params_svm.append(bp_svm)

    dt.append(classifiers.dt(bp_dt))
    knn.append(classifiers.knn(bp_knn))
    mlp.append(classifiers.mlp(bp_mlp))
    nb.append(classifiers.nb())
    svm.append(classifiers.svm(bp_svm))

with open("results.csv", "w") as fp:
    fp.write(
        "Dt, Depth, Knn, Weights, K, Mlp, Max Ite, Hidden Layers, Learning Rate, Nb, No Params, Svm, Kernel, Penalty\n")
    for index in range(20):
        fp.write("%f, %s, %f, %s, %s, %f, %s, %s, %s, %f, %s, %f, %s,%s\n" % (
            dt[index], params_dt[index], knn[index],'1-Distancia Normalizada' if inspect.isfunction(params_knn[index][0]) else params_knn[index][0], params_knn[index][1], mlp[index],
            params_mlp[index][0], params_mlp[index][1], params_mlp[index][2], nb[index], "Default", svm[index],
            params_svm[index][0], params_svm[index][1]))
    fp.write("\n%f, %s, %f, %s, %s, %f, %s, %s, %s, %f, %s, %f, %s,%s\n" % (
        np.mean(dt), '', np.mean(knn), '', '', np.mean(mlp), '', '', '', np.mean(nb), '', np.mean(svm), '', ''))
    fp.write("%f, %s, %f, %s, %s, %f, %s, %s, %s, %f, %s, %f, %s,%s\n" % (
        statistics.pstdev(dt), '', statistics.pstdev(knn), '', '', statistics.pstdev(mlp), '', '', '', statistics.pstdev(nb), '', statistics.pstdev(svm), '', ''))