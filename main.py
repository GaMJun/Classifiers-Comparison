import classifiers
import find_best_params
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

dt = []
knn = []
mlp = []
nb = []
svm = []

print('Calculando Melhores Parametros\n')
bp_dt = find_best_params.dt()
bp_knn = find_best_params.knn()
bp_mlp = find_best_params.mlp()
print('Naive Bayes\nUsando Configuração Padrão')
bp_svm = find_best_params.svm()

for index in range(20):
    classifiers.redo_ds_partitions()
    dt.append(classifiers.dt(bp_dt))
    print('dt ', index)
    knn.append(classifiers.knn(bp_knn))
    print('knn', index)
    mlp.append(classifiers.mlp(bp_mlp))
    print('mlp', index)
    nb.append(classifiers.nb())
    print('nb', index)
    svm.append(classifiers.svm(bp_svm))
    print('svm', index)

with open("results.csv", "w") as fp:
    fp.write("Dt, Knn, Mlp, Nb, Svm\n")
    for index in range(20):
        fp.write("%f, %f, %f, %f, %f\n" % (dt[index], knn[index], mlp[index], nb[index], svm[index]))
    fp.write("\n%f, %f, %f, %f, %f\n" % (np.mean(dt), np.mean(knn), np.mean(mlp), np.mean(nb), np.mean(svm)))
