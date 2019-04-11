import classifiers
import numpy as np

dst = []
knn = []
mlp = []
nb = []
svm = []

for index in range(10):
    classifiers.redo_ds_partitions()
    dst.append(classifiers.dst())
    print('dst ', index)
    knn.append(classifiers.knn())
    print('knn', index)
    mlp.append(classifiers.mlp())
    print('mlp', index)
    nb.append(classifiers.nb())
    print('nb', index)
    svm.append(classifiers.svm())
    print('svm', index)

with open("results.csv", "w") as fp:
    fp.write("Dst, Knn, Mlp, Nb, Svm\n")
    for index in range(10):
        fp.write("%f, %f, %f, %f, %f\n" % (dst[index], knn[index], mlp[index], nb[index], svm[index]))

    fp.write("\n")

    fp.write("%f, %f, %f, %f, %f\n" % (np.mean(dst), np.mean(knn), np.mean(mlp), np.mean(nb), np.mean(svm)))
