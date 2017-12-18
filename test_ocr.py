import numpy as np
import pickle
import argparse
import svhn_data
import features_generator as fg

parser = argparse.ArgumentParser()
parser.add_argument('-data_path')
parser.add_argument('-model_path')
args = parser.parse_args()

patch_size = 8
patch_stride = 1

data_generator = svhn_data.SVHNData(args.data_path, patch_size, patch_stride)
data_generator.get_mean_std(100)

test_generator = data_generator.generate('test')

centers_fname = '/home/a.boiarov/Projects/spsa_clustering_gmm_log/centers.npy'
centers = np.load(centers_fname)

features_gen = fg.FeaturesGenerator(centers, patch_size, patch_stride, 5, 5, data_generator.mean,
                                    data_generator.std)

cl_test_num = data_generator.test_number
X_test = []
y_test = []
for i in range(cl_test_num-10):
    if i % 1000 == 0:
        print(i)
    test_data = next(test_generator)
    x = features_gen.forward(data_generator.test_gray[i])
    X_test.append(x)
    y_test.append(data_generator.test_labels[i])

clf = pickle.load(open(args.model_path, 'rb'))
y_pred = clf.predict(X_test)

acc = 0
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        acc += 1

print('Acc: {0}'.format(acc / len(y_test)))


