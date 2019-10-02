import numpy as np 
import os
import itertools
import operator
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.feature import hog
from skimage import color, exposure
from scipy.misc import imread,imsave,imresize
import numpy.random as nprnd
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.svm import LinearSVC
import matplotlib
import pickle
from sklearn.neighbors import KNeighborsClassifier



if __name__ == '__main__':

#paths for the training samples 
    dir_1 = './training_type/1/'
    dir_2 = './training_type/2/'
    dir_3 = './training_type/3/'
    dir_4 = './training_type/4/'
    dir_5 = './training_type/5/'
    dir_6 = './training_type/6/'
    dir_7 = './training_type/7/'
    dir_8 = './training_type/8/'
    dir_9 = './training_type/9/'
    dir_a = './training_type/a/'
    dir_b = './training_type/b/'
    dir_c = './training_type/c/'
    dir_d = './training_type/d/'
    dir_e = './training_type/e/'
    dir_f = './training_type/f/'
    dir_g = './training_type/g/'
    dir_h = './training_type/h/'
    dir_i = './training_type/i/'
    dir_j = './training_type/j/'
    dir_k = './training_type/k/'
    dir_l = './training_type/l/'
    dir_m = './training_type/m/'
    dir_n = './training_type/n/'
    dir_o = './training_type/o/'
    dir_p = './training_type/p/'
    dir_q = './training_type/q/'
    dir_r = './training_type/r/'
    dir_s = './training_type/s/'
    dir_t = './training_type/t/'
    dir_u = './training_type/u/'
    dir_v = './training_type/v/'
    dir_w = './training_type/w/'
    dir_x = './training_type/x/'
    dir_y = './training_type/y/'
    dir_z = './training_type/z/'
    

    
dataset = []
identity = []

def retrieveFiles(path, index, dataset, identity):
    filenames = sorted([filename for filename in os.listdir(path) if filename.endswith('.png')])
    filenames = [path+filename for filename in filenames]
    for filename in filenames:
	    image = imread(filename,1)
            #flatten it
            #image = imresize(image, (200,200))
            hog_features = hog(image, orientations=12, pixels_per_cell=(3, 3), cells_per_block=(2, 2))
            dataset.append(hog_features)
            identity.append(index)
    return
retrieveFiles(dir_1, 0, dataset, identity)
retrieveFiles(dir_2, 1, dataset, identity)
retrieveFiles(dir_3, 2, dataset, identity)
retrieveFiles(dir_4, 3, dataset, identity)
retrieveFiles(dir_5, 4, dataset, identity)
retrieveFiles(dir_6, 5, dataset, identity)
retrieveFiles(dir_7, 6, dataset, identity)
retrieveFiles(dir_8, 7, dataset, identity)
retrieveFiles(dir_9, 8, dataset, identity)
retrieveFiles(dir_a, 9, dataset, identity)
retrieveFiles(dir_b, 10, dataset, identity)
retrieveFiles(dir_c, 11, dataset, identity)
retrieveFiles(dir_d, 12, dataset, identity)
retrieveFiles(dir_e, 13, dataset, identity)
retrieveFiles(dir_f, 14, dataset, identity)
retrieveFiles(dir_g, 15, dataset, identity)
retrieveFiles(dir_h, 16, dataset, identity)
retrieveFiles(dir_i, 17, dataset, identity)
retrieveFiles(dir_j, 18, dataset, identity)
retrieveFiles(dir_k, 19, dataset, identity)
retrieveFiles(dir_l, 20, dataset, identity)
retrieveFiles(dir_m, 21, dataset, identity)
retrieveFiles(dir_n, 22, dataset, identity)
retrieveFiles(dir_o, 23, dataset, identity)
retrieveFiles(dir_p, 24, dataset, identity)
retrieveFiles(dir_q, 25, dataset, identity)
retrieveFiles(dir_r, 26, dataset, identity)
retrieveFiles(dir_s, 27, dataset, identity)
retrieveFiles(dir_t, 28, dataset, identity)
retrieveFiles(dir_u, 29, dataset, identity)
retrieveFiles(dir_v, 30, dataset, identity)
retrieveFiles(dir_w, 31, dataset, identity)
retrieveFiles(dir_x, 32, dataset, identity)
retrieveFiles(dir_y, 33, dataset, identity)
retrieveFiles(dir_z, 34, dataset, identity)

#create the SVC
#clf = LinearSVC(dual=False,verbose=1)
    #train the svm
#clf.fit(dataset, identity)
#pickle it - save it to a file
#pickle.dump( clf, open( "character.identifier", "wb" ) )

k=1
knn_classifier=KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(dataset, identity)
pickle.dump(knn_classifier, open("character.identifier", "wb"))





