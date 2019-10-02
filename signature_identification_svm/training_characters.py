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

if __name__ == '__main__':

#paths for the training samples 
    dir_1 = './training_type/Lydia/'
    dir_2 = './training_type/Josh/'
    
    

    
dataset = []
identity = []

def retrieveFiles(path, index, dataset, identity):
    filenames = sorted([filename for filename in os.listdir(path) if filename.endswith('.png')])
    filenames = [path+filename for filename in filenames]
    for filename in filenames:
	    image = imread(filename,1)
            #flatten it
            #image = imresize(image, (200,200))
            hog_features = hog(image, orientations=12, pixels_per_cell=(4, 4), cells_per_block=(2, 2))
            dataset.append(hog_features)
            identity.append(index)
    return
retrieveFiles(dir_1, 0, dataset, identity)
retrieveFiles(dir_2, 1, dataset, identity)

#create the SVC
clf = LinearSVC(dual=False,verbose=1)
    #train the svm
clf.fit(dataset, identity)
#pickle it - save it to a file
pickle.dump( clf, open( "character.identifier", "wb" ) )


