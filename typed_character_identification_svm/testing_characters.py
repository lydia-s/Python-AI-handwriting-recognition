from __future__ import division
from skimage.feature import hog
from scipy.misc import imread,imresize
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import pickle
import os
import math


global errors 

if __name__ == '__main__':

    #paths for the testing data
    
    dir_1 = './testing_type/1/'
    dir_2 = './testing_type/2/'
    dir_3 = './testing_type/3/'
    dir_4 = './testing_type/4/'
    dir_5 = './testing_type/5/'
    dir_6 = './testing_type/6/'
    dir_7 = './testing_type/7/'
    dir_8 = './testing_type/8/'
    dir_9 = './testing_type/9/'
    dir_a = './testing_type/a/'
    dir_b = './testing_type/b/'
    dir_c = './testing_type/c/'
    dir_d = './testing_type/d/'
    dir_e = './testing_type/e/'
    dir_f = './testing_type/f/'
    dir_g = './testing_type/g/'
    dir_h = './testing_type/h/'
    dir_i = './testing_type/i/'
    dir_j = './testing_type/j/'
    dir_k = './testing_type/k/'
    dir_l = './testing_type/l/'
    dir_m = './testing_type/m/'
    dir_n = './testing_type/n/'
    dir_o = './testing_type/o/'
    dir_p = './testing_type/p/'
    dir_q = './testing_type/q/'
    dir_r = './testing_type/r/'
    dir_s = './testing_type/s/'
    dir_t = './testing_type/t/'
    dir_u = './testing_type/u/'
    dir_v = './testing_type/v/'
    dir_w = './testing_type/w/'
    dir_x = './testing_type/x/'
    dir_y = './testing_type/y/'
    dir_z = './testing_type/z/'
    
    errors = 0
    all_images = 0
  
    
    def retrieveFiles(path, index):
       filenames = sorted([filename for filename in os.listdir(path) if filename.endswith('.png')]) #Only use png files
       filenames = [path+filename for filename in filenames] #add rest of path to png file
       clf = pickle.load( open("character.identifier","rb")) #loads the identifier

       total = 0
  
       

       for filename in filenames: #for filename in filenames in the current directory
	    image = imread(filename) #read the image
            #flatten it
            #image = imresize(image, (200,200))
            
            hog_features = hog(image, orientations=12, pixels_per_cell=(4, 4),  cells_per_block=(2, 2)) #get the hog feature for the image
            hog_features = hog_features.reshape(1, -1) # reshape the hog_features
            
            
            
            result_type = clf.predict(hog_features) #get result_type from hog_features
            global all_images
            counter = 0
            for i in result_type:
                 all_images += 1
                 #print i
                 if i != index:
                    
                    counter = counter + 1
            total = counter + total      
               
            
            #print(result_type)
       global errors
       
       errors = errors + total
     
       print 'The characters at index: ' + str(index) + ' have: ' + str(total) + ' errors'        
       return  

retrieveFiles(dir_1, 0)
retrieveFiles(dir_2, 1)
retrieveFiles(dir_3, 2)
retrieveFiles(dir_4, 3)
retrieveFiles(dir_5, 4)
retrieveFiles(dir_6, 5)
retrieveFiles(dir_7, 6)
retrieveFiles(dir_8, 7)
retrieveFiles(dir_9, 8)
retrieveFiles(dir_a, 9)
retrieveFiles(dir_b, 10)
retrieveFiles(dir_c, 11)
retrieveFiles(dir_d, 12)
retrieveFiles(dir_e, 13)
retrieveFiles(dir_f, 14)
retrieveFiles(dir_g, 15)
retrieveFiles(dir_h, 16)
retrieveFiles(dir_i, 17)
retrieveFiles(dir_j, 18)
retrieveFiles(dir_k, 19)
retrieveFiles(dir_l, 20)
retrieveFiles(dir_m, 21)
retrieveFiles(dir_n, 22)
retrieveFiles(dir_o, 23)
retrieveFiles(dir_p, 24)
retrieveFiles(dir_q, 25)
retrieveFiles(dir_r, 26)
retrieveFiles(dir_s, 27)
retrieveFiles(dir_t, 28)
retrieveFiles(dir_u, 29)
retrieveFiles(dir_v, 30)
retrieveFiles(dir_w, 31)
retrieveFiles(dir_x, 32)
retrieveFiles(dir_y, 33)
retrieveFiles(dir_z, 34)


percent = round((((all_images - errors)/(all_images))*100), 2)
print str(percent) + ' % accurate'




print '\nFinished identifying'
