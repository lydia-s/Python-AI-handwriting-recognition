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
    
    dir_1 = './testing_type/Lydia/'
    dir_2 = './testing_type/Josh/'
    
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
     
       print 'The signatures at index: ' + str(index) + ' have: ' + str(total) + ' errors'        
       return  

retrieveFiles(dir_1, 0)
retrieveFiles(dir_2, 1)



percent = round((((all_images - errors)/(all_images))*100), 2)
print str(percent) + ' % accurate'




print '\nFinished identifying'
