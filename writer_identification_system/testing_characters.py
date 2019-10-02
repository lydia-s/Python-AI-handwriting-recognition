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
    
    dir_1_l = './testing_type/Lydia/1/'
    dir_2_l = './testing_type/Lydia/2/'
    dir_3_l = './testing_type/Lydia/3/'
    dir_4_l = './testing_type/Lydia/4/'
    dir_5_l = './testing_type/Lydia/5/'
    dir_6_l = './testing_type/Lydia/6/'
    dir_7_l = './testing_type/Lydia/7/'
    dir_8_l = './testing_type/Lydia/8/'
    dir_9_l = './testing_type/Lydia/9/'
    dir_a_l = './testing_type/Lydia/a/'
    dir_b_l = './testing_type/Lydia/b/'
    dir_c_l = './testing_type/Lydia/c/'
    dir_d_l = './testing_type/Lydia/d/'
    dir_e_l = './testing_type/Lydia/e/'
    dir_f_l = './testing_type/Lydia/f/'
    dir_g_l = './testing_type/Lydia/g/'
    dir_h_l = './testing_type/Lydia/h/'
    dir_i_l = './testing_type/Lydia/i/'
    dir_j_l = './testing_type/Lydia/j/'
    dir_k_l = './testing_type/Lydia/k/'
    dir_l_l = './testing_type/Lydia/l/'
    dir_m_l = './testing_type/Lydia/m/'
    dir_n_l = './testing_type/Lydia/n/'
    dir_o_l = './testing_type/Lydia/o/'
    dir_p_l = './testing_type/Lydia/p/'
    dir_q_l = './testing_type/Lydia/q/'
    dir_r_l = './testing_type/Lydia/r/'
    dir_s_l = './testing_type/Lydia/s/'
    dir_t_l = './testing_type/Lydia/t/'
    dir_u_l = './testing_type/Lydia/u/'
    dir_v_l = './testing_type/Lydia/v/'
    dir_w_l = './testing_type/Lydia/w/'
    dir_x_l = './testing_type/Lydia/x/'
    dir_y_l = './testing_type/Lydia/y/'
    dir_z_l = './testing_type/Lydia/z/'


    dir_1_j = './testing_type/Josh/1/'
    dir_2_j = './testing_type/Josh/2/'
    dir_3_j = './testing_type/Josh/3/'
    dir_4_j = './testing_type/Josh/4/'
    dir_5_j = './testing_type/Josh/5/'
    dir_6_j = './testing_type/Josh/6/'
    dir_7_j = './testing_type/Josh/7/'
    dir_8_j = './testing_type/Josh/8/'
    dir_9_j = './testing_type/Josh/9/'
    dir_a_j = './testing_type/Josh/a/'
    dir_b_j = './testing_type/Josh/b/'
    dir_c_j = './testing_type/Josh/c/'
    dir_d_j = './testing_type/Josh/d/'
    dir_e_j = './testing_type/Josh/e/'
    dir_f_j = './testing_type/Josh/f/'
    dir_g_j = './testing_type/Josh/g/'
    dir_h_j = './testing_type/Josh/h/'
    dir_i_j = './testing_type/Josh/i/'
    dir_j_j = './testing_type/Josh/j/'
    dir_k_j = './testing_type/Josh/k/'
    dir_l_j = './testing_type/Josh/l/'
    dir_m_j = './testing_type/Josh/m/'
    dir_n_j = './testing_type/Josh/n/'
    dir_o_j = './testing_type/Josh/o/'
    dir_p_j = './testing_type/Josh/p/'
    dir_q_j = './testing_type/Josh/q/'
    dir_r_j = './testing_type/Josh/r/'
    dir_s_j = './testing_type/Josh/s/'
    dir_t_j = './testing_type/Josh/t/'
    dir_u_j = './testing_type/Josh/u/'
    dir_v_j = './testing_type/Josh/v/'
    dir_w_j = './testing_type/Josh/w/'
    dir_x_j = './testing_type/Josh/x/'
    dir_y_j = './testing_type/Josh/y/'
    dir_z_j = './testing_type/Josh/z/'
    
    errors = 0
    all_images = 0
    lydia_errors =0
    josh_errors = 0
  
    
    def retrieveFiles(path, index, name):
       filenames = sorted([filename for filename in os.listdir(path) if filename.endswith('.png')]) #Only use png files
       filenames = [path+filename for filename in filenames] #add rest of path to png file
       clf = pickle.load( open(name,"rb")) #loads the identifier

       total = 0
  
       

       for filename in filenames: #for filename in filenames in the current directory
	    image = imread(filename) #read the image
            #flatten it
            #image = imresize(image, (15,15))
            
            hog_features = hog(image, orientations=12, pixels_per_cell=(3, 3),  cells_per_block=(2, 2)) #get the hog feature for the image
            hog_features = hog_features.reshape(1, -1) # reshape the hog_features
            
            
            
            result_type = clf.predict(hog_features) #get result_type from hog_features
            global all_images
            global lydia_errors
            global josh_errors
            counter = 0
            for i in result_type:
                 all_images += 1
                
                 if i != index:          
                    counter = counter + 1
	            if index == 1:
				lydia_errors += 1
	            elif index == 0:
				josh_errors += 1
            total = counter + total      
               
            
            
       global errors
       
       errors = errors + total
         
       return  

retrieveFiles(dir_1_l, 1, "character.identifier1")
retrieveFiles(dir_2_l, 1, "character.identifier2")
retrieveFiles(dir_3_l, 1, "character.identifier3")
retrieveFiles(dir_4_l, 1, "character.identifier4")
retrieveFiles(dir_5_l, 1, "character.identifier5")
retrieveFiles(dir_6_l, 1, "character.identifier6")
retrieveFiles(dir_7_l, 1, "character.identifier7")
retrieveFiles(dir_8_l, 1, "character.identifier8")
retrieveFiles(dir_9_l, 1, "character.identifier9")
retrieveFiles(dir_a_l, 1, "character.identifiera")
retrieveFiles(dir_b_l, 1, "character.identifierb")
retrieveFiles(dir_c_l, 1, "character.identifierc")
retrieveFiles(dir_d_l, 1, "character.identifierd")
retrieveFiles(dir_e_l, 1, "character.identifiere")
retrieveFiles(dir_f_l, 1, "character.identifierf")
retrieveFiles(dir_g_l, 1, "character.identifierg")
retrieveFiles(dir_h_l, 1, "character.identifierh")
retrieveFiles(dir_i_l, 1, "character.identifieri")
retrieveFiles(dir_j_l, 1, "character.identifierj")
retrieveFiles(dir_k_l, 1, "character.identifierk")
retrieveFiles(dir_l_l, 1, "character.identifierl")
retrieveFiles(dir_m_l, 1, "character.identifierm")
retrieveFiles(dir_n_l, 1, "character.identifiern")
retrieveFiles(dir_o_l, 1, "character.identifiero")
retrieveFiles(dir_p_l, 1, "character.identifierp")
retrieveFiles(dir_q_l, 1, "character.identifierq")
retrieveFiles(dir_r_l, 1, "character.identifierr")
retrieveFiles(dir_s_l, 1, "character.identifiers")
retrieveFiles(dir_t_l, 1, "character.identifiert")
retrieveFiles(dir_u_l, 1, "character.identifieru")
retrieveFiles(dir_v_l, 1, "character.identifierv")
retrieveFiles(dir_w_l, 1, "character.identifierw")
retrieveFiles(dir_x_l, 1, "character.identifierx")
retrieveFiles(dir_y_l, 1, "character.identifiery")
retrieveFiles(dir_z_l, 1, "character.identifierz")

retrieveFiles(dir_1_j, 0, "character.identifier1")
retrieveFiles(dir_2_j, 0, "character.identifier2")
retrieveFiles(dir_3_j, 0, "character.identifier3")
retrieveFiles(dir_4_j, 0, "character.identifier4")
retrieveFiles(dir_5_j, 0, "character.identifier5")
retrieveFiles(dir_6_j, 0, "character.identifier6")
retrieveFiles(dir_7_j, 0, "character.identifier7")
retrieveFiles(dir_8_j, 0, "character.identifier8")
retrieveFiles(dir_9_j, 0, "character.identifier9")
retrieveFiles(dir_a_j, 0, "character.identifiera")
retrieveFiles(dir_b_j, 0, "character.identifierb")
retrieveFiles(dir_c_j, 0, "character.identifierc")
retrieveFiles(dir_d_j, 0, "character.identifierd")
retrieveFiles(dir_e_j, 0, "character.identifiere")
retrieveFiles(dir_f_j, 0, "character.identifierf")
retrieveFiles(dir_g_j, 0, "character.identifierg")
retrieveFiles(dir_h_j, 0, "character.identifierh")
retrieveFiles(dir_i_j, 0, "character.identifieri")
retrieveFiles(dir_j_j, 0, "character.identifierj")
retrieveFiles(dir_k_j, 0, "character.identifierk")
retrieveFiles(dir_l_j, 0, "character.identifierl")
retrieveFiles(dir_m_j, 0, "character.identifierm")
retrieveFiles(dir_n_j, 0, "character.identifiern")
retrieveFiles(dir_o_j, 0, "character.identifiero")
retrieveFiles(dir_p_j, 0, "character.identifierp")
retrieveFiles(dir_q_j, 0, "character.identifierq")
retrieveFiles(dir_r_j, 0, "character.identifierr")
retrieveFiles(dir_s_j, 0, "character.identifiers")
retrieveFiles(dir_t_j, 0, "character.identifiert")
retrieveFiles(dir_u_j, 0, "character.identifieru")
retrieveFiles(dir_v_j, 0, "character.identifierv")
retrieveFiles(dir_w_j, 0, "character.identifierw")
retrieveFiles(dir_x_j, 0, "character.identifierx")
retrieveFiles(dir_y_j, 0, "character.identifiery")
retrieveFiles(dir_z_j, 0, "character.identifierz")

print 'Josh characters had ' + str(josh_errors) + ' errors' 
print 'Lydia characters had ' + str(josh_errors) + ' errors' 

percent = round((((all_images - errors)/(all_images))*100), 2)
print str(percent) + ' % accurate'




print '\nFinished identifying'
