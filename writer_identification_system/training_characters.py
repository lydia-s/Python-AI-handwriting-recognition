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
    dir_1_l = './training_type/Lydia/1/'
    dir_2_l = './training_type/Lydia/2/'
    dir_3_l = './training_type/Lydia/3/'
    dir_4_l = './training_type/Lydia/4/'
    dir_5_l = './training_type/Lydia/5/'
    dir_6_l = './training_type/Lydia/6/'
    dir_7_l = './training_type/Lydia/7/'
    dir_8_l = './training_type/Lydia/8/'
    dir_9_l = './training_type/Lydia/9/'
    dir_a_l = './training_type/Lydia/a/'
    dir_b_l = './training_type/Lydia/b/'
    dir_c_l = './training_type/Lydia/c/'
    dir_d_l = './training_type/Lydia/d/'
    dir_e_l = './training_type/Lydia/e/'
    dir_f_l = './training_type/Lydia/f/'
    dir_g_l = './training_type/Lydia/g/'
    dir_h_l = './training_type/Lydia/h/'
    dir_i_l = './training_type/Lydia/i/'
    dir_j_l = './training_type/Lydia/j/'
    dir_k_l = './training_type/Lydia/k/'
    dir_l_l = './training_type/Lydia/l/'
    dir_m_l = './training_type/Lydia/m/'
    dir_n_l = './training_type/Lydia/n/'
    dir_o_l = './training_type/Lydia/o/'
    dir_p_l = './training_type/Lydia/p/'
    dir_q_l = './training_type/Lydia/q/'
    dir_r_l = './training_type/Lydia/r/'
    dir_s_l = './training_type/Lydia/s/'
    dir_t_l = './training_type/Lydia/t/'
    dir_u_l = './training_type/Lydia/u/'
    dir_v_l = './training_type/Lydia/v/'
    dir_w_l = './training_type/Lydia/w/'
    dir_x_l = './training_type/Lydia/x/'
    dir_y_l = './training_type/Lydia/y/'
    dir_z_l = './training_type/Lydia/z/'


    dir_1_j = './training_type/Josh/1/'
    dir_2_j = './training_type/Josh/2/'
    dir_3_j = './training_type/Josh/3/'
    dir_4_j = './training_type/Josh/4/'
    dir_5_j = './training_type/Josh/5/'
    dir_6_j = './training_type/Josh/6/'
    dir_7_j = './training_type/Josh/7/'
    dir_8_j = './training_type/Josh/8/'
    dir_9_j = './training_type/Josh/9/'
    dir_a_j = './training_type/Josh/a/'
    dir_b_j = './training_type/Josh/b/'
    dir_c_j = './training_type/Josh/c/'
    dir_d_j = './training_type/Josh/d/'
    dir_e_j = './training_type/Josh/e/'
    dir_f_j = './training_type/Josh/f/'
    dir_g_j = './training_type/Josh/g/'
    dir_h_j = './training_type/Josh/h/'
    dir_i_j = './training_type/Josh/i/'
    dir_j_j = './training_type/Josh/j/'
    dir_k_j = './training_type/Josh/k/'
    dir_l_j = './training_type/Josh/l/'
    dir_m_j = './training_type/Josh/m/'
    dir_n_j = './training_type/Josh/n/'
    dir_o_j = './training_type/Josh/o/'
    dir_p_j = './training_type/Josh/p/'
    dir_q_j = './training_type/Josh/q/'
    dir_r_j = './training_type/Josh/r/'
    dir_s_j = './training_type/Josh/s/'
    dir_t_j = './training_type/Josh/t/'
    dir_u_j = './training_type/Josh/u/'
    dir_v_j = './training_type/Josh/v/'
    dir_w_j = './training_type/Josh/w/'
    dir_x_j = './training_type/Josh/x/'
    dir_y_j = './training_type/Josh/y/'
    dir_z_j = './training_type/Josh/z/'
    
    dataset_1 = []
    identity_1 = []
    dataset_2 = []
    identity_2 = []
    dataset_3 = []
    identity_3 = []
    dataset_4 = []
    identity_4 = []
    dataset_5 = []
    identity_5 = []
    dataset_6 = []
    identity_6 = []
    dataset_7 = []
    identity_7 = []
    dataset_8 = []
    identity_8 = []
    dataset_9 = []
    identity_9 = []
    dataset_a = []
    identity_a = []
    dataset_b = []
    identity_b = []
    dataset_c = []
    identity_c = []
    dataset_d = []
    identity_d = []
    dataset_e = []
    identity_e = []
    dataset_f = []
    identity_f = []
    dataset_g = []
    identity_g = []
    dataset_h = []
    identity_h = []
    dataset_i = []
    identity_i = []
    dataset_j = []
    identity_j = []
    dataset_k = []
    identity_k = []
    dataset_l = []
    identity_l = []
    dataset_m = []
    identity_m = []
    dataset_n = []
    identity_n = []
    dataset_o = []
    identity_o = []
    dataset_p = []
    identity_p = []
    dataset_q = []
    identity_q = []
    dataset_r = []
    identity_r = []
    dataset_s = []
    identity_s = []
    dataset_t = []
    identity_t = []
    dataset_u = []
    identity_u = []
    dataset_v = []
    identity_v = []
    dataset_w = []
    identity_w = []
    dataset_x = []
    identity_x = []
    dataset_y = []
    identity_y = []
    dataset_z = []
    identity_z = []


   

def retrieveFiles(path, index, dataset, identity):
    filenames = sorted([filename for filename in os.listdir(path) if filename.endswith('.png')])
    filenames = [path+filename for filename in filenames]
    for filename in filenames:
	    image = imread(filename,1)
            #flatten it
            #image = imresize(image, (15,15))
            hog_features = hog(image, orientations=12, pixels_per_cell=(3, 3), cells_per_block=(2, 2))
            dataset.append(hog_features)
            identity.append(index)
    return

retrieveFiles(dir_1_l, 1, dataset_1, identity_1)
retrieveFiles(dir_2_l, 1, dataset_2, identity_2)
retrieveFiles(dir_3_l, 1, dataset_3, identity_3)
retrieveFiles(dir_4_l, 1, dataset_4, identity_4)
retrieveFiles(dir_5_l, 1, dataset_5, identity_5)
retrieveFiles(dir_6_l, 1, dataset_6, identity_6)
retrieveFiles(dir_7_l, 1, dataset_7, identity_7)
retrieveFiles(dir_8_l, 1, dataset_8, identity_8)
retrieveFiles(dir_9_l, 1, dataset_9, identity_9)
retrieveFiles(dir_a_l, 1, dataset_a, identity_a)
retrieveFiles(dir_b_l, 1, dataset_b, identity_b)
retrieveFiles(dir_c_l, 1, dataset_c, identity_c)
retrieveFiles(dir_d_l, 1, dataset_d, identity_d)
retrieveFiles(dir_e_l, 1, dataset_e, identity_e)
retrieveFiles(dir_f_l, 1, dataset_f, identity_f)
retrieveFiles(dir_g_l, 1, dataset_g, identity_g)
retrieveFiles(dir_h_l, 1, dataset_h, identity_h)
retrieveFiles(dir_i_l, 1, dataset_i, identity_i)
retrieveFiles(dir_j_l, 1, dataset_j, identity_j)
retrieveFiles(dir_k_l, 1, dataset_k, identity_k)
retrieveFiles(dir_l_l, 1, dataset_l, identity_l)
retrieveFiles(dir_m_l, 1, dataset_m, identity_m)
retrieveFiles(dir_n_l, 1, dataset_n, identity_n)
retrieveFiles(dir_o_l, 1, dataset_o, identity_o)
retrieveFiles(dir_p_l, 1, dataset_p, identity_p)
retrieveFiles(dir_q_l, 1, dataset_q, identity_q)
retrieveFiles(dir_r_l, 1, dataset_r, identity_r)
retrieveFiles(dir_s_l, 1, dataset_s, identity_s)
retrieveFiles(dir_t_l, 1, dataset_t, identity_t)
retrieveFiles(dir_u_l, 1, dataset_u, identity_u)
retrieveFiles(dir_v_l, 1, dataset_v, identity_v)
retrieveFiles(dir_w_l, 1, dataset_w, identity_w)
retrieveFiles(dir_x_l, 1, dataset_x, identity_x)
retrieveFiles(dir_y_l, 1, dataset_y, identity_y)
retrieveFiles(dir_z_l, 1, dataset_z, identity_z)

retrieveFiles(dir_1_j, 0, dataset_1, identity_1)
retrieveFiles(dir_2_j, 0, dataset_2, identity_2)
retrieveFiles(dir_3_j, 0, dataset_3, identity_3)
retrieveFiles(dir_4_j, 0, dataset_4, identity_4)
retrieveFiles(dir_5_j, 0, dataset_5, identity_5)
retrieveFiles(dir_6_j, 0, dataset_6, identity_6)
retrieveFiles(dir_7_j, 0, dataset_7, identity_7)
retrieveFiles(dir_8_j, 0, dataset_8, identity_8)
retrieveFiles(dir_9_j, 0, dataset_9, identity_9)
retrieveFiles(dir_a_j, 0, dataset_a, identity_a)
retrieveFiles(dir_b_j, 0, dataset_b, identity_b)
retrieveFiles(dir_c_j, 0, dataset_c, identity_c)
retrieveFiles(dir_d_j, 0, dataset_d, identity_d)
retrieveFiles(dir_e_j, 0, dataset_e, identity_e)
retrieveFiles(dir_f_j, 0, dataset_f, identity_f)
retrieveFiles(dir_g_j, 0, dataset_g, identity_g)
retrieveFiles(dir_h_j, 0, dataset_h, identity_h)
retrieveFiles(dir_i_j, 0, dataset_i, identity_i)
retrieveFiles(dir_j_j, 0, dataset_j, identity_j)
retrieveFiles(dir_k_j, 0, dataset_k, identity_k)
retrieveFiles(dir_l_j, 0, dataset_l, identity_l)
retrieveFiles(dir_m_j, 0, dataset_m, identity_m)
retrieveFiles(dir_n_j, 0, dataset_n, identity_n)
retrieveFiles(dir_o_j, 0, dataset_o, identity_o)
retrieveFiles(dir_p_j, 0, dataset_p, identity_p)
retrieveFiles(dir_q_j, 0, dataset_q, identity_q)
retrieveFiles(dir_r_j, 0, dataset_r, identity_r)
retrieveFiles(dir_s_j, 0, dataset_s, identity_s)
retrieveFiles(dir_t_j, 0, dataset_t, identity_t)
retrieveFiles(dir_u_j, 0, dataset_u, identity_u)
retrieveFiles(dir_v_j, 0, dataset_v, identity_v)
retrieveFiles(dir_w_j, 0, dataset_w, identity_w)
retrieveFiles(dir_x_j, 0, dataset_x, identity_x)
retrieveFiles(dir_y_j, 0, dataset_y, identity_y)
retrieveFiles(dir_z_j, 0, dataset_z, identity_z)


#create the SVC
clf1 = LinearSVC(dual=False,verbose=1)
clf2 = LinearSVC(dual=False,verbose=1)
clf3 = LinearSVC(dual=False,verbose=1)
clf4 = LinearSVC(dual=False,verbose=1)
clf5 = LinearSVC(dual=False,verbose=1)
clf6 = LinearSVC(dual=False,verbose=1)
clf7 = LinearSVC(dual=False,verbose=1)
clf8 = LinearSVC(dual=False,verbose=1)
clf9 = LinearSVC(dual=False,verbose=1)
clfa = LinearSVC(dual=False,verbose=1)
clfb = LinearSVC(dual=False,verbose=1)
clfc = LinearSVC(dual=False,verbose=1)
clfd = LinearSVC(dual=False,verbose=1)
clfe = LinearSVC(dual=False,verbose=1)
clff = LinearSVC(dual=False,verbose=1)
clfg = LinearSVC(dual=False,verbose=1)
clfh = LinearSVC(dual=False,verbose=1)
clfi = LinearSVC(dual=False,verbose=1)
clfj = LinearSVC(dual=False,verbose=1)
clfk = LinearSVC(dual=False,verbose=1)
clfl = LinearSVC(dual=False,verbose=1)
clfm = LinearSVC(dual=False,verbose=1)
clfn = LinearSVC(dual=False,verbose=1)
clfo = LinearSVC(dual=False,verbose=1)
clfp = LinearSVC(dual=False,verbose=1)
clfq = LinearSVC(dual=False,verbose=1)
clfr = LinearSVC(dual=False,verbose=1)
clfs = LinearSVC(dual=False,verbose=1)
clft = LinearSVC(dual=False,verbose=1)
clfu = LinearSVC(dual=False,verbose=1)
clfv = LinearSVC(dual=False,verbose=1)
clfw = LinearSVC(dual=False,verbose=1)
clfx = LinearSVC(dual=False,verbose=1)
clfy = LinearSVC(dual=False,verbose=1)
clfz = LinearSVC(dual=False,verbose=1)


    #train the svm
clf1.fit(dataset_1, identity_1)
clf2.fit(dataset_2, identity_2)
clf3.fit(dataset_3, identity_3)
clf4.fit(dataset_4, identity_4)
clf5.fit(dataset_5, identity_5)
clf6.fit(dataset_6, identity_6)
clf7.fit(dataset_7, identity_7)
clf8.fit(dataset_8, identity_8)
clf9.fit(dataset_9, identity_9)
clfa.fit(dataset_a, identity_a)
clfb.fit(dataset_b, identity_b)
clfc.fit(dataset_c, identity_c)
clfd.fit(dataset_d, identity_d)
clfe.fit(dataset_e, identity_e)
clff.fit(dataset_f, identity_f)
clfg.fit(dataset_g, identity_g)
clfh.fit(dataset_h, identity_h)
clfi.fit(dataset_i, identity_i)
clfj.fit(dataset_j, identity_j)
clfk.fit(dataset_k, identity_j)
clfl.fit(dataset_l, identity_l)
clfm.fit(dataset_m, identity_m)
clfn.fit(dataset_n, identity_n)
clfo.fit(dataset_o, identity_o)
clfp.fit(dataset_p, identity_p)
clfq.fit(dataset_q, identity_q)
clfr.fit(dataset_r, identity_r)
clfs.fit(dataset_s, identity_s)
clft.fit(dataset_t, identity_t)
clfu.fit(dataset_u, identity_u)
clfv.fit(dataset_v, identity_v)
clfw.fit(dataset_w, identity_w)
clfx.fit(dataset_x, identity_x)
clfy.fit(dataset_y, identity_y)
clfz.fit(dataset_z, identity_z)

#pickle it - save it to a file
pickle.dump( clf1, open( "character.identifier1", "wb" ) )
pickle.dump( clf2, open( "character.identifier2", "wb" ) )
pickle.dump( clf3, open( "character.identifier3", "wb" ) )
pickle.dump( clf4, open( "character.identifier4", "wb" ) )
pickle.dump( clf5, open( "character.identifier5", "wb" ) )
pickle.dump( clf6, open( "character.identifier6", "wb" ) )
pickle.dump( clf7, open( "character.identifier7", "wb" ) )
pickle.dump( clf8, open( "character.identifier8", "wb" ) )
pickle.dump( clf9, open( "character.identifier9", "wb" ) )
pickle.dump( clfa, open( "character.identifiera", "wb" ) )
pickle.dump( clfb, open( "character.identifierb", "wb" ) )
pickle.dump( clfc, open( "character.identifierc", "wb" ) )
pickle.dump( clfd, open( "character.identifierd", "wb" ) )
pickle.dump( clfe, open( "character.identifiere", "wb" ) )
pickle.dump( clff, open( "character.identifierf", "wb" ) )
pickle.dump( clfg, open( "character.identifierg", "wb" ) )
pickle.dump( clfh, open( "character.identifierh", "wb" ) )
pickle.dump( clfi, open( "character.identifieri", "wb" ) )
pickle.dump( clfj, open( "character.identifierj", "wb" ) )
pickle.dump( clfk, open( "character.identifierk", "wb" ) )
pickle.dump( clfl, open( "character.identifierl", "wb" ) )
pickle.dump( clfm, open( "character.identifierm", "wb" ) )
pickle.dump( clfn, open( "character.identifiern", "wb" ) )
pickle.dump( clfo, open( "character.identifiero", "wb" ) )
pickle.dump( clfp, open( "character.identifierp", "wb" ) )
pickle.dump( clfq, open( "character.identifierq", "wb" ) )
pickle.dump( clfr, open( "character.identifierr", "wb" ) )
pickle.dump( clfs, open( "character.identifiers", "wb" ) )
pickle.dump( clft, open( "character.identifiert", "wb" ) )
pickle.dump( clfu, open( "character.identifieru", "wb" ) )
pickle.dump( clfv, open( "character.identifierv", "wb" ) )
pickle.dump( clfw, open( "character.identifierw", "wb" ) )
pickle.dump( clfx, open( "character.identifierx", "wb" ) )
pickle.dump( clfy, open( "character.identifiery", "wb" ) )
pickle.dump( clfz, open( "character.identifierz", "wb" ) )



