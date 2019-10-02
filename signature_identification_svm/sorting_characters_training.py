import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.misc import imread,imresize,imsave
from skimage.segmentation import clear_border
#from skimage.morphology import label
from skimage.measure import regionprops
from skimage.measure import label

class Extract_Letters:
    def extractFile(self, filename, thresh):
        image = imread(filename,1)
    
        #apply threshold in order to make the image binary
        bw = image < thresh
    
        # remove artifacts connected to image border
        cleared = bw.copy()
        #clear_border(cleared)

        # label image regions
        label_image = label(cleared,neighbors=8)
        borders = np.logical_xor(bw, cleared)
        label_image[borders] = -1
    
    
        fig = plt.figure()
        #ax = fig.add_subplot(131)
        #ax.imshow(bw, cmap='jet')

        letters = list()
        order = list()
	final = list() 
        for region in regionprops(label_image):
            minc, minr, maxc, maxr = region.bbox
            # skip small images
            if maxc - minc > len(image)/150: # better to use height rather than area.
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)

		tl,tr,bl,br = region.bbox
        
		letter_raw = bw[tl:bl,tr:br]
		letter_norm = imresize(letter_raw.astype(np.float32) ,(30 ,90))
		final.append(letter_norm)
	return final
         


    def __init__(self):
        print "Extracting characters..."

start_time = time.time()
extract = Extract_Letters()
training_files = ['./ocr/training/training1.png', './ocr/training/training2.png']



name_counter = 600
sig1 = extract.extractFile(training_files[0], 190)
print len(sig1)
string_counter = 0
	
	
for i in sig1:
	if string_counter < 40:
		string_counter = 0
		
		imsave('./training_type/' + 'Lydia' + '/' + str(name_counter) + '_snippet.png', i)
		#print 'training character: ' + str(folder_string[string_counter]) + ' (' + str(name_counter) + '/' + str(len(letters)) + ')'
		string_counter += 1
		name_counter += 1

sig2 = extract.extractFile(training_files[1], 180)
print len(sig2)
string_counter = 0
for i in sig2:
	if string_counter < 40:
		string_counter = 0
		imsave('./training_type/' + 'Josh' + '/' + str(name_counter) + '_snippet.png', i)
		#print 'training character: ' + str(folder_string[string_counter]) + ' (' + str(name_counter) + '/' + str(len(letters)) + ')'
		string_counter += 1
		name_counter += 1


print time.time() - start_time, "seconds" 
