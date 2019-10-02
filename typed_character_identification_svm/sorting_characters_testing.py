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
    def extractFile(self, filename):
        image = imread(filename,1)
    
        #apply threshold in order to make the image binary
        bw = image < 120
    
        # remove artifacts connected to image border
        cleared = bw.copy()
        clear_border(cleared)

        # label image regions
        label_image = label(cleared,neighbors=8)
        borders = np.logical_xor(bw, cleared)
        label_image[borders] = -1
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(bw, cmap='jet')
    
        fig = plt.figure()
        #ax = fig.add_subplot(131)
        #ax.imshow(bw, cmap='jet')

        letters = list()
        order = list()
    
        for region in regionprops(label_image):
            minc, minr, maxc, maxr = region.bbox
            # skip small images
            if maxc - minc > len(image)/150: # better to use height rather than area.
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
                order.append(region.bbox)


        #sort the detected characters left->right, top->bottom
        lines = list()
        first_in_line = ''
        counter = 0

        #worst case scenario there can be 1 character per line
        for x in range(len(order)):
            lines.append([])
    
        for character in order:
            if first_in_line == '':
                first_in_line = character
                lines[counter].append(character)
            elif abs(character[0] - first_in_line[0]) < (first_in_line[2] - first_in_line[0]):
                lines[counter].append(character)
            elif abs(character[0] - first_in_line[0]) > (first_in_line[2] - first_in_line[0]):
                first_in_line = character
                counter += 1
                lines[counter].append(character)


        for x in range(len(lines)):       
            lines[x].sort(key=lambda tup: tup[1])

        final = list()
        prev_tr = 0
        prev_line_br = 0
        
        for i in range(len(lines)):
            for j in range(len(lines[i])):
                tl_2 = lines[i][j][1]
                bl_2 = lines[i][j][0]
                if tl_2 > prev_tr and bl_2 > prev_line_br:
                    tl,tr,bl,br = lines[i][j]
                    letter_raw = bw[tl:bl,tr:br]
                    letter_norm = imresize(letter_raw.astype(np.float32) ,(20 ,20))
                    final.append(letter_norm)
                    prev_tr = lines[i][j][3]
                if j == (len(lines[i])-1):
                    prev_line_br = lines[i][j][2]
            prev_tr = 0
            tl_2 = 0
        print 'Characters recognized: ' + str(len(final))
        return final


    def __init__(self):
        print "Extracting characters..."

start_time = time.time()
extract = Extract_Letters()
testing_files = ['./ocr/testing/adobe.png','./ocr/testing/shazam.png']


folder_string1 = 'InaregulatorydocumentfiledwiththeSECtodayAdobeannouncedthatchieftechnologyofficerKevinLynchwouldbetakinghisleaveasofthiscomingFridayOnMarch182013KevinLynchresignedfromhispositionasexecutivevicepresidentchieftechnologyofficerofAdobeSystemsIncorporatedeffectiveMarch222013topursueotheropportunitiesthefilingreadsLynchwhocametothecompanyin2005duringitsacquisitionofMacromedialedAdobeschargeintosomeofthemorecuttingedgeareasoftechnologyincludingmultiscreencomputingcloudcomputingandsocialmediaForagesAdobehadbeenrootedintheworkflowsoftheprintdesigncommunityLynchwasresponsibleforthecompanysshiftintowebpublishingstartingwithDreamweaverHealsooversawAdobesresearchandexperiencedesignteamsandwasasAdobeputsitinchargeofshapingAdobeslongtermtechnologyvisionandfocusinginnovationacrossthecompanyduringatransformativetimeRumorsaroundthewebhavepinpointedAppleasLynchsnextdestinationanditsnotanentirelynonsensicalrumorAdobestransitiontowebtechnologieshasbeennothingifnotprofitableApplestillagiantinconsumerhardwarecoulduseahelpinghandwhenitcomestomultiscreenfluiditysocialmediaandwebbasedsoftware'
folder_string1 = folder_string1.lower()
folder_string2 = 'NvidiasgraphicschipsarentusedjustforgamesanymoreSincethecompanymadeitschipsprogrammabletheyaretakingonfunctionsthatIntelmicroprocessorsdoAndmorethanevertheyarebeingusedforBigDatanumbercrunchingandsolvingproblemsthatpreviouslyhadtobedonebynongraphicssupercomputersJenHsunHuang(pictured)chiefexecutiveofNvidiasaidinakeynotespeechatthecompanysGPUTechconferencethattopenterpriseandmobileappcompaniesareusingGPUs(graphicsprocessingunits)tohandleBigDataanalyticsadvancesearchandtaskslikefacerecognitionTheseapplicationsarespreadacrossthecommercialandconsumerspacesCustomerswhoareusingNvidiaGPUsinserversmadebycomputermakersincludeShazamSalesforcecomandCortexicaTheyareusingNvidiaTeslaGPUsforcomputingtasksbeyondtraditionalgraphicsprocessingTheproblemsathandincludeaudiosearchsuchasShazamsservicethattellsyouwhatsongyourelisteningtoontheradioGPUsenableustohandleourtremendousprocessingneedsatasubstantialcostsavingsdeliveringtwicetheperformanceperdollarcomparedtoaCPUbasedsystemsaidJasonTituschieftechnologyofficerofShazamEntertainmentWeareaddingmillionsofvideoandforeignlanguageaudiotrackstoourexistingservicesandGPUacceleratorsgiveusawaytoachievescalablegrowthShazamcanidentifytheacousticfingerprintofsongsfroma27milliontrackdatabaseandthenserveanswersto300millionusersPeoplearedoing10millionsongsearchesaday'
folder_string2 = folder_string2.lower()
name_counter = 600

letters = extract.extractFile(testing_files[0])
string_counter = 0

for i in letters:
    if(folder_string1[string_counter] != '0' and folder_string1[string_counter] != '(' and folder_string1[string_counter] != ')'):
	imsave('./testing_type/' + str(folder_string1[string_counter]) + '/' + str(name_counter) + '_snippet.png', i)
		
	name_counter += 1
        
    string_counter += 1


letters = extract.extractFile(testing_files[1])
string_counter = 0
for i in letters:
    if(folder_string2[string_counter] != '0' and folder_string2[string_counter] != '(' and folder_string2[string_counter] != ')'):
	imsave('./testing_type/' + str(folder_string2[string_counter]) + '/' + str(name_counter) + '_snippet.png', i)
		
	name_counter += 1
        
    string_counter += 1


print time.time() - start_time, "seconds" 
