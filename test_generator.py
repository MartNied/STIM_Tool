from random import randint
import cv2
import numpy as np
import csv
import cv2
from ImageFunctions import cc_measurement
## Parameters for test generation

test_res = (520, 696)  # resolution of test image
N = 1000 # number of randomly generated ellipses
min_a = 3 # minimum value of major-semi-axis
max_a = 30 # max value of major-semi-axis
min_dist = 3 # min_dist value for a second ellipse lying around the original ellipse: with pars. a+min_dist, b+min_dist for creating a minimum distance parameter  to the neighbours
min_b = 2 # minimum value of minor-semi-axis
max_b = 5 # minimum value of minor-semi-axis


test_folderpath = "test_cases" #folderpath to saving folder where test data is stored
test_case = f"test_N_{N}_min_a_{min_a}_max_a_{max_a}_min_dist_{min_dist}_min_b_{min_b}_ max_b_{max_b}"

test_im = np.zeros(test_res, dtype=np.uint8) #new generated single ellipse
control_im = np.zeros(test_res, dtype=np.uint8) #check image for overlapping new ellipse with old image
merge_im = np.zeros(test_res, dtype=np.uint8) #merging old and new image 
blank_im = np.zeros(test_res, dtype=np.uint8)

data_dict = list() #list for storing dictionary for csv writing

N_count = 0 #counter for created structs in image

while(N_count < N):
    
    x = randint(0+max_a, test_res[0]-1-max_a) # choose random center position
    y = randint(0+max_a, test_res[1]-1-max_a)
    
    a = randint(min_a, max_a) # choose random major and minor axis
    b = randint(min_b, max_b)

    phi = randint(0, 360) # choose random orientation

    test_im = np.copy(blank_im) #create a test image were a minimum enclosing rectangle of the ellipse is generated and tested for overlaps with already generated ellipses
    ell_im = np.copy(blank_im) #cv2.ellipse is void function making it necessary to copy a blank array 

    cv2.ellipse(test_im, center=(y,x), axes=(a+min_dist, b+min_dist), angle=phi, color=(255,0,0), thickness=-1, startAngle=0, endAngle=360)

    if not (np.any(np.logical_and(control_im, test_im))): #compare old and new image for any overlapp between the ellipses
        
        cv2.ellipse(ell_im, center=(y,x), axes=(a,b), angle=phi, color=(255,0,0), thickness=-1, startAngle=0, endAngle=360) #generate new single ellipse
        control_im += ell_im #add test ellipse to control image

        # cv2.imshow("Generated ellipse", ell_im)
        # cv2.imshow("Generated MER", test_im)
        # cv2.imshow("Generated Image", control_im)
        # cv2.waitKey(0)
        
        N_count += 1 #increment struct count
        
        area = np.count_nonzero(ell_im) #calculate area
        length = max(a,b) * 2 #determine length

        eccentricity= np.sqrt(1-(b**2 / a**2)) #calculate eccentricity

        #calculate solidity
        contour, _ = cv2.findContours(
            test_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
        hull = cv2.convexHull(contour[0]) #calculate solidity
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area
        
        extent = float(area)/(a*b) #calculte extent

        data_dict.append({"Test Case": test_case, "a":a, "b":b ,"Area": area, "Length": length, #append to csv dict
                         "Eccentricity": eccentricity, "Solidity": solidity, "Extent": extent})

        

cv2.imshow("Generated Image", control_im)
cv2.waitKey(0)


## writing generated parameters to csv file 

csv_columns = ["Test Case", "a", "b", "Area", "Length",    # create header for csv
                       "Eccentricity", "Solidity", "Extent"]

with open(test_folderpath + "/" + test_case + ".csv", 'w') as csvfile:  # write data to csv
                writer = csv.DictWriter(
                    csvfile, fieldnames = csv_columns, dialect="excel", lineterminator="\n")
                writer.writeheader()
                for data in data_dict:
                    writer.writerow(data)

 
### write test image to png
cv2.imwrite(test_folderpath + "/" + test_case + ".png", control_im) #write generated image to path 


print("Test case generated!")