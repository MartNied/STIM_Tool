import cv2
import numpy as np
from pathlib import Path

from numpy.core.fromnumeric import clip


############ function implementations ####################### 


def contrast_cut(im, clip_val, dataType="uint8"):
    """Clipping contrast values at certain threshold value
        im - input image
        clip_val - above this value the intensity will be clipped
        dataType - specifing input image Datatype
    """
    assert isinstance(clip_val, int)  #Check if clipval is int

    if  dataType == "uint8":        # 8 bit Case
        
        dyn_range = 2**8

        fp = [0, dyn_range-1]           #Input image Range
        xp = [0, clip_val]              #Input image will be mapped on full dynamic range withing this interval.
    
        x = np.arange(dyn_range)                        #create axis for LUT
        table = np.interp(x, xp, fp).astype(dataType)   #Create LUT
        im_clipped = cv2.LUT(im, table)                 #Transform image
        
    elif dataType == "uint16":      # 16 bit Case

        dyn_range = 2**16
        im[im > clip_val] = clip_val

        im_clipped = cv2.normalize(im, None, alpha=0, beta=dyn_range-1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)

    return im_clipped



def cc_filter(bin_im, min_area=1, max_area=520*696, min_eccentricity=0, max_eccentricity=1, connectivity=4):
    
    output = cv2.connectedComponentsWithStats(bin_im, connectivity=connectivity) #Connected component analysis
    
    (numLabels, labels, stats, centroids) = output
    
    filt_idx = np.where(np.logical_and(stats[:, cv2.CC_STAT_AREA] >= min_area, stats[:, cv2.CC_STAT_AREA] <= max_area))[0]  #calculate index where CC-Area lies in the intervall [min_area, max_area] --> idx is label
    
    if filt_idx[0] == 0:        #remove 0 label (Background)
        filt_idx = filt_idx[1:filt_idx.size]   
    
    # for i in filt_idx:
    #     filt_labels[filt_labels != i] = 0  #set label to zero where area is smaller than threshold (area_min)
    
    pass

#####################################       Main code       ###########################

#Import

root = Path(".")
filePath = root.absolute() / "test_data" / "resting1.tif" 
filePath = str(filePath)
im = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)


#Preprocessing

# clip_val = int(im.max())
clip_val = 3000
im_clip_norm = contrast_cut(im, clip_val, dataType="uint16")
disp_image = cv2.convertScaleAbs(im_clip_norm, alpha=(2**8 / 2**16))

#Processing

_, im_th = cv2.threshold(disp_image, 140, 255, cv2.THRESH_BINARY, None)

output = cv2.connectedComponentsWithStats(im_th, connectivity=4)
(numLabels, labels, stats, centroids) = output


#Display images

# cv2.imshow("Threshold - image", im_th)
# cv2.imshow("Clipped Image", im_clip_norm)
# cv2.imshow("Converted Image", disp_image)
# cv2.imshow("8 Bit transform", im8)

cv2.imshow("Connected Component",((labels != 0)*255).astype("uint8"))
k = cv2.waitKey(0)


#Saving images

# cv2.imwrite("test_data/test_file.png", disp_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

print(0)