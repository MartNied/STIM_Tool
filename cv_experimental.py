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



def cc_filter_idx(output_cc, min_area=5, max_area=520*696, min_eccentricity=0, max_eccentricity=1):
    
    labels = output_cc[1]
    stats = output_cc[2]
    
    filt_idx = np.where(np.logical_and(stats[:, cv2.CC_STAT_AREA] >= min_area, stats[:, cv2.CC_STAT_AREA] <= max_area))[0]  #calculate index where CC-Area lies in the intervall [min_area, max_area] --> idx is label
    
    if filt_idx[0] == 0:     #remove 0 label (Background)
        filt_idx = filt_idx[1:filt_idx.size]   
    

    eccentricity = np.zeros_like(filt_idx)  #array for storing bool of the eccentricity condition (see later)
    
    for i in filt_idx:      #iterate only over CC where Area is in [min_area, max_area]
        
        struct_mask = ((labels == i)*255).astype("uint8")   #create uint8 binary image mask for connected component with label (index) i        
        
        contour,_ = cv2.findContours(struct_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)     #calculate outer contours of struct mask

        ## Fit contour and calculate eccentricity

        ellipse = cv2.fitEllipse(contour[0])     ## Danger 5 points are requiered for fitting an ellipse !!                                                   
        a = np.max(ellipse[1])      ### store big and small (a,b) semiaxis of the ellipse
        b = np.min(ellipse[1])
    
        ### Alternate form using rotated rectangle
        # rect = cv2.minAreaRect(contour[0])
        # a = np.max(rect[1])      ### store big and small (a,b) side of the rectangle
        # b = np.min(rect[1])
        
        assert(a > 0 & b > 0)
        
        eccentricity[i] = np.sqrt(1-(b**2 / a**2))
    
    filt_idx_eccent = np.where(np.logical_and(np.array(eccentricity) >= min_eccentricity, np.array(eccentricity) <= max_eccentricity))      #calculate indeces of filt_idx array!, where CC-Eccentricity lies in the intervall [min_eccentricity, max_eccentricity]
    
    filt_idx = filt_idx[filt_idx_eccent]  # Store the labels where Area and Eccentricity conditions are both fullfilled !!
    
    return filt_idx

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

filt_idx = cc_filter_idx(output, min_area=5, max_area=np.prod(im.shape), min_eccentricity=0, max_eccentricity=1)


#Display images

# cv2.imshow("Threshold - image", im_th)
# cv2.imshow("Clipped Image", im_clip_norm)
# cv2.imshow("Converted Image", disp_image)
# cv2.imshow("8 Bit transform", im8)

struct_bin = ((labels == 168)*255).astype("uint8") ##168 was a big one

contours,_ = cv2.findContours(struct_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]

#Copy with RGB channel

result = cv2.cvtColor(struct_bin, cv2.COLOR_GRAY2RGB)

### calc min enclosing rect
# rect = cv2.minAreaRect(cnt)
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# cv2.drawContours(result,[box],0,(0,0,255),2)

### bounding rect

# x,y,w,h = cv2.boundingRect(cnt)
# cv2.rectangle(result, (x,y), (x+w, y+h), (100, 100, 255), 2)

### min enclosing circle

# (x,y),radius = cv2.minEnclosingCircle(cnt)
# center = (int(x),int(y))
# radius = int(radius)
# cv2.circle(result ,center, radius,(0,255,0),2)


### Ellipse 
# ellipse = cv2.fitEllipse(cnt)
# cv2.ellipse(result ,ellipse,(0,255,0),2)


#display image
# cv2.imshow("Connected Component", result)
# k = cv2.waitKey(0)


#Saving images

# cv2.imwrite("test_data/test_file.png", disp_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

print(0)