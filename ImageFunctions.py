import cv2
import numpy as np
from pathlib import Path

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



def cc_filter_idx(output_cc, min_area=5, max_area=520*696, min_eccentricity=0.0, max_eccentricity=1.0, min_solidity=0.0, max_solidity=1.0, min_extent=0.0, max_extent=1.0, filter_area=True, filter_eccentricity=False, filter_solidity=False, filter_extent=False):
    
    n_labels = output_cc[0]  #conneceted components analysis output
    labels = output_cc[1]
    stats = output_cc[2]
    
    filt_idx = np.arange(0, n_labels) #create filt idx array

    ##apply filters

    if filter_area:
        filt_idx = cc_filter_area(stats, filt_idx, min_area=min_area, max_area=max_area)

    if filter_eccentricity:
        filt_idx = cc_filter_eccentricity(labels, filt_idx, min_eccentricity=min_eccentricity, max_eccentricity=max_eccentricity)
    
    if filter_solidity:
        filt_idx = cc_filter_solidity(labels, filt_idx, min_solidity=min_solidity, max_solidity=max_solidity)
    
    if filter_extent:
        filt_idx = cc_filter_extent(labels, filt_idx, min_extent=min_extent, max_extent=max_extent)
    
    ##remove background

    if filt_idx[0] == 0:
        filt_idx = filt_idx[1:filt_idx.size]
    
    return filt_idx

############ filter funtions  ############################

def cc_filter_area(stats, filt_idx, min_area=5, max_area=520*696):

    filt_idx_area = np.where(np.logical_and(stats[:, cv2.CC_STAT_AREA] >= min_area, stats[:, cv2.CC_STAT_AREA] <= max_area))[0]  #calculate index where CC-Area lies in the intervall [min_area, max_area] --> idx is label
    
    filt_idx = apply_filt_idx_stat(filt_idx, filt_idx_area)
    
    return filt_idx


def cc_filter_eccentricity(labels, filt_idx, min_eccentricity=0, max_eccentricity=1):

    eccentricity = np.zeros(filt_idx.size)  #array for storing eccentricity for each struct mask (see later)
    
    count = 0               #counter for indexing the eccentricity array
    
    for i in filt_idx:      #iterate only over CC where Area is in [min_area, max_area]
        
        struct_mask = ((labels == i)*255).astype("uint8")   #create uint8 binary image mask for connected component with label (index) i        
        
        contour,_ = cv2.findContours(struct_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)     #calculate outer contours of struct mask
        
        ## Fit contour and calculate eccentricity

        ellipse = cv2.fitEllipse(contour[0])     ## Danger 5 points are requiered for fitting an ellipse !!                                                   
        a = np.max(ellipse[1])      ### store big and small (a,b) semiaxis of the ellipse
        b = np.min(ellipse[1])

        ### Alternate form using rotated rectangle
        # rect = cv2.minAreaRect(contour[0])
        # a = np.max(rect[1])      ### store big and small (a,b) side of the rectangle
        # b = np.min(rect[1])
        
        assert((a >= 0) and (b >= 0))
        
        eccentricity[count] = np.sqrt(1-(b**2 / a**2))
        count += 1

        
    
    filt_idx_eccent = np.where(np.logical_and(eccentricity >= min_eccentricity, eccentricity <= max_eccentricity))[0]      #calculate indeces of filt_idx array!, where CC-Eccentricity lies in the intervall [min_eccentricity, max_eccentricity]
    
    filt_idx = apply_filt_idx_stat(filt_idx, filt_idx_eccent)  # Store the labels (indices) where Eccentricity conditions are fullfilled !!
    
    return filt_idx


def cc_filter_solidity(labels, filt_idx, min_solidity=0, max_solidity=1):
    
    solidity = np.zeros(filt_idx.size)  #array for storing solidity
    
    count = 0               #counter for indexing
    
    for i in filt_idx:      #iterate over filt idx element!
        
        struct_mask = ((labels == i)*255).astype("uint8")   #create uint8 binary image mask for connected component with label (index) i        
        
        contour,_ = cv2.findContours(struct_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)     #calculate outer contours of struct mask
        
        ## Fit contour and calculate eccentricity
        area = cv2.contourArea(contour[0])
        hull = cv2.convexHull(contour[0])
        hull_area = cv2.contourArea(hull)

        if np.logical_or(area == 0, hull_area == 0): #avoid division by zero error
            solidity[count] = 1.0
        else:
            solidity[count] = float(area)/hull_area
        
        count += 1

   
    filt_idx_solidity = np.where(np.logical_and(solidity >= min_solidity, solidity <= max_solidity))[0]      #calculate indeces of filt_idx array!, where CC-solidity lies in the intervall [min_sol, max_sol]
    
    filt_idx = apply_filt_idx_stat(filt_idx, filt_idx_solidity)  # Store the labels (indices) where solidity conditions are fullfilled !!
    
    return filt_idx
    
def cc_filter_extent(labels, filt_idx, min_extent=0, max_extent=1):
    
    extent = np.zeros(filt_idx.size)  #array for storing solidity
    
    count = 0               #counter for indexing
    
    for i in filt_idx:      #iterate over filt idx element!
        
        struct_mask = ((labels == i)*255).astype("uint8")   #create uint8 binary image mask for connected component with label (index) i        
        
        contour,_ = cv2.findContours(struct_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)     #calculate outer contours of struct mask
        
        ## calculate contour and calculate extent
        area = cv2.contourArea(contour[0])
        _,_,w,h = cv2.boundingRect(contour[0])
        rect_area = w*h

        if np.logical_or(area == 0, rect_area == 0): #avoid division by zero error
            extent[count] = 1.0
        else:
            extent[count] = float(area)/rect_area
        count += 1

   
    filt_idx_extent = np.where(np.logical_and(extent >= min_extent, extent <= max_extent))[0]      #calculate indeces of filt_idx array!, where CC-solidity lies in the intervall [min_sol, max_sol]
    
    filt_idx = apply_filt_idx_stat(filt_idx, filt_idx_extent)  # Store the labels (indices) where solidity conditions are fullfilled !!
    
    return filt_idx


##### Helper fuctions #####


def apply_filt_idx_stat(filt_idx, filt_idx_stat):

    if np.logical_or(filt_idx.size == 0, filt_idx_stat.size == 0): ## if index or filter array is empty return unfiltered array
        
        return filt_idx
     
    else: 
        return filt_idx[filt_idx_stat]  #else return filtered index array

