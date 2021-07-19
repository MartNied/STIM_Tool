import cv2
import numpy as np



############ function implementations #######################
def contrast_cut(im, clip_val, dataType="uint8"):
    """Clipping contrast values at certain threshold value
        im - input image
        clip_val - above this value the intensity will be clipped
        dataType - specifing input image Datatype
    """
    assert isinstance(clip_val, int)  # Check if clipval is int

    if dataType == "uint8":        # 8 bit Case

        dyn_range = 2**8

        fp = [0, dyn_range-1]  # Input image Range
        # Input image will be mapped on full dynamic range withing this interval.
        xp = [0, clip_val]

        x = np.arange(dyn_range)  # create axis for LUT
        table = np.interp(x, xp, fp).astype(dataType)  # Create LUT
        im_clipped = cv2.LUT(im, table)  # Transform image

    elif dataType == "uint16":      # 16 bit Case

        dyn_range = 2**16
        im[im > clip_val] = clip_val

        im_clipped = cv2.normalize(
            im, None, alpha=0, beta=dyn_range-1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)

    return im_clipped


def cc_filter_idx(output_cc, min_area=5, max_area=520*696, min_length=0.0, max_length=1000, min_eccentricity=0.0, max_eccentricity=1.0, min_solidity=0.0, max_solidity=1.0, min_extent=0.0, max_extent=1.0, filter_area=True, filter_length=False, filter_eccentricity=False, filter_solidity=False, filter_extent=False):

    n_labels = output_cc[0]  # conneceted components analysis output
    labels = output_cc[1]
    stats = output_cc[2]

    filt_idx = np.arange(0, n_labels)  # create filt idx array
    # apply filters

    if filter_area:
        filt_idx = cc_filter_area(
            stats, filt_idx, min_area=min_area, max_area=max_area)

    if filter_length:
        filt_idx = cc_filter_length(
            labels, filt_idx, min_length=min_length, max_length=max_length)

    if filter_eccentricity:
        filt_idx = cc_filter_eccentricity(
            labels, filt_idx, min_eccentricity=min_eccentricity, max_eccentricity=max_eccentricity)

    if filter_solidity:
        filt_idx = cc_filter_solidity(
            labels, filt_idx, min_solidity=min_solidity, max_solidity=max_solidity)

    if filter_extent:
        filt_idx = cc_filter_extent(
            labels, filt_idx, min_extent=min_extent, max_extent=max_extent)

    # remove background
    if (filt_idx.size != 0) and (filt_idx[0] == 0):
        filt_idx = filt_idx[1:filt_idx.size]

    return filt_idx

############ filter funtions  ############################


def cc_filter_area(stats, filt_idx, min_area=5, max_area=520*696):

    filt_idx_area = np.where(np.logical_and(stats[:, cv2.CC_STAT_AREA] >= min_area, stats[:, cv2.CC_STAT_AREA] <= max_area))[
        0]  # calculate index where CC-Area lies in the intervall [min_area, max_area] --> idx is label

    filt_idx = apply_filt_idx_stat(filt_idx, filt_idx_area)

    return filt_idx


def cc_filter_length(labels, filt_idx, min_length=0, max_length=1000):

    length = np.zeros(filt_idx.size)  # array for storing length

    count = 0  # counter for indexing

    for i in filt_idx:  # iterate over filt idx element!

        # create uint8 binary image mask for connected component with label (index) i
        struct_mask = ((labels == i)*255).astype("uint8")

        # calculate outer contours of struct mask
        contour, _ = cv2.findContours(
            struct_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # calculate contour and calculate length

        rect = cv2.minAreaRect(contour[0])
        # biggest side of the minimum enclosing rectangle
        max_side = np.max(rect[1])
        length[count] = max_side
        count += 1

    # calculate indeces of filt_idx array!, where CC-solidity lies in the intervall [min_sol, max_sol]
    filt_idx_length = np.where(np.logical_and(
        length >= min_length, length <= max_length))[0]

    # Store the labels (indices) where solidity conditions are fullfilled !!
    filt_idx = apply_filt_idx_stat(filt_idx, filt_idx_length)

    return filt_idx


def cc_filter_eccentricity(labels, filt_idx, min_eccentricity=0, max_eccentricity=1):

    # array for storing eccentricity for each struct mask (see later)
    eccentricity = np.zeros(filt_idx.size)

    count = 0  # counter for indexing the eccentricity array

    # iterate only over CC where Area is in [min_area, max_area]
    for i in filt_idx:

        # create uint8 binary image mask for connected component with label (index) i
        struct_mask = ((labels == i)*255).astype("uint8")

        # calculate outer contours of struct mask
        contour, _ = cv2.findContours(
            struct_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Fit contour and calculate eccentricity

        if contour[0].shape[0] > 5:

            # Danger 5 points are requiered for fitting an ellipse !!
            ellipse = cv2.fitEllipse(contour[0])
            # store big and small (a,b) semiaxis of the ellipse
            a = np.max(ellipse[1])
            b = np.min(ellipse[1])

        # Alternate form using rotated minimum enclosing rectangle
        else:
            rect = cv2.minAreaRect(contour[0])
            # store big and small (a,b) side of the minimum enclosing rectangle
            a = np.max(rect[1])
            b = np.min(rect[1])

        # if avoid division by zero in eccentriciy formula (see else statement)
        if a == 0:
            eccentricity[count] = 0.0
            count += 1
        else:
            eccentricity[count] = np.sqrt(1-(b**2 / a**2))
            count += 1

    # calculate indeces of filt_idx array!, where CC-Eccentricity lies in the intervall [min_eccentricity, max_eccentricity]
    filt_idx_eccent = np.where(np.logical_and(
        eccentricity >= min_eccentricity, eccentricity <= max_eccentricity))[0]

    # Store the labels (indices) where Eccentricity conditions are fullfilled !!
    filt_idx = apply_filt_idx_stat(filt_idx, filt_idx_eccent)

    return filt_idx


def cc_filter_solidity(labels, filt_idx, min_solidity=0, max_solidity=1):

    solidity = np.zeros(filt_idx.size)  # array for storing solidity

    count = 0  # counter for indexing

    for i in filt_idx:  # iterate over filt idx element!

        # create uint8 binary image mask for connected component with label (index) i
        struct_mask = ((labels == i)*255).astype("uint8")

        # calculate outer contours of struct mask
        contour, _ = cv2.findContours(
            struct_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Fit contour and calculate eccentricity
        area = cv2.contourArea(contour[0])
        hull = cv2.convexHull(contour[0])
        hull_area = cv2.contourArea(hull)

        if np.logical_or(area == 0, hull_area == 0):  # avoid division by zero error
            solidity[count] = 1.0
        else:
            solidity[count] = float(area)/hull_area

        count += 1

    # calculate indeces of filt_idx array!, where CC-solidity lies in the intervall [min_sol, max_sol]
    filt_idx_solidity = np.where(np.logical_and(
        solidity >= min_solidity, solidity <= max_solidity))[0]

    # Store the labels (indices) where solidity conditions are fullfilled !!
    filt_idx = apply_filt_idx_stat(filt_idx, filt_idx_solidity)

    return filt_idx


def cc_filter_extent(labels, filt_idx, min_extent=0, max_extent=1):

    extent = np.zeros(filt_idx.size)  # array for storing solidity

    count = 0  # counter for indexing

    for i in filt_idx:  # iterate over filt idx element!

        # create uint8 binary image mask for connected component with label (index) i
        struct_mask = ((labels == i)*255).astype("uint8")

        # calculate outer contours of struct mask
        contour, _ = cv2.findContours(
            struct_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # calculate contour and calculate extent
        area = cv2.contourArea(contour[0])

        rect = cv2.minAreaRect(contour[0])
        a = rect[1][0]  # store (a,b) side of the minimum enclosing rectangle
        b = rect[1][1]

        rect_area = a * b

        if np.logical_or(area == 0, rect_area == 0):  # avoid division by zero error
            extent[count] = 1.0
        else:
            extent[count] = float(area)/rect_area
        count += 1

    # calculate indeces of filt_idx array!, where CC-solidity lies in the intervall [min_sol, max_sol]
    filt_idx_extent = np.where(np.logical_and(
        extent >= min_extent, extent <= max_extent))[0]

    # Store the labels (indices) where solidity conditions are fullfilled !!
    filt_idx = apply_filt_idx_stat(filt_idx, filt_idx_extent)

    return filt_idx


##### Helper fuctions #####


def apply_filt_idx_stat(filt_idx, filt_idx_stat):

    # if index or filter array is empty return unfiltered array
    if np.logical_or(filt_idx.size == 0, filt_idx_stat.size == 0):

        return np.array([])

    else:
        return filt_idx[filt_idx_stat]  # else return filtered index array


def cc_measurement(output_cc):
    """Function for calculatiation a total measusurement of connected components output"""
    
    n_labels = output_cc[0]  # conneceted components number of labels
    labels = output_cc[1] #label matrix
    stats = output_cc[2]
    
    data_dict = list() #create list for storing the row entries of the measurement.csv file

    for i in range(1, n_labels):  # iterate over labels and ignoring 0 label (background)

        # create uint8 binary image mask for connected component with label (index) i
        struct_mask = ((labels == i)*255).astype("uint8")
        # calculate outer contours of struct mask
        contour, _ = cv2.findContours(
            struct_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        
        #############       calculate measureands       #######################
        #area
        area = stats[i, cv2.CC_STAT_AREA]
        #length
        rect = cv2.minAreaRect(contour[0]) # min enclosing rectangle
        max_side = np.max(rect[1]) # save max side
        length = max_side   #save as length

        #eccentricity
        if contour[0].shape[0] > 5:
            # Danger 5 points are required for fitting an ellipse !!
            ellipse = cv2.fitEllipse(contour[0])
            # store big and small (a,b) semiaxis of the ellipse
            a = np.max(ellipse[1])
            b = np.min(ellipse[1])

        # Alternate form using rotated minimum enclosing rectangle
        else:
            # store big and small (a,b) side of the minimum enclosing rectangle
            a = np.max(rect[1])
            b = np.min(rect[1])

        # if avoid division by zero in eccentriciy formula (see else statement)
        if a == 0:
            eccentricity = 0.0
        else:
            eccentricity = np.sqrt(1-(b**2 / a**2))

        
        #solidity
        hull = cv2.convexHull(contour[0])
        hull_area = cv2.contourArea(hull)

        if np.logical_or(area == 0, hull_area == 0):  # avoid division by zero error
            solidity = 1.0
        else:
            solidity = float(area)/hull_area

        
        #extent
        a = rect[1][0]  # store (a,b) side of the minimum enclosing rectangle
        b = rect[1][1]

        rect_area = a * b

        if np.logical_or(area == 0, rect_area == 0):  # avoid division by zero error
            extent = 1.0
        else:
            extent = float(area)/rect_area
        
        data_dict.append({"Label" : i, "Area" : area, "Length": length, "Eccentricity" : eccentricity, "Solidity" : solidity, "Extent" : extent})
        
    return data_dict, labels


###############                         Line Iterator           ######################


def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
    """
   # define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y+1, P1Y+dYa+1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X-1, P1X-dXa-1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X+1, P1X+dXa+1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y-1, P1Y-dYa-1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y+1, P1Y+dYa+1)
            itbuffer[:, 0] = (slope*(itbuffer[:, 1]-P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                itbuffer[:, 0] = np.arange(P1X-1, P1X-dXa-1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X+1, P1X+dXa+1)
            itbuffer[:, 1] = (slope*(itbuffer[:, 0]-P1X)).astype(np.int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) &
                        (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(
        np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer
