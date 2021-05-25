import cv2
import numpy as np
from pathlib import Path

from ImageFunctions import contrast_cut, cc_filter_idx


#####################################       Main code       ###########################

#Import

root = Path(".")
filePath = root.absolute() / "test_data" / "aktiviert8.tif" 
filePath = str(filePath)
im = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)

#Preprocessing

# clip_val = int(im.max())
clip_val = 5000
im_clip_norm = contrast_cut(im, clip_val, dataType="uint16")
disp_image = cv2.convertScaleAbs(im_clip_norm, alpha=(2**8 / 2**16))

disp_image_cut = disp_image[:,100:300]
# #Processing

# _, im_th = cv2.threshold(disp_image, 140, 255, cv2.THRESH_BINARY, None)
# output = cv2.connectedComponentsWithStats(im_th, connectivity=4)

# filt_idx = cc_filter_idx(output, min_area=5, max_area=np.prod(im.shape), min_eccentricity=0.0, max_eccentricity=1.0, min_solidity=0.6, max_solidity=1.0, min_extent=0.2, max_extent=1.0, filter_area=True, filter_eccentricity=False, filter_solidity=True, filter_extent=True)

# ### set labels which are not contained in filt_idx to zero

# labels = output[1]
# stats = output[2]

# # labels_filt = np.copy(labels)

# labels_filt = np.zeros_like(labels)

# for i in filt_idx:
#    labels_filt[labels == i] = 1     #set everything to backgroud which is not containted in filt_idx 


# #display results

# mask = ((labels != 0)*255).astype("uint8")
# mask_filt = ((labels_filt != 0)*255).astype("uint8")

cv2.imshow("Raw Image", disp_image)
cv2.imshow("Clipped image", disp_image_cut)


k = cv2.waitKey(0)

#cv2.imwrite("test_data/test_file_8bit.png", disp_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
#cv2.imwrite("test_data/test_file_16bit.png", im_clip_norm, [cv2.IMWRITE_PNG_COMPRESSION, 0])

print(0)



### calc and fit contours
# struct_bin = ((labels == 168)*255).astype("uint8") ##168 was a big one
# contours,_ = cv2.findContours(struct_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnt = contours[0]

#Copy with RGB channel

# result = cv2.cvtColor(struct_bin, cv2.COLOR_GRAY2RGB)

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

# k = cv2.waitKey(0)


#Saving images

# cv2.imwrite("test_data/test_file.png", disp_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

