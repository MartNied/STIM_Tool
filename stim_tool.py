from PyQt5 import QtCore
from PyQt5 import QtGui, uic
from PyQt5.QtCore import QSize, pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QAction, QSizePolicy, QLabel, QScrollArea
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
import sys
import os
import cv2
import numpy as np
import functools
from PhotoViewer import PhotoViewer
from ImageFunctions import contrast_cut, cc_filter_idx


class LoadQt(QMainWindow):
    def __init__(self):
        super(LoadQt, self).__init__()
        uic.loadUi("stim_tool_main_window.ui", self)  # load .ui file created with QT designer
        
        # Create additional GUI parts
        self.viewerFeedback = PhotoViewer(self)     #constructor of photoviewer widget
        self.viewerProcess = PhotoViewer(self)                                              
                    
        self.placeholder1.hide()      #hide GraphicsView Place holders!
        self.placeholder2.hide()

        self.displayLayout.insertWidget(0, self.viewerFeedback) #place Feedback viewer on left position in imageDisplayLayout
        self.displayLayout.insertWidget(1, self.viewerProcess)  #place Feedback viewer on right position in imageDisplayLayout   
        
        self.viewerFeedback.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        self.viewerFeedback.setMinimumSize(160,90)

        self.viewerProcess.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        self.viewerProcess.setMinimumSize(160,90)
        
        self.displayLayout.update()
        
        # create and set default settings of the app
        self.colorspace = 0  # zero means grayscale: opencv --> imread() --> grayscaleflag = 0
        self.clipval = 10000  # intensity clipvalue for the import of 16bit tiff files --> they are clippend normalized and then converted to 8bit grayscale images 
        self.thNeighborhood = 31 # Gaussian Kernel Size for adaptive thresholding in masking process
        self.connectivity = 8   # Connected components connectivity
        self.alphaBlend = 0.3 # alpha value for blending mask over process images (see doVisualFeedback function)
        
        self.FiltMinArea=5  
        self.FiltMaxArea= 100      #361920 max resolution
        self.FiltMinEccentricity=0.0
        self.FiltMaxEccentricity=1.0
        self.FiltMinSolidity=0.6
        self.FiltMaxSolidity=1.0
        self.FiltMinExtent=0.2
        self.FiltMaxExtent=1.0
        self.FiltArea=True
        self.FiltEccentricity=False
        self.FiltSolidity=False
        self.FiltExtent=False
        
        # create some properties
        self.fileName = "" # filename of currenty loaded Image 
        self.fileExt = None  # File extension of currently loaded image
        self.filePath = None # File Path of currently loaded image
        self.activeTool = "drag"
        
        self.SessionFileNames = None   # Array with all filenames which are consideres for the current session
        self.SessionFilePaths = None
        
        self.pixmapFeedback = None # Qt Pixmap of current shown Image in Feedback window
        self.pixmapProcess = None     # Same for Mask image on the left side
        
        self.imageDataRaw = None  #Data of image file which is currently loaded (if tiff --> normalization and conversion too 8 bit grayscale)
        self.imageDataFeedback = None  #Data of image on the left positioned feedback viewer 
        self.imageDataProcess = None #Data of image on the right positioned processing viewer
        self.labelData = None  #Label Matrix for corresponding mask Data 
        self.maskData = None # Mask Data from Threhsolding and filtering
        
        self.nStructs = "--" # number of counted structs
        

        self.wireActions()  # connect actions, buttons etc. and set default values
        self.wireButtons()
        self.wireSliders()
        self.wireCheckBoxes()
        self.wireSpinBoxes()

        #Wire functions to widgets and set default values !!!

    def wireActions(self):
        self.openAction.triggered.connect(self.openImage)
        self.saveAction.triggered.connect(self.saveImage)
        self.exitAction.triggered.connect(self.close)

        self.dragAction.triggered.connect(functools.partial(self.toolSelector, "drag"))
        self.roiAction.triggered.connect(functools.partial(self.toolSelector, "roi"))
        self.cutAction.triggered.connect(functools.partial(self.toolSelector, "cut"))
        self.eraseAction.triggered.connect(functools.partial(self.toolSelector, "erase"))

    def wireButtons(self):
        self.maskButton.clicked.connect(self.doMasking)
        self.undoMaskButton.clicked.connect(self.undoMasking)

    def wireSliders(self):
        self.contrastTransformSlider_1.valueChanged.connect(self.contrastTransform)
        self.contrastTransformSlider_2.valueChanged.connect(self.contrastTransform)

    def wireCheckBoxes(self):
        #set default values
        self.areaFilterCheckBox.setChecked(self.FiltArea)
        self.eccentFilterCheckBox.setChecked(self.FiltEccentricity)
        self.solidityFilterCheckBox.setChecked(self.FiltSolidity)
        self.extentFilterCheckBox.setChecked(self.FiltExtent)
        
        
        #wiring checkboxes to function doCheckBox
        self.areaFilterCheckBox.stateChanged.connect(functools.partial(self.doCheckBox, self.areaFilterCheckBox, "FiltArea"))   #functools.partial allows do combine the callback doCheckbox with input arguments. Without this trick a QAction doens't allow a callback with input arguments because None is somhow returned!!
        self.eccentFilterCheckBox.stateChanged.connect(functools.partial(self.doCheckBox, self.eccentFilterCheckBox, "FiltEccentricity"))
        self.solidityFilterCheckBox.stateChanged.connect(functools.partial(self.doCheckBox, self.solidityFilterCheckBox, "FiltSolidity"))
        self.extentFilterCheckBox.stateChanged.connect(functools.partial(self.doCheckBox, self.extentFilterCheckBox, "FiltExtent"))

    def wireSpinBoxes(self):
        #set default values

        self.minAreaFilterSpinBox.setValue(self.FiltMinArea)
        self.minEccentFilterSpinBox.setValue(self.FiltMinEccentricity)
        self.minSolidityFilterSpinBox.setValue(self.FiltMinSolidity)
        self.minExtentFilterSpinBox.setValue(self.FiltMinExtent)
        self.maxAreaFilterSpinBox.setValue(self.FiltMaxArea)
        self.maxEccentFilterSpinBox.setValue(self.FiltMaxEccentricity)
        self.maxSolidityFilterSpinBox.setValue(self.FiltMaxSolidity)
        self.maxExtentFilterSpinBox.setValue(self.FiltMaxExtent)

        #Wiring Spin Boxes to function doSpinBox
        self.minAreaFilterSpinBox.valueChanged.connect(functools.partial(self.doSpinBox, self.minAreaFilterSpinBox, "FiltMinArea"))  
        self.minEccentFilterSpinBox.valueChanged.connect(functools.partial(self.doSpinBox, self.minEccentFilterSpinBox, "FiltMinEccentricity"))
        self.minSolidityFilterSpinBox.valueChanged.connect(functools.partial(self.doSpinBox, self.minSolidityFilterSpinBox, "FiltMinSolidity"))
        self.minExtentFilterSpinBox.valueChanged.connect(functools.partial(self.doSpinBox, self.minExtentFilterSpinBox, "FiltMinExtent"))
        self.maxAreaFilterSpinBox.valueChanged.connect(functools.partial(self.doSpinBox, self.maxAreaFilterSpinBox,"FiltMaxArea"))
        self.maxEccentFilterSpinBox.valueChanged.connect(functools.partial(self.doSpinBox, self.maxEccentFilterSpinBox, "FiltMaxEccentricity"))
        self.maxSolidityFilterSpinBox.valueChanged.connect(functools.partial(self.doSpinBox, self.maxSolidityFilterSpinBox, "FiltMaxSolidity"))
        self.maxExtentFilterSpinBox.valueChanged.connect(functools.partial(self.doSpinBox, self.maxExtentFilterSpinBox, "FiltMaxExtent"))
    
    ################################################  Create Methods #############################################################

    def doCheckBox(self, CBox, AttributeName):
        """Checking the state of a given Check Box and changing the corresponding Attribute (must be bool). Changes self.AttributeName to (True or False)"""
        if CBox.isChecked():
            setattr(self, AttributeName, True)
        else:
            setattr(self, AttributeName, False)

    def doSpinBox(self, SBox, AttributeName):
        """Checking state of given Spin Box and changing the corresponding Attribute (can be float)"""
        val = SBox.value()  #Get value of currently changed Spin box
        setattr(self, AttributeName, val) #set attribute to value of current spinBox


    def openImage(self):
        """Creates file dialog and sets the imported image as pixmap in in the image display label """
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "",  # open File Dialog
                                                  "Images (*.*)", options=options)

        if not filePath:  # if file dialog is canceled filename is bool-false
            return
        else:
            self.filePath = filePath  #set current file Path of image which is loaded
            self.fileExt = os.path.splitext(filePath)[1]        #save file extension from path
            
            if self.colorspace == 0:  # needs to be adapated if non grayscale images are imported (0 flag for Grayscale images)

                image_import = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)

                if self.fileExt == ".tif":      #if tif file is importet preprocess it
            
                    im_clip_norm = contrast_cut(image_import, self.clipval, dataType="uint16") #clip intensity values above certain value
                    self.imageDataRaw = cv2.convertScaleAbs(im_clip_norm, alpha=(2**8 / 2**16))    #conver to 8 bit grayscale image for processing

                else:
                    self.imageDataRaw = image_import 
            
            # if a no image file is loades a message box will pop up
            if np.all(self.imageDataRaw == None):
                QMessageBox.information(
                    self, "Error Loading Image", "Cannot load %s." % filePath)
                return
            
            #set Data to imported Image
            self.imageDataProcess = np.copy(self.imageDataRaw)
            self.imageDataFeedback = np.copy(self.imageDataRaw)

            ##set slider and buttons enables status
            
            self.contrastTransformSlider_1.setValue(255)
            self.contrastTransformSlider_2.setValue(0)

            self.contrastTransformSlider_1.setEnabled(True)
            self.contrastTransformSlider_2.setEnabled(True)
            self.doMeasurementButton.setEnabled(False)

            self.fileName = os.path.basename(filePath) #create filename from path
            self.displayImage(self.imageDataRaw, display="both") #display raw data on both viewers
            self.setBottomLabel() #set bottom label txt

    def displayImage(self, ImageData, display, fitView=True, colorspace="grayscale"):
        """ Converts openCv imported Data to QImage and sets it as pixmap in the image Label """
        if colorspace == "grayscale":  # needs to be adapated if non grayscale images are imported
            qformat = QImage.Format_Grayscale8
            height, width = ImageData.shape[:2]
            nbytes = ImageData.nbytes  # count total bytes
            # calculate bytes per Line for correct conversion process
            bytesPerLine = int(nbytes/height)

            image = QImage(ImageData.data, width,
                           height, bytesPerLine, qformat)

        elif colorspace == "rgb":
            qformat = QImage.Format_RGB888
            height, width = ImageData.shape[:2]
            bytesPerLine = 3 * width
            
            image = QImage(ImageData.data, width, height, bytesPerLine, qformat)


        #create pixmap and set photo on certain viewer specified by the value of display
        if display == "feedback":
            self.pixmapFeedback = QPixmap.fromImage(image)
            self.viewerFeedback.setPhoto(self.pixmapFeedback, fitView=fitView)
        
        elif display == "process":
            self.pixmapProcess = QPixmap.fromImage(image)
            self.viewerProcess.setPhoto(self.pixmapProcess, fitView=fitView)

        elif display == "both":
            self.pixmapFeedback = QPixmap.fromImage(image)
            self.viewerFeedback.setPhoto(self.pixmapFeedback, fitView=fitView)

            self.pixmapProcess = QPixmap.fromImage(image)
            self.viewerProcess.setPhoto(self.pixmapProcess, fitView=fitView)
        

    def saveImage(self):
        """ Opens File Dialog and writes the current image Data to an uncompressed png-File """
        if np.all(self.imageData == None):  # basically same procedure as openImage
            QMessageBox.information(
                self, "Error Saving Image", "No image was loaded!")
        else:
            filePath, _ = QFileDialog.getSaveFileName(
                self, "Save Image", "", "Image Files (*.png)")
            if not filePath:
                return
            else:
                try:
                    cv2.imwrite(filePath, self.imageData, [  # write imageData to uncompressed .png File
                        cv2.IMWRITE_PNG_COMPRESSION, 0])
                except:
                    QMessageBox.information(
                        self, "Error Saving Image", "Cannot save %s." % filePath)
                    return

    def contrastTransform(self):
        if np.all(self.imageDataProcess == None):  # check if image was loaded
            QMessageBox.information(
                self, "Error processing Image", "No image was loaded!")
                
            return

        upper_contrast_val = self.contrastTransformSlider_1.value()
        lower_contrast_val = self.contrastTransformSlider_2.value()

        cv2.intensity_transform.contrastStretching(self.imageDataRaw, self.imageDataProcess, lower_contrast_val, 0, upper_contrast_val, 255)
        
        self.displayImage(self.imageDataProcess, display="both", fitView=False)
        self.setBottomLabel()
        

    def doMasking(self):
        """creates mask from image Data process (contrast transformed image). First an adapative threshold method is used and afterwards the mask is filtered with by the function cc_filter_idx. From the filtered labels a filtered mask is calculated and displayed"""
        if np.all(self.imageDataProcess == None):  # check if image was loaded
            QMessageBox.information(
                self, "Error processing Image", "No image was loaded!")
            
            return

        self.maskData = cv2.adaptiveThreshold(self.imageDataProcess, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.thNeighborhood, 0) #set process Data to raw mask
        
        output_cc = cv2.connectedComponentsWithStats(self.maskData, connectivity=self.connectivity) # calcualte connected components of thresholded image (needed for cc_filter idx.)
        
        self.labelData = output_cc[1]  #set label data to unfiltred label Matrix
        self.nStructs = output_cc[0] - 1 #set number of structs to numer of labels - 1 (0 is Background label)
        
        if np.any([self.FiltArea, self.FiltEccentricity, self.FiltSolidity, self.FiltExtent]):      # if any Filter is enabled apply filter functions
            
            filt_idx = cc_filter_idx(output_cc, min_area=self.FiltMinArea, max_area=self.FiltMaxArea, min_eccentricity=self.FiltMinEccentricity, max_eccentricity=self.FiltMaxEccentricity, min_solidity=self.FiltMinSolidity, max_solidity=self.FiltMaxSolidity, min_extent=self.FiltMinExtent, max_extent=self.FiltMaxExtent, filter_area=self.FiltArea, filter_eccentricity=self.FiltEccentricity, filter_solidity=self.FiltSolidity, filter_extent=self.FiltExtent)  #calculate labels where Filter conditions are fullfilled!
            

            filt_labels = np.zeros_like(self.labelData)  #created empty filtered label array
            
            for i in filt_idx:  #iterate over all filter indices
                filt_labels[self.labelData == i] = i  #fill empty label array with filtered labels (0 is Background label)

            filt_mask = ((filt_labels != 0)*255).astype("uint8") #calculate binary mask from filtered labels
        
            self.maskData = filt_mask #set image data to filtered Mask
            self.labelData = filt_labels    #set label Data to filtered label Matrix
        
            self.nStructs = filt_idx.size  #set struct count to size of filter idx array

        self.displayImage(self.maskData, display="process", fitView=False) #display image on process side
        self.doVisualFeedback()
        self.setBottomLabel()
    
    def doVisualFeedback(self):
        """creates overlay of imageDataProcess (contrast transformed image) and mask. The two images are blended together and displayed on the feedback viewer"""
        
        raw3chan = cv2.cvtColor(self.imageDataProcess, cv2.COLOR_GRAY2RGB)
        
        colorMask = np.zeros_like(raw3chan)
        colorMask[:,:,0] = self.maskData

        self.imageDataFeedback = cv2.addWeighted(raw3chan,(1-self.alphaBlend), colorMask, self.alphaBlend, 0)

        self.displayImage(self.imageDataFeedback, display="feedback", fitView=False, colorspace="rgb")

    def undoMasking(self):
        """reseting the overlay of mask and image Data to normal View"""

        if np.all(self.imageDataProcess == None):  # check if image was loaded
            QMessageBox.information(
                self, "Error processing Image", "No image was loaded!")
            
            return

        self.displayImage(self.imageDataProcess, display="both", fitView=False)
        self.nStructs = "--" #set bottom label struct count
        self.setBottomLabel() 
    
    
    def setBottomLabel(self):
        "creates a string which is displayed in the bottom laybel for user information"
        dispStr = " File: " + self.fileName + "       Lower Th: " + str(self.contrastTransformSlider_2.value()) + "       Upper Th: " + str(self.contrastTransformSlider_1.value()) + "        Struct count: " + str(self.nStructs)
        self.bottomLabel.setText(dispStr)

    def toolSelector(self, selected_tool="drag"):
        
        self.activeTool = selected_tool
        self.viewerFeedback._tool=selected_tool
        self.viewerProcess._tool=selected_tool

        if selected_tool == "drag":
            self.dragAction.setChecked(True)
            self.roiAction.setChecked(False)
            self.cutAction.setChecked(False)
            self.eraseAction.setChecked(False)
        
        elif selected_tool == "roi":
            self.dragAction.setChecked(False)
            self.roiAction.setChecked(True)
            self.cutAction.setChecked(False)
            self.eraseAction.setChecked(False)

        elif selected_tool == "cut":
            self.dragAction.setChecked(False)
            self.roiAction.setChecked(False)
            self.cutAction.setChecked(True)
            self.eraseAction.setChecked(False)

        elif selected_tool == "erase":
            self.dragAction.setChecked(False)
            self.roiAction.setChecked(False)
            self.cutAction.setChecked(False)
            self.eraseAction.setChecked(True)


#Launch app
app = QApplication(sys.argv)
win = LoadQt()
win.show()
sys.exit(app.exec())
