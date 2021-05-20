from PyQt5 import QtCore
from PyQt5 import QtGui, uic
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QAction, QSizePolicy, QLabel, QScrollArea
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
import sys
import os
import cv2
import numpy as np
from PhotoViewer import PhotoViewer
from ImageFunctions import contrast_cut, cc_filter_idx


class LoadQt(QMainWindow):
    def __init__(self):
        super(LoadQt, self).__init__()
        uic.loadUi("stim_tool_main_window.ui", self)  # load .ui file created with QT designer
        
        # Create additional GUI parts
        self.viewerFeedback = PhotoViewer(self)     #constructor of photoviewer widget
        self.viewerProcess = PhotoViewer(self)                                              
                    
        self.imageDisplayLayout.insertWidget(0, self.viewerFeedback) #place Feedback viewer on left position in imageDisplayLayout
        self.imageDisplayLayout.insertWidget(1, self.viewerProcess)  #place Feedback viewer on right position in imageDisplayLayout   
        
        
        # create and set default settings of the app
        self.colorspace = 0  # zero means grayscale: opencv --> imread() --> grayscaleflag = 0
        self.clipval = 10000  # intensity clipvalue for the import of 16bit tiff files --> they are clippend normalized and then converted to 8bit grayscale images 

        self.FiltMinArea=0
        self.FiltMaxArea=361920
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
        self.fileName = None # filename of currenty loaded Image 
        self.fileExt = None  # File extension of currently loaded image
        self.filePath = None # File Path of currently loaded image
        
        self.SessionFileNames = None   # Array with all filenames which are consideres for the current session
        self.SessionFilePaths = None
        
        self.pixmapFeedback = None # Qt Pixmap of current shown Image in Feedback window
        self.pixmapProcess = None     # Same for Mask image on the left side
        
        self.imageDataRaw = None  #Data of image file which is currently loaded (if tiff --> normalization and conversion too 8 bit grayscale)
        self.imageDataFeedback = None  #Data of image on the left positioned feedback viewer 
        self.imageDataProcess = None #Data of image on the right positioned processing viewer
        
        self.nStructs = "--" # number of counted structs
        

        self.wireActions()  # connect actions, buttons etc. and set default values
        self.wireButtons()
        self.wireSliders()

        #Wire functions to widgets and set default values

    def wireActions(self):
        self.openAction.triggered.connect(self.openImage)
        self.saveAction.triggered.connect(self.saveImage)
        self.exitAction.triggered.connect(self.close)

    def wireButtons(self):
        self.maskButton.clicked.connect(self.openImage)

    def wireSliders(self):
        self.contrastTransformSlider.valueChanged.connect(self.contrastTransform)

    def wireCheckBoxes(self):
        pass

    def wireSpinBoxes(self):
        pass

    
    ##################  Create Methods ########################

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

            self.contrastTransformSlider.setValue(0)

            self.fileName = os.path.basename(filePath) #create filename from path
            self.displayImage(self.imageDataRaw, display="both") #display raw data on both viewers
            self.bottomLabel.setText(
                "File: " + self.fileName + "       Lower Intensity Bound: -- " + "        Struct count: --")    #set text in bottom info label
            

    def displayImage(self, ImageData, display, fitView=True):
        """ Converts openCv imported Data to QImage and sets it as pixmap in the image Label """
        if self.colorspace == 0:  # needs to be adapated if non grayscale images are imported
            qformat = QImage.Format_Grayscale8
            height, width = ImageData.shape[:2]
            nbytes = ImageData.nbytes  # count total bytes
            # calculate bytes per Line for correct conversion process
            bytesPerLine = int(nbytes/height)

            image = QImage(ImageData.data, width,
                           height, bytesPerLine, qformat)

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

        lower_contrast_val = self.contrastTransformSlider.value()
        
        cv2.intensity_transform.contrastStretching(self.imageDataRaw, self.imageDataProcess, lower_contrast_val, 0, 255, 255)
        
        self.displayImage(self.imageDataProcess, display="process", fitView=False)
        
        self.bottomLabel.setText("File: " + self.fileName + "       Lower Intensity Bound: " + str(
            self.contrastTransformSlider.value()) + "        Struct count: " + str(self.nStructs))
        

    # def structTh(self):
    #     self.contrastStretch()
    #     nbh = 31
    #     self.structMask = cv2.adaptiveThreshold(
    #         self.processData, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, nbh, 0)
        


    # def structCount(self):
    #     if np.all(self.imageData == None):  # check if image was loaded
    #         QMessageBox.information(
    #             self, "Error processing Image", "No image was loaded!")
    #         return
    #     self.morphOpening()
    #     nLabels, _ = cv2.connectedComponents(self.structMaskFilt, labels=None, connectivity=8)
    #     self.nStructs = nLabels - 1
    #     self.bottomLabel.setText("File: " + self.fileName + "       Lower Th: " + str(self.par1Slider.value()) + "    Upper Th: " + str(self.par2Slider.value()) + "        Struct count: " + str(self.nStructs))
    #     self.displayImage(self.structMaskFilt)
    #     self.dispData = self.structMaskFilt

#Launch app
app = QApplication(sys.argv)
win = LoadQt()
win.show()
sys.exit(app.exec())
