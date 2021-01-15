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
import matplotlib.pyplot as plt
from PhotoViewer import PhotoViewer


class LoadQt(QMainWindow):
    def __init__(self):
        super(LoadQt, self).__init__()
        uic.loadUi("stim_tool_main_window.ui", self)  # load .ui file created with QT designer
        # Create additional GUI parts
        self.viewer = PhotoViewer(self)  # from importet PhotoViewerClass
        self.mainVerticalLayout.insertWidget(0, self.viewer)

        # set some properties
        self.colorspace = 0  # zero means grayscale: opencv --> imread() --> grayscaleflag = 0
        self.fileName = None # filename of openend Image
        self.pixmap = None # Qt Pixmap of current shown Image
        self.imageData = None  # openCv importet numpy array of raw Image
        self.dispData = None # array of current shown image
        self.processData = None # contrast stretched Data
        self.structMask = None # thresholded Data
        self.structMaskFilt = None # morph. openend Data
        self.nStructs = "--" # number of counted struchts
        

        self.wireActions()  # connect actions, buttons etc.
        self.wireButtons()
        self.wireSliders()

        # Wire functions to widgets

    def wireActions(self):
        self.openAction.triggered.connect(self.openImage)
        self.saveAction.triggered.connect(self.saveImage)
        self.exitAction.triggered.connect(self.close)

    def wireButtons(self):
        self.saveButton.clicked.connect(self.saveImage)
        self.countButton.clicked.connect(self.structCount)

    def wireSliders(self):
        self.par1Slider.valueChanged.connect(self.contrastStretch)
        self.par2Slider.valueChanged.connect(self.contrastStretch)

        # Create Methods

    def openImage(self):
        """Creates file dialog and sets the imported image as pixmap in in the image display label """
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "",  # open File Dialog
                                                  "Images (*.*)", options=options)

        if not filePath:  # if file dialog is canceled filename is bool-false
            return
        else:
            if self.colorspace == 0:  # needs to be adapated if non grayscale images are imported
                # uint8 array type for opencv processing
                self.imageData = np.array([1], ndmin=2, dtype=np.uint8)
                self.imageData = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
                self.processData = np.array(self.imageData)
            # if a no image file is loades a message box will pop up
            if np.all(self.imageData == None):
                QMessageBox.information(
                    self, "Error Loading Image", "Cannot load %s." % filePath)
                return
            self.fileName = os.path.basename(filePath)
            self.displayImage(self.imageData)
            self.dispData = self.imageData
            self.bottomLabel.setText(
                "File: " + self.fileName + "       Lower Th: -- " + "   Upper Th: --" + "        Struct count: --")
            self.par1Slider.setValue(0)
            self.par2Slider.setValue(255)

    def displayImage(self, cvData):
        """ Converts openCv imported Data to QImage and sets it as pixmap in the image Label """
        if self.colorspace == 0:  # needs to be adapated if non grayscale images are imported
            qformat = QImage.Format_Grayscale8
            height, width = cvData.shape[:2]
            nbytes = cvData.nbytes  # count total bytes
            # calculate bytes per Line for correct conversion process
            bytesPerLine = int(nbytes/height)

            image = QImage(cvData.data, width,
                           height, bytesPerLine, qformat)

        self.pixmap = QPixmap.fromImage(image)
        self.viewer.setPhoto(self.pixmap)

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
                    cv2.imwrite(filePath, self.dispData, [  # write imageData to uncompressed .png File
                        cv2.IMWRITE_PNG_COMPRESSION, 0])
                except:
                    QMessageBox.information(
                        self, "Error Saving Image", "Cannot save %s." % filePath)
                    return

    def contrastStretch(self):
        if np.all(self.imageData == None):  # check if image was loaded
            QMessageBox.information(
                self, "Error processing Image", "No image was loaded!")
            return

        par1 = self.par1Slider.value()
        par2 = self.par2Slider.value()
        cv2.intensity_transform.contrastStretching(
            self.imageData, self.processData, par1, 0, par2, 255)
        self.displayImage(self.processData)
        self.dispData = self.processData
        self.bottomLabel.setText("File: " + self.fileName + "       Lower Th: " + str(
            self.par1Slider.value()) + "    Upper Th: " + str(self.par2Slider.value()) + "        Struct count: " + str(self.nStructs))
        

    def structTh(self):
        self.contrastStretch()
        nbh = 31
        self.structMask = cv2.adaptiveThreshold(
            self.processData, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, nbh, 0)
        
        
    def morphOpening(self):
        self.structTh()
        sizeKernel = self.par3Slider.value()
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (sizeKernel, sizeKernel))
        self.structMaskFilt = cv2.morphologyEx(
            self.structMask, cv2.MORPH_OPEN, kernel, iterations=1)
        

    def structCount(self):
        if np.all(self.imageData == None):  # check if image was loaded
            QMessageBox.information(
                self, "Error processing Image", "No image was loaded!")
            return
        self.morphOpening()
        nLabels, _ = cv2.connectedComponents(self.structMaskFilt, labels=None, connectivity=8)
        self.nStructs = nLabels - 1
        self.bottomLabel.setText("File: " + self.fileName + "       Lower Th: " + str(self.par1Slider.value()) + "    Upper Th: " + str(self.par2Slider.value()) + "        Struct count: " + str(self.nStructs))
        self.displayImage(self.structMaskFilt)
        self.dispData = self.structMaskFilt

#Launch app
app = QApplication(sys.argv)
win = LoadQt()
win.show()
sys.exit(app.exec())
