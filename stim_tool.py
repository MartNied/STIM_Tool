from PyQt5 import QtCore
from PyQt5 import QtGui, uic
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, QSize, pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QTextCursor
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QSizePolicy, QLabel, QScrollArea, QAction
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
import sys
import os
import cv2
import numpy as np
import functools
import csv
from PhotoViewer import PhotoViewer
from ImageFunctions import contrast_cut, cc_filter_idx, cc_measurement


class LoadQt(QMainWindow):
    def __init__(self):
        super(LoadQt, self).__init__()
        # load .ui file created with QT designer
        uic.loadUi("stim_tool_main_window.ui", self)

        # Create additional GUI parts
        # constructor of photoviewer widget
        self.viewerFeedback = PhotoViewer(self)
        self.viewerProcess = PhotoViewer(self)

        self.placeholder1.hide()  # hide GraphicsView Place holders!
        self.placeholder2.hide()

        # place Feedback viewer on left position in imageDisplayLayout
        self.displayLayout.insertWidget(0, self.viewerFeedback)
        # place Feedback viewer on right position in imageDisplayLayout
        self.displayLayout.insertWidget(1, self.viewerProcess)

        self.viewerFeedback.setSizePolicy(QSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        self.viewerFeedback.setMinimumSize(160, 90)

        self.viewerProcess.setSizePolicy(QSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        self.viewerProcess.setMinimumSize(160, 90)

        self.displayLayout.update()


        # create and set default settings of the app
        self.colorspace = 0  # zero means grayscale: opencv --> imread() --> grayscaleflag = 0
        self.clipval = 2**16  # intensity clipvalue for the import of 16bit tiff files --> they are clippend normalized and then converted to 8bit grayscale images
        # Gaussian Kernel Size for adaptive thresholding in masking process
        self.thNeighborhood = 31
        self.connectivity = 4   # Connected components connectivity
        self.maxUndos = 5       # max list length for storing old mask states

        # alpha value for blending mask over process images (see doVisualFeedback function)
        self.alphaBlend = 0.3

        # filter default settings
        self.FiltMinArea = 5
        self.FiltMaxArea = 300  # 361920 max resolution
        self.FiltMinLength = 0
        self.FiltMaxLength = 1000
        self.FiltMinEccentricity = 0.0
        self.FiltMaxEccentricity = 1.0
        self.FiltMinSolidity = 0.6
        self.FiltMaxSolidity = 1.0
        self.FiltMinExtent = 0.2
        self.FiltMaxExtent = 1.0
        self.FiltArea = True
        self.FiltLength = False
        self.FiltEccentricity = False
        self.FiltSolidity = False
        self.FiltExtent = False

        # default values for Contrast sliders
        self.contrastTransformSlider_1Default = 36
        self.contrastTransformSlider_2Default = 23

        # create some properties
        self.fileName = ""  # filename of currently loaded Image
        self.fileExt = None  # File extension of currently loaded image
        self.filePath = None  # File Path of currently loaded image
        self.folderPath = None
        self.activeTool = "roi"

        # Array with all filenames which are consideres for the current session
        self.SessionFileNames = None
        self.SessionFilePaths = None

        self.pixmapFeedback = None  # Qt Pixmap of current shown Image in Feedback window
        self.pixmapProcess = None     # Same for Mask image on the left side

        # Data of image file which is currently loaded (if tiff --> normalization and conversion too 8 bit grayscale)
        self.imageDataRaw = None  # raw imported Data from the input dialog
        self.imageDataTrans = None  # contrast transformed Raw input Data
        # feedback data: overlay of Mask with contrast strechted Input Image
        self.imageDataFeedback = None
        # Data of image on the right positioned processing viewer
        self.imageDataProcess = None
        self.labelData = None  # Label Matrix for corresponding mask Data
        self.maskData = None  # Mask Data from Threhsolding and filtering
        self.roiData = None  # Data from Roi: Selected from image Data raw
        self.roiPoints = None  # Numpy array of points which define the ROI Polygon
        self.nStructs = "--"  # number of counted structs

        # Containers for Undo functionality
        self.maskDataCont = list()  # container for mask Data
        self.labelDataCont = list()  # container for label Data
        self.nStructsCont = list()  # same for number of detected structures

        self.wireActions()  # connect actions, buttons etc. and set default values
        self.wireButtons()
        self.wireSliders()
        self.wireCheckBoxes()
        self.wireSpinBoxes()
        self.wireViewers()
        self.wireFileBrowser()
        # display bottom label
        self.setBottomLabel()

        # set some inital layout things if no image was loaded
        self.filterGroupBox.setEnabled(False)
        self.maskGroupBox.setEnabled(False)
        self.roiGroupBox.setEnabled(False)
        self.contrastTransformGroupBox.setEnabled(False)

        # Wire functions to widgets and set default values !!!

    def wireActions(self):
        self.openAction.triggered.connect(
            functools.partial(self.openImage, "dialog"))
        self.exitAction.triggered.connect(self.close)
        self.dragAction.triggered.connect(
            functools.partial(self.toolSelector, "drag"))
        self.roiAction.triggered.connect(
            functools.partial(self.toolSelector, "roi"))
        self.cutAction.triggered.connect(
            functools.partial(self.toolSelector, "cut"))
        self.eraseAction.triggered.connect(
            functools.partial(self.toolSelector, "erase"))
        self.saveRoiAction.triggered.connect(
            functools.partial(self.saveImage, datafrom="roiData"))

    def wireButtons(self):
        self.maskButton.clicked.connect(self.doMasking)
        self.undoMaskButton.clicked.connect(self.undoMasking)
        self.saveRoiButton.clicked.connect(
            functools.partial(self.saveImage, datafrom="roiData"))
        self.doMeasurementButton.clicked.connect(self.doMeasurement)

    def wireSliders(self):
        # set default values
        self.contrastTransformSlider_1.setValue(
            self.contrastTransformSlider_1Default)
        self.contrastTransformSlider_2.setValue(
            self.contrastTransformSlider_2Default)

        # wiring the slider
        self.contrastTransformSlider_1.valueChanged.connect(
            self.doContrastTransform)
        self.contrastTransformSlider_2.valueChanged.connect(
            self.doContrastTransform)

    def wireCheckBoxes(self):
        # set default values
        self.areaFilterCheckBox.setChecked(self.FiltArea)
        self.lengthFilterCheckBox.setChecked(self.FiltLength)
        self.eccentFilterCheckBox.setChecked(self.FiltEccentricity)
        self.solidityFilterCheckBox.setChecked(self.FiltSolidity)
        self.extentFilterCheckBox.setChecked(self.FiltExtent)

        # wiring checkboxes to function doCheckBox
        # functools.partial allows do combine the callback doCheckbox with input arguments. Without this trick a QAction doens't allow a callback with input arguments because None is somhow returned!!
        self.areaFilterCheckBox.stateChanged.connect(functools.partial(
            self.doCheckBox, self.areaFilterCheckBox, "FiltArea"))
        self.lengthFilterCheckBox.stateChanged.connect(functools.partial(
            self.doCheckBox, self.lengthFilterCheckBox, "FiltLength"))
        self.eccentFilterCheckBox.stateChanged.connect(functools.partial(
            self.doCheckBox, self.eccentFilterCheckBox, "FiltEccentricity"))
        self.solidityFilterCheckBox.stateChanged.connect(functools.partial(
            self.doCheckBox, self.solidityFilterCheckBox, "FiltSolidity"))
        self.extentFilterCheckBox.stateChanged.connect(functools.partial(
            self.doCheckBox, self.extentFilterCheckBox, "FiltExtent"))

    def wireSpinBoxes(self):
        # set default values
        self.minAreaFilterSpinBox.setValue(self.FiltMinArea)
        self.minLengthFilterSpinBox.setValue(self.FiltMinLength)
        self.minEccentFilterSpinBox.setValue(self.FiltMinEccentricity)
        self.minSolidityFilterSpinBox.setValue(self.FiltMinSolidity)
        self.minExtentFilterSpinBox.setValue(self.FiltMinExtent)
        self.maxAreaFilterSpinBox.setValue(self.FiltMaxArea)
        self.maxLengthFilterSpinBox.setValue(self.FiltMaxLength)
        self.maxEccentFilterSpinBox.setValue(self.FiltMaxEccentricity)
        self.maxSolidityFilterSpinBox.setValue(self.FiltMaxSolidity)
        self.maxExtentFilterSpinBox.setValue(self.FiltMaxExtent)

        # Wiring Spin Boxes to function doSpinBox
        self.minAreaFilterSpinBox.valueChanged.connect(functools.partial(
            self.doSpinBox, self.minAreaFilterSpinBox, "FiltMinArea"))
        self.minLengthFilterSpinBox.valueChanged.connect(functools.partial(
            self.doSpinBox, self.minLengthFilterSpinBox, "FiltMinLength"))
        self.minEccentFilterSpinBox.valueChanged.connect(functools.partial(
            self.doSpinBox, self.minEccentFilterSpinBox, "FiltMinEccentricity"))
        self.minSolidityFilterSpinBox.valueChanged.connect(functools.partial(
            self.doSpinBox, self.minSolidityFilterSpinBox, "FiltMinSolidity"))
        self.minExtentFilterSpinBox.valueChanged.connect(functools.partial(
            self.doSpinBox, self.minExtentFilterSpinBox, "FiltMinExtent"))
        self.maxAreaFilterSpinBox.valueChanged.connect(functools.partial(
            self.doSpinBox, self.maxAreaFilterSpinBox, "FiltMaxArea"))
        self.maxLengthFilterSpinBox.valueChanged.connect(functools.partial(
            self.doSpinBox, self.maxLengthFilterSpinBox, "FiltMaxLength"))
        self.maxEccentFilterSpinBox.valueChanged.connect(functools.partial(
            self.doSpinBox, self.maxEccentFilterSpinBox, "FiltMaxEccentricity"))
        self.maxSolidityFilterSpinBox.valueChanged.connect(functools.partial(
            self.doSpinBox, self.maxSolidityFilterSpinBox, "FiltMaxSolidity"))
        self.maxExtentFilterSpinBox.valueChanged.connect(functools.partial(
            self.doSpinBox, self.maxExtentFilterSpinBox, "FiltMaxExtent"))

    def wireViewers(self):
        self.viewerFeedback.photoClicked.connect(self.viewerSlotClicked)
        self.viewerProcess.photoClicked.connect(self.viewerSlotClicked)

        self.viewerFeedback.photoClickedReleased.connect(
            self.viewerSlotClickedReleased)
        self.viewerProcess.photoClickedReleased.connect(
            self.viewerSlotClickedReleased)

        self.viewerFeedback.photoHitButton.connect(self.doRoiToolPoly)

        self.viewerFeedback.photoUndoButton.connect(self.retrieveDataState)
        self.viewerProcess.photoUndoButton.connect(self.retrieveDataState)

    def wireFileBrowser(self):
        self.fileModel = QtWidgets.QFileSystemModel()
        self.fileModel.setRootPath((QtCore.QDir.rootPath()))

        self.fileTree.setModel(self.fileModel)
        self.fileTree.setRootIndex(self.fileModel.index(self.folderPath))
        self.fileTree.setSortingEnabled(True)
        self.fileTree.setColumnHidden(1, True)
        self.fileTree.setColumnWidth(0, 250)
        self.fileTree.hide()

        # self.fileTree.customContextMenuRequested.emit
        self.fileTree.customContextMenuRequested.connect(
            self.fileTreeContextMenu)

    ################################################  Create Methods #############################################################
    def fileTreeContextMenu(self, cursor_point):
        """Creates context menu for File Browser window"""
        menu = QtWidgets.QMenu()
        open = menu.addAction("Open Image")
        open.triggered.connect(functools.partial(self.openImage, "browser"))
        cursor = QtGui.QCursor()
        menu.exec_(cursor.pos())

    def doCheckBox(self, CBox, AttributeName):
        """Checking the state of a given Check Box and changing the corresponding Attribute (must be bool). Changes self.AttributeName to (True or False)"""
        if CBox.isChecked():
            setattr(self, AttributeName, True)
        else:
            setattr(self, AttributeName, False)

    def doSpinBox(self, SBox, AttributeName):
        """Checking state of given Spin Box and changing the corresponding Attribute (can be float)"""
        val = SBox.value()  # Get value of currently changed Spin box
        # set attribute to value of current spinBox
        setattr(self, AttributeName, val)

    def openImage(self, mode="dialog"):
        """Creates file dialog and sets the imported image as pixmap in in the image display label """
        if mode == "dialog":
            options = QFileDialog.Options()
            filePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "",  # open File Dialog
                                                      "Images (*.*)", options=options)

        elif mode == "browser":
            idx = self.fileTree.currentIndex()  # index of currently selected file
            # filepath receivied from the browser window
            filePath = self.fileModel.filePath(idx)

        if not filePath:  # if file dialog is canceled filename is bool-false
            return
        else:

            # do some file path oprations
            self.filePath = filePath  # set current file Path of image which is loaded
            # save file extension from path
            self.fileExt = os.path.splitext(filePath)[1]
            self.folderPath = os.path.dirname(filePath)
            self.fileName = os.path.basename(
                filePath)  # create filename from path

            # set root path of fileSystem to folder path of currently opened image
            self.fileTree.setRootIndex(self.fileModel.index(self.folderPath))
            # Display file Tree (its first hidden that the user cannot mess around in all directories)
            self.fileTree.show()

            # needs to be adapated if non grayscale images are imported (0 flag for Grayscale images)
            if self.colorspace == 0:

                image_import = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)

                if self.fileExt == ".tif":  # if tif file is importet preprocess it

                    # clip intensity values above certain value
                    im_clip_norm = contrast_cut(
                        image_import, self.clipval, dataType="uint16")
                    # conver to 8 bit grayscale image for processing
                    self.imageDataRaw = cv2.convertScaleAbs(
                        im_clip_norm, alpha=(2**8 / 2**16))

                else:
                    self.imageDataRaw = image_import

            # if a no image file is loades a message box will pop up
            if np.all(self.imageDataRaw == None):
                QMessageBox.information(
                    self, "Error Loading Image", "Cannot load %s." % filePath)
                return

            # set Data to imported Image
            # create copy of image Data raw that cv2.contrastStretch return array  value and not None
            self.imageDataTrans = np.copy(self.imageDataRaw)

            # set widget status to begin state when image is loaded

            self.contrastTransformSlider_1.setValue(
                self.contrastTransformSlider_1Default)
            self.contrastTransformSlider_2.setValue(
                self.contrastTransformSlider_2Default)

            self.contrastTransformGroupBox.setEnabled(True)
            self.filterGroupBox.setEnabled(True)

            # reset some attributes to None
            self.labelData = None
            self.maskData = None
            self.roiData = None
            self.imageDataFeedback = None
            self.nStructs = "--"

            self.toolBar.setEnabled(True)
            self.toolSelector()
            self.setBottomLabel()  # set bottom label txt
            self.clearDataHistory()  # Clear Undo history
            # remove ROI annotation if image is loaded
            self.viewerFeedback._polygon_item.removeAllPoints()
            # set roimode to active (see PhotoViewer class) to be able to draw a Polygon
            self.viewerFeedback._roimode = "active"
            # display contrast transformed data on both viewers
            self.contrastTransform()  # do contrast transform and store it in self.imageDataProcess
            self.displayImage(self.imageDataProcess, display="both")

    def displayImage(self, ImageData, display, fitView=True, colorspace="grayscale"):
        """ Converts openCv imported Data to QImage and sets it as pixmap in the image Label """
        if ImageData is None:
            image = QImage()

        else:

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

                image = QImage(ImageData.data, width,
                               height, bytesPerLine, qformat)

        # create pixmap and set photo on certain viewer specified by the value of display
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

        # toggle corrrect drag mode: set photo always changed it needs to be refined when there is time for that
        self.viewerProcess.toggleDragMode()
        self.viewerFeedback.toggleDragMode()

    def saveImage(self,  datafrom="roiData"):
        """ Opens File Dialog and writes the current data to an uncompressed png-File """
        data = getattr(self, datafrom)

        if np.all(data == None):  # basically same procedure as openImage
            if datafrom == "roiData":
                QMessageBox.information(
                    self, "Error Saving ROI", "No ROI was selected!")
            else:
                QMessageBox.information(
                    self, "Error Saving Image", "No image was loaded!")
        else:
            filePath, _ = QFileDialog.getSaveFileName(
                self, "Save Image", "", "Image Files (*.png)")
            if not filePath:
                return
            else:
                try:
                    cv2.imwrite(filePath, data, [  # write imageData to uncompressed .png File
                        cv2.IMWRITE_PNG_COMPRESSION, 0])
                except:
                    QMessageBox.information(
                        self, "Error Saving Image", "Cannot save %s." % filePath)
                    return

    def contrastTransform(self):

        upper_contrast_val = self.contrastTransformSlider_1.value()
        lower_contrast_val = self.contrastTransformSlider_2.value()

        # calculate contrast transformed image from raw image data
        cv2.intensity_transform.contrastStretching(
            self.imageDataRaw, self.imageDataTrans, lower_contrast_val, 0, upper_contrast_val, 255)

        # if no roi is selected use transformed Raw image data for processing
        if np.all(self.roiData == None):
            self.imageDataProcess = self.imageDataTrans.copy()
        else:  # if roi Data is selecte use data from Roi for further processing
            cv2.intensity_transform.contrastStretching(
                self.roiData, self.imageDataProcess, lower_contrast_val, 0, upper_contrast_val, 255)  # do the intensity transform
        self.setBottomLabel()  # update bottom label for different slider positions

    def doContrastTransform(self):

        self.contrastTransform()

        if self.activeTool == "roi":
            self.displayImage(self.imageDataTrans,
                              display="feedback", fitView=False)
            self.displayImage(self.imageDataProcess,
                              display="process", fitView=False)

        else:
            self.displayImage(self.imageDataProcess, display="both")

    def doMasking(self):
        """creates mask from image Data process (contrast transformed image). First an adapative threshold method is used and afterwards the mask is filtered with by the function cc_filter_idx. From the filtered labels a filtered mask is calculated and displayed"""
        if np.all(self.imageDataProcess == None):  # check if image was loaded
            QMessageBox.information(
                self, "Error processing Image", "No image was loaded!")

            return

        self.maskData = cv2.adaptiveThreshold(self.imageDataProcess, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, self.thNeighborhood, 0)  # set process Data to raw mask

        # calcualte connected components of thresholded image (needed for cc_filter idx.)
        output_cc = cv2.connectedComponentsWithStats(
            self.maskData, connectivity=self.connectivity)

        # set label data to unfiltred label Matrix
        self.labelData = output_cc[1]
        # set number of structs to numer of labels - 1 (0 is Background label)
        self.nStructs = output_cc[0] - 1

        # if any Filter is enabled apply filter functions
        if np.any([self.FiltArea, self.FiltLength, self.FiltEccentricity, self.FiltSolidity, self.FiltExtent]):

            filt_idx = cc_filter_idx(output_cc, min_area=self.FiltMinArea, max_area=self.FiltMaxArea, min_length=self.FiltMinLength, max_length=self.FiltMaxLength, min_eccentricity=self.FiltMinEccentricity, max_eccentricity=self.FiltMaxEccentricity, min_solidity=self.FiltMinSolidity, max_solidity=self.FiltMaxSolidity,
                                     min_extent=self.FiltMinExtent, max_extent=self.FiltMaxExtent, filter_area=self.FiltArea, filter_length=self.FiltLength, filter_eccentricity=self.FiltEccentricity, filter_solidity=self.FiltSolidity, filter_extent=self.FiltExtent)  # calculate labels where Filter conditions are fullfilled!

            # created empty filtered label array
            filt_labels = np.zeros_like(self.labelData)

            for i in filt_idx:  # iterate over all filter indices
                # fill empty label array with filtered labels (0 is Background label)
                filt_labels[self.labelData == i] = i

            # calculate binary mask from filtered labels
            filt_mask = ((filt_labels != 0)*255).astype("uint8")

            self.maskData = filt_mask  # set image data to filtered Mask
            self.labelData = filt_labels  # set label Data to filtered label Matrix

            self.nStructs = filt_idx.size  # set struct count to size of filter idx array

        # set some settings after masking

        self.contrastTransformGroupBox.setEnabled(False)
        self.doMeasurementButton.setEnabled(True)
        self.undoMaskButton.setEnabled(True)

        self.displayImage(self.maskData, display="process",
                          fitView=False)  # display image on process side
        self.doVisualFeedback()
        self.setBottomLabel()

        self.clearDataHistory()  # clear old Data history
        self.storeDataState()  # store Data in history for Undo functionality

    def doVisualFeedback(self):
        """creates overlay of imageDataProcess (contrast transformed image) and mask. The two images are blended together and displayed on the feedback viewer"""
        if self.imageDataProcess is None or self.maskData is None:  # check if some of the required attributes are None if so feedback data is None
            self.imageDataFeedback = None
        else:
            raw3chan = cv2.cvtColor(self.imageDataProcess, cv2.COLOR_GRAY2RGB)

            colorMask = np.zeros_like(raw3chan)
            colorMask[:, :, 0] = self.maskData

            self.imageDataFeedback = cv2.addWeighted(
                raw3chan, (1-self.alphaBlend), colorMask, self.alphaBlend, 0)

            self.displayImage(self.imageDataFeedback,
                              display="feedback", fitView=False, colorspace="rgb")

    def undoMasking(self):
        """reseting the overlay of mask and image Data to normal View"""

        self.displayImage(self.imageDataProcess, display="both", fitView=True)
        self.nStructs = "--"  # set bottom label struct count
        self.setBottomLabel()
        self.imageDataFeedback = None
        self.maskData = None
        self.labelData = None
        self.contrastTransformGroupBox.setEnabled(True)
        self.doMeasurementButton.setEnabled(False)
        self.undoMaskButton.setEnabled(False)

    def setBottomLabel(self):
        "creates a string which is displayed in the bottom laybel for user information"
        dispStr = " File: " + self.fileName + "       Lower Th: " + str(self.contrastTransformSlider_2.value()) + "       Upper Th: " + str(
            self.contrastTransformSlider_1.value()) + "        Struct count: " + str(self.nStructs) + "        Interaction Mode: " + self.activeTool
        self.bottomLabel.setText(dispStr)

    def toolSelector(self, selected_tool="roi"):
        """sets tool attribute of Photoviewer so that the correct tool from the Toolbar is selected"""

        # switch block for controlling fit screen functionality of displayImage function!
        # if a direct switch between cut and erase tool happens  the display function will not fit to the screen (avoids popping around)
        if self.activeTool == "cut" and (selected_tool == "cut" or selected_tool == "erase"):
            fitCutErase = False
        elif self.activeTool == "erase" and (selected_tool == "cut" or selected_tool == "erase"):
            fitCutErase = False
        else:
            fitCutErase = True

        # Selection of tool
        # set attribute for current tool (drag, roi, cut, erase)
        self.activeTool = selected_tool

        self.setBottomLabel()  # set bottom label to correct tool which is currently selected

        # set correct drag mode of Feedback Viewer
        self.viewerFeedback._tool = selected_tool
        # set correct drag mode of Process Viewer
        self.viewerProcess._tool = selected_tool

        self.viewerFeedback.toggleDragMode()  # toogle drag mode of image viewers
        self.viewerProcess.toggleDragMode()

        if selected_tool == "drag":  # set correct check state for each button in toolbar
            self.dragAction.setChecked(True)
            self.roiAction.setChecked(False)
            self.cutAction.setChecked(False)
            self.eraseAction.setChecked(False)

        elif selected_tool == "roi":
            self.dragAction.setChecked(False)
            self.roiAction.setChecked(True)
            self.cutAction.setChecked(False)
            self.eraseAction.setChecked(False)

            self.undoMasking()

            self.maskGroupBox.setEnabled(False)
            self.roiGroupBox.setEnabled(True)

            self.viewerFeedback._modifiable = True
            self.viewerProcess._modifiable = False

            # if no roi is selected use Raw contstrast transoimage data for processing
            if np.all(self.roiData == None):
                self.displayImage(self.imageDataTrans, display="both")
            else:  # if roi Data is selecte use data from Roi for further processing
                self.displayImage(self.imageDataTrans, display="feedback")
                self.displayImage(self.imageDataProcess, display="process")

        elif selected_tool == "cut":
            self.dragAction.setChecked(False)
            self.roiAction.setChecked(False)
            self.cutAction.setChecked(True)
            self.eraseAction.setChecked(False)

            self.maskGroupBox.setEnabled(True)
            self.roiGroupBox.setEnabled(False)

            self.viewerFeedback._modifiable = True
            self.viewerProcess._modifiable = True

            if self.imageDataFeedback is None and self.maskData is None:
                self.displayImage(self.imageDataProcess,
                                  display="both", fitView=fitCutErase)
            else:
                self.displayImage(
                    self.imageDataFeedback, display="feedback", fitView=fitCutErase, colorspace="rgb")
                self.displayImage(
                    self.maskData, display="process", fitView=fitCutErase)

        elif selected_tool == "erase":
            self.dragAction.setChecked(False)
            self.roiAction.setChecked(False)
            self.cutAction.setChecked(False)
            self.eraseAction.setChecked(True)

            self.maskGroupBox.setEnabled(True)
            self.roiGroupBox.setEnabled(False)

            self.viewerFeedback._modifiable = True
            self.viewerFeedback._modifiable = True

            if self.imageDataFeedback is None and self.maskData is None:
                self.displayImage(self.imageDataProcess,
                                  display="both", fitView=fitCutErase)
            else:
                self.displayImage(
                    self.imageDataFeedback, display="feedback", fitView=fitCutErase, colorspace="rgb")
                self.displayImage(
                    self.maskData, display="process", fitView=fitCutErase)

    def viewerSlotClicked(self, press_point):
        """recieves mouse press coordinate from the currently clicked viewer. This slot is for tools which just need 1 click"""
        if self.activeTool == "drag":
            pass

        elif self.activeTool == "erase":
            self.doEraseTool(press_point)

    def viewerSlotClickedReleased(self, press_point, release_point):
        """recieves mouse press and release coordinate from the currently clicked viewer. This slot is for tools which need 1 click and 1 release coordinate"""
        if self.activeTool == "roi":
            pass

        elif self.activeTool == "cut":

            self.doCutTool(press_point, release_point)

    def doRoiToolPoly(self, roi_points):
        # covert qt points to numpy array for usage with open cv
        if roi_points:  # if list of roi points is not empty do roi calculation
            pts_list = []  # create list for storing
            for point in roi_points:  # iterate over all points
                # extract x and y coordinate of qt points put it to list and append it on storing list
                pts_list.append([point.x(), point.y()])
            # convert list created to numpy array
            cvpoints = np.array(pts_list, dtype=np.int32)

            # extract image shape for mask creation
            height = self.imageDataRaw.shape[0]
            width = self.imageDataRaw.shape[1]

            # create mask array
            mask = np.zeros((height, width), dtype=np.uint8)

            # create mask with ones where the polygon area is located
            cv2.fillPoly(mask, [cvpoints], (255))

            # perform bitwise and values >1 are treated as 1
            masked_image = cv2.bitwise_and(
                self.imageDataRaw, self.imageDataRaw, mask=mask)

            # calculate bounding rectangle for cropping the image
            bounding_rect = cv2.boundingRect(cvpoints)

            roiData = masked_image[bounding_rect[1]: bounding_rect[1] + bounding_rect[3], bounding_rect[0]                                   : bounding_rect[0] + bounding_rect[2]].copy()  # cropp image to appropriate size

            # check if roi selection has shape dimension 2 and non zero dimension
            if np.all(roiData.shape) != 0 and len(roiData.shape) == 2:

                # display Roi on process side
                self.displayImage(roiData, display="process", fitView=True)
                # set viewer to not modifiable (avoiding user from using the roi tool on the process side)
                self.viewerProcess._modifiable = False
                # set attribute for accessing the roi data by other routines
                setattr(self, "roiData", roiData)
                # set roiPoints to numpy array containing the corners of the polygon
                setattr(self, "roiPoints", cvpoints)
                self.imageDataProcess = self.roiData.copy()  # copy roi data to process data
                self.contrastTransform()  # apply contrast transform
                self.displayImage(self.imageDataProcess,
                                  display="process", fitView=True)

        else:  # if list of roi points is empty
            self.displayImage(self.imageDataTrans,
                              display="both", fitView=True)
            # set attribute for accessing the roi data by other routines
            setattr(self, "roiData", None)
            # set roiPoints to numpy array containing the corners of the polygon
            setattr(self, "roiPoints", None)
            self.imageDataProcess = self.imageDataTrans.copy()

    def doRoiTool(self, press_point, release_point):
        """calculate roi from rawImage data and press and release points in image Viewer. This method is obsulete. In the current version doRoiTool Poly is used"""
        pp = np.array([press_point.x(), press_point.y()]
                      )  # save press and release QT points as numpy arrays
        rp = np.array([release_point.x(), release_point.y()])

        # do roi calculation for different quadrants (posibilities to describe rectangle by two points) then choose indicies (pixel coordinates accordingly)
        if np.all(rp >= pp):
            roiData = self.imageDataRaw[pp[1]:rp[1], pp[0]:rp[0]].copy()

        elif np.all(rp < pp):
            roiData = self.imageDataRaw[rp[1]:pp[1], rp[0]:pp[0]].copy()

        elif rp[0] < pp[0] and rp[1] >= pp[1]:
            roiData = self.imageDataRaw[pp[1]:rp[1], rp[0]:pp[0]].copy()

        elif rp[0] >= pp[0] and rp[1] < pp[1]:
            roiData = self.imageDataRaw[rp[1]:pp[1], pp[0]:rp[0]].copy()

        # check if roi selection has shape dimension 2 and non zero dimension
        if np.all(roiData.shape) != 0 and len(roiData.shape) == 2:

            # display Roi on process side
            self.displayImage(roiData, display="process", fitView=True)
            # set viewer to not modifiable (avoiding user from using the roi tool on the process side)
            self.viewerProcess._modifiable = False
            # set attribute for accessing the roi data by other routines
            setattr(self, "roiData", roiData)

    def doCutTool(self, press_point, release_point):
        "Sets Mask Data along a line between press and release point to 0. From the old filtered mask a new mask with labels is calculated"

        if self.maskData is None:  # if no mask data is available do nothing
            return

        # save press and release QT points as numpy arrays
        pp = np.array([press_point.x(), press_point.y()])
        rp = np.array([release_point.x(), release_point.y()])

        # draw line with black color over mask data --> setting the zeros in the mask to zero :)
        cv2.line(self.maskData, pp, rp, 0, thickness=1, lineType=4, shift=0)
        # recalculate connected components of the cutted mask
        n_labels, label_matrix = cv2.connectedComponents(
            self.maskData, connectivity=self.connectivity)

        # update number of detecteed structs (-1 from Background subtraction)
        self.nStructs = n_labels - 1
        self.labelData = label_matrix  # update label Data

        self.displayImage(self.maskData, display="process", fitView=False)
        self.doVisualFeedback()
        self.setBottomLabel()
        self.storeDataState()  # create history entry for Undo functionality

    def doEraseTool(self, press_point):
        """Choses point in mask Data and sets points with the same label to Zero"""

        if self.labelData is None or self.maskData is None:  # if label or mask data is empty return
            return

        # convert presspoint to numpy array
        pp = np.array([press_point.x(), press_point.y()])

        # extract the label number of the clicked point
        erase_label = self.labelData[pp[1], pp[0]]

        # filter erase label from Mask Data
        new_mask = np.logical_and(
            self.labelData != erase_label, self.maskData == 255)
        # create uint8 numpy array from bool area in the previous step
        new_mask = (new_mask*255).astype("uint8")

        # set every label entry which corresponds to the erase_label number to zero (Background)
        self.labelData[self.labelData == erase_label] = 0
        self.maskData = new_mask  # update self attribute

        if erase_label != 0:  # if no background pixel was clicked, increment number of counted structs by one
            self.nStructs -= 1

        self.displayImage(self.maskData, display="process",
                          fitView=False)  # display new mask
        self.doVisualFeedback()  # do visual feedback of new mask Data
        self.setBottomLabel()  # update bottom label
        self.storeDataState()  # create history entry for Undo functionality

    def storeDataState(self):
        """Function implementation for storing the current State of Cut and Erase Tool. Necessary for an Undo functionality"""

        if len(self.maskDataCont) != 0 and len(self.labelDataCont) != 0 and len(self.nStructsCont) != 0:
            # if the Data hasn't changed don't store it in Data history (avoids that clicking with erase and cut makes redundant History)
            if (np.all(self.maskDataCont[-1] == self.maskData) and np.all(self.labelDataCont[-1] == self.labelData) and (self.nStructsCont[-1] == self.nStructs)):
                return

        # append copy of current Data to storing container (list)
        self.maskDataCont.append(self.maskData.copy())
        self.labelDataCont.append(self.labelData.copy())
        self.nStructsCont.append(self.nStructs)

        # if storing list len exceeds maximum number of undos delete the first (oldest) element in the history
        if len(self.maskDataCont) > self.maxUndos and len(self.labelDataCont) > self.maxUndos and len(self.nStructsCont) > self.maxUndos:
            self.maskDataCont.pop(0)
            self.labelDataCont.pop(0)
            self.nStructsCont.pop(0)

    def retrieveDataState(self):
        """Function implementation for an Undo functionality. """

        # len of Data container is 0 the Undo Functionalty should do nothing because the history is empty.
        if (len(self.maskDataCont) == 0 or len(self.maskDataCont) == 1) and (len(self.labelDataCont) == 0 or len(self.labelDataCont) == 1) and (len(self.nStructsCont) == 0 or len(self.nStructsCont) == 1):
            return

        # if history has proper length of 2 the undo functionality can be used
        else:
            # old Data element from history
            self.maskData = self.maskDataCont[-2].copy()
            self.labelData = self.labelDataCont[-2].copy()
            self.nStructs = self.nStructsCont[-2]

            # Delete newest entry from list: selected element is now last history entry.
            self.maskDataCont.pop()
            self.labelDataCont.pop()
            self.nStructsCont.pop()

            # set display and all the other stuff
            self.displayImage(self.maskData, display="process",
                              fitView=False)  # display  mask
            self.doVisualFeedback()  # do visual feedback of new mask Data
            self.setBottomLabel()  # update bottom label

    def clearDataHistory(self):
        """Function for clearing the storing containers of the Data history. Necessary for loading actions to start with a fresh History"""
        self.maskDataCont.clear()
        self.labelDataCont.clear()
        self.nStructsCont.clear()

    def doMeasurement(self):
        """Using current mask data for measurement and write to a .csv File"""
        
        #Create measurement
        output_cc = cv2.connectedComponentsWithStats(
            self.maskData, connectivity=self.connectivity)  # calculate connected components from current mask Data

        dict_data, labels = cc_measurement(output_cc)

        csv_columns = ["Label", "Area", "Length",
                       "Eccentricity", "Solidity", "Extent"]

        
        
        #Write measurement Data to csv
        csv_file = "measurement_test.csv"

        try:
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns, lineterminator = "\n")
                writer.writeheader()
                for data in dict_data:
                    writer.writerow(data)
        except IOError:
            QMessageBox.information(
                self, "Error writing CSV", "Could not write measurement.csv")
            return

        #Write Labelmatrix to csv
        np.savetxt("label_test.csv", labels, delimiter=",")
        #Mask Data

# Launch app
def main():

    app = QApplication(sys.argv)
    win = LoadQt()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
