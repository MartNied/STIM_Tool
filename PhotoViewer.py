from PyQt5 import QtCore, QtGui, QtWidgets

class PhotoViewer(QtWidgets.QGraphicsView):
    
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)
    photoClickedReleased = QtCore.pyqtSignal(QtCore.QPoint, QtCore.QPoint)

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self._tool = "drag" #for accessing different tools from outside, should act like a switch for different drag behaviours
        self._pressPoint = QtCore.QPoint(0,0) #for storing current press point
        self._releasePoint = QtCore.QPoint(0,0) #same for release
        self._modifiable = True
        
        
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None, fitView=True):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        
        if fitView:
            self.fitInView()

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        """Switching to different drag modes if different tools are selected"""
        if  self._photo.pixmap().isNull():

            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        
        elif not self._photo.pixmap().isNull():

            if self._tool == "drag":
                self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

            elif self._tool == "roi":
                self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)    
            else:
                self.setDragMode(QtWidgets.QGraphicsView.NoDrag)


    
    def mousePressEvent(self, event):  #defines the mouse press event if the scene is clicked
        if self._photo.isUnderMouse():
            if self._modifiable:
                self._pressPoint = self.mapToScene(event.pos()).toPoint() #store press point
                self.photoClicked.emit(self._pressPoint) #emit signal which triggers action in main app
            
        super(PhotoViewer, self).mousePressEvent(event) # ad mousePressEvent to modified GraphicsView class (Photoviewer)



    def mouseReleaseEvent(self, event):  #defines the mouse release event if the scene is clicked
        if self._photo.isUnderMouse():
            if self._modifiable:
                self._releasePoint = self.mapToScene(event.pos()).toPoint() #store release point
                self.photoClickedReleased.emit(self._pressPoint, self._releasePoint) #emit signal which triggers action in main app

        super(PhotoViewer, self).mouseReleaseEvent(event) # ad mousePressEvent to modified GraphicsView class (Photoviewer)
  