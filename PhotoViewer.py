from PyQt5 import QtCore, QtGui, QtWidgets
from AnnotationClasses import PolygonAnnotation

class PhotoViewer(QtWidgets.QGraphicsView):
    
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)
    photoClickedReleased = QtCore.pyqtSignal(QtCore.QPoint, QtCore.QPoint)
    photoHitButton = QtCore.pyqtSignal(list)
    
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
        self._polygon_item = PolygonAnnotation() ##adding a Polygon for Roi selection
        self._scene.addItem(self._polygon_item)     ##add polygon to scene
        self._roimode = "active"
            
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
                #self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
                self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
                #self._photo.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))    
            else:
                self.setDragMode(QtWidgets.QGraphicsView.NoDrag)

    
    def mousePressEvent(self, event):  #defines the mouse press event if the scene is clicked
        if self._photo.isUnderMouse():
            if self._modifiable and not self._tool == "roi":
                self._pressPoint = self.mapToScene(event.pos()).toPoint() #store press point
                self.photoClicked.emit(self._pressPoint) #emit signal which triggers action in main app
            
            elif self._modifiable and self._tool == "roi" and self._roimode == "active":
                self._polygon_item.removeLastPoint()
                self._polygon_item.addPoint(self.mapToScene(event.pos()).toPoint())
                # movable element
                self._polygon_item.addPoint(self.mapToScene(event.pos()).toPoint())
        super(PhotoViewer, self).mousePressEvent(event) # ad mousePressEvent to modified GraphicsView class (Photoviewer)


    def mouseReleaseEvent(self, event):  #defines the mouse release event if the scene is clicked
        if self._photo.isUnderMouse():
            if self._modifiable  and not self._tool == "roi":
                self._releasePoint = self.mapToScene(event.pos()).toPoint() #store release point
                self.photoClickedReleased.emit(self._pressPoint, self._releasePoint) #emit signal which triggers action in main app

        super(PhotoViewer, self).mouseReleaseEvent(event) # ad mousePressEvent to modified GraphicsView class (Photoviewer)
  
    def mouseMoveEvent(self, event):
        if self._photo.isUnderMouse():
            if self._modifiable and self._tool == "roi" and self._roimode == "active":
                self._polygon_item.movePoint(self._polygon_item.number_of_points()-1, self.mapToScene(event.pos()).toPoint())
        
        super(PhotoViewer, self).mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._polygon_item.isUnderMouse():
            if self._modifiable and self._tool == "roi" and self._roimode == "passive":
                self._roimode = "active"
        super(PhotoViewer, self).mouseDoubleClickEvent(event)
    
    def keyPressEvent(self, event):
        if self._photo.isUnderMouse():
            if self._modifiable and self._tool == "roi":
                if event.key() == QtCore.Qt.Key_Escape:
                    self._roimode = "passive"

                if event.key() == QtCore.Qt.Key_Return:
                    if self._polygon_item.m_points: #checks if polygon points list is empty
                        self.photoHitButton.emit(self._polygon_item.m_points)
                        self._roimode = "passive"

                if event.key() == QtCore.Qt.Key_Delete:
                    self._polygon_item.removeAllPoints()
                    self.photoHitButton.emit(self._polygon_item.m_points)
                    self._roimode = "active"
                   
                
        super(PhotoViewer, self).keyPressEvent(event)