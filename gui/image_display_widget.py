from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QPainter, QColor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
import cv2
import numpy as np

class ImageDisplayWidget(QWidget):
    def __init__(self, enabledraw=False, images=[],originalImage=None, currIdx=0, dotRadius=2, dotColor=QColor(255,0,0)):
        super().__init__()

        self._images = images
        self._slice = currIdx
        self.originalImage = originalImage

        self.imageCount = QLabel(f"0/0 image",self)
        self.imagePlaceholder = QLabel(self)

        imageLayout = QVBoxLayout()
        self.imageCount.setAlignment(Qt.AlignHCenter)
        imageLayout.addWidget(self.imageCount)
        imageLayout.addWidget(self.imagePlaceholder)
        self.setLayout(imageLayout)


        self.originalPixmap = None
        self.modifiedPixmap = None
        self.dots = [[] for _ in range(len(images))]
        self.dotRadius = dotRadius
        self.dotColor = dotColor
        self.enableDraw=enabledraw
        self._type = 'mask'

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, images):
        self._images = images
        self.slice=0
        self.dots = [[] for _ in range(len(self._images))]
        self.imageCount.setText(f"{self.slice+(1 if self.images.shape[2]>0 else 0)}/{self.images.shape[2]} image")
    @property
    def slice(self):
        return self._slice
    @slice.setter
    def slice(self,idx):
        self._slice = idx
        self.imageCount.setText(f"{self.slice+(1 if self.images.shape[2]>0 else 0)}/{self.images.shape[2]} image")

    @property
    def displayType(self):
        return self._type
    @displayType.setter
    def displayType(self, val):
        self._type=val
        self.displayImage()

    @property
    def image(self):
        return self._images[:,:,self.slice]

    @property
    def seeds(self):
        return self.dots[self.slice] if len(self.dots[self.slice])>0 else None


    def wheelEvent(self, event):
        if self.images is None or len(self.images)==0 or self.images.shape[2] == 1:
            return
        delta = event.angleDelta().y()
        if delta>0:
            self.slice = min(self.slice+1,self.images.shape[2]-1)
        else:
            self.slice = max(self.slice-1,0)

        self.imageCount.setText(f"{self.slice+(1 if self.images.shape[2]>0 else 0)}/{self.images.shape[2]} image")
        self.displayImage()

        return

    def displayImage(self):
        if self.images is None or len(self.images)==0:
            return
        normImg = cv2.normalize(self.images[:,:,self.slice].astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cvImage = cv2.cvtColor(normImg, cv2.COLOR_GRAY2RGB)

        h,w,_ = cvImage.shape
        qim = QImage( cvImage,w,h,cvImage.strides[0],QImage.Format_RGB888 )

        self.originalPixmap = QPixmap(qim)
        self.modifiedPixmap = self.originalPixmap.copy()

        painter = QPainter(self.modifiedPixmap)
        for dot in self.dots[self.slice]:
            painter.setBrush(self.dotColor)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(dot, self.dotRadius, self.dotRadius)
        painter.end()

        if self._type == 'contour':
            origCvImage = cv2.cvtColor(
                cv2.normalize(self.originalImage.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLOR_GRAY2RGB)

            normImg[normImg!=0]=255

            contours, _ = cv2.findContours(normImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                cv2.drawContours(origCvImage, [c], -1, (255, 0, 0), thickness=2)
            h, w, _ = origCvImage.shape
            qim = QImage(origCvImage, w, h, origCvImage.strides[0], QImage.Format_RGB888)
            self.modifiedPixmap = QPixmap(qim)

        self.imagePlaceholder.setPixmap(self.modifiedPixmap)

    def mousePressEvent(self, event: QMouseEvent, /) -> None:
        if not self.enableDraw:
            return
        if self.modifiedPixmap is None:
            return

        mousePos = event.pos()
        relativePos = self.imagePlaceholder.mapFromParent(mousePos)

        if not self.modifiedPixmap.rect().contains(relativePos):
            return #outside image click

        if event.button() == Qt.LeftButton:
            self.placeDot(relativePos)
        elif event.button() == Qt.RightButton:
            self.removeDot(relativePos)

        self.displayImage()

    def placeDot(self, pos: QPoint):
        dot = pos - QPoint(self.dotRadius // 2, self.dotRadius // 2)
        self.dots[self.slice].append(dot)


    def removeDot(self, pos: QPoint):
        if not self.dots[self.slice]:
            return

        closestDot = min(self.dots[self.slice], key=lambda dot: (dot-pos).manhattanLength())
        dist = (closestDot - pos).manhattanLength()
        if dist > self.dotRadius*2:
            return
        self.dots[self.slice].remove(closestDot)
