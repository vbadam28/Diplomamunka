from PySide6.QtWidgets import QMainWindow
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QSize, Qt, QPoint
from PySide6.QtGui import QAction, QPixmap, QImage, QPainter, QColor
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget, QMenu, QFileDialog, QPushButton

import os

import cv2
import numpy as np

from DebugLogic import DebugLogic
from Logic import  Logic

class MainWindow(QMainWindow):

    def __init__(self ,logic=None,debugLogic=None):
        super().__init__()

        self.releaseLogic=Logic()
        self.debugLogic = DebugLogic()

        self.logic=self.releaseLogic


        self.setWindowTitle("Image Viewer")
        #self.setGeometry(100, 100, 800, 600)

        # Create a QLabel to display the image
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        # Create a button to open the file dialog
        self.button = QPushButton("Choose Image", self)
        self.button.clicked.connect(self.open_file_dialog)  # Connect button click to the function

        self.selectAlg = QtWidgets.QComboBox()
        self.selectAlg.addItems(["Alg 1","Alg 2", "Custom"])
        self.selectAlg.currentIndexChanged.connect(self.onAlgChange)

        self.selectType = QtWidgets.QComboBox()
        self.selectType.addItems(["Release","Debug"])
        self.selectType.currentIndexChanged.connect(self.onTypeChange)

        self.evaluateBtn = QPushButton("Kiértékelés")
        self.evaluateBtn.clicked.connect(self.evaluatePicture)



        # Set up the layout
        self.origImagePlaceholder = QLabel("bal")
        self.resultImagePlaceholder = QLabel("jobb")

        imageLayout = QtWidgets.QHBoxLayout()
        imageLayout.addWidget(self.origImagePlaceholder)
        imageLayout.addWidget(self.resultImagePlaceholder)

        selectorLayout = QtWidgets.QHBoxLayout()
        selectorLayout.addWidget(self.selectAlg)
        selectorLayout.addWidget(self.selectType)

        layout = QVBoxLayout()
        layout.addWidget(self.button)

        layout.addLayout(selectorLayout)

        layout.addWidget(self.label)
        layout.addLayout(imageLayout)
        layout.addWidget(self.evaluateBtn)

        # Set the layout inside a QWidget to be set as the central widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        #képek a megjelenítéshez
        self.original_pixmap = None
        self.modified_pixmap = None
        self.dots = []
        self.image = None
        self.slice = 0

    def onAlgChange(self,idx):
        value = self.selectAlg.currentIndex()
        print(f"Selected: {value},  {idx},   {self.selectAlg.currentData()}")

    def onTypeChange(self,idx):
        if idx==0:
            self.logic = self.releaseLogic
        else:
            self.logic = self.debugLogic
        return

    def open_file_dialog(self):
        # Open a file dialog to select an image file
        file_path, c = QFileDialog.getOpenFileName(self, "Choose Image", "",
                                                   "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif *.nii *.nii.gz)")

        if not file_path:  # If a file was selected
            return

        ext = os.path.splitext(file_path)[1].lower()
        self.slice = 0

        # ezen a ponton lehet rgb vagy szürkeárnyalatos

        if ext in ['.nii','.nii.gz']:
            self.image = self.logic.loadNifti(file_path)
            self.slice = self.image.shape[2]//2
        else:
            self.image = np.moveaxis(np.array([cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)]),0,-1)

        self.display_image()

    def display_image(self, resultPixmap = None):
        normImg = cv2.normalize(self.image[:,:,self.slice].astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cvImage = cv2.cvtColor(normImg, cv2.COLOR_GRAY2RGB)

        h,w,_ = cvImage.shape
        qim = QImage( cvImage,w,h,cvImage.strides[0],QImage.Format_RGB888 )

        self.original_pixmap = QPixmap(qim)
        self.modified_pixmap = self.original_pixmap.copy()

        # Set the pixmap to the label
        #self.label.setPixmap(self.modified_pixmap)
        if resultPixmap is None:
            self.origImagePlaceholder.setPixmap(self.modified_pixmap)
        else:
            self.resultImagePlaceholder.setPixmap(resultPixmap)

    def wheelEvent(self, event):
        if self.image is None or self.image.shape[2] == 1:
            return
        delta = event.angleDelta().y()
        if delta>0:
            self.slice = min(self.slice+1,self.image.shape[2]-1)
        else:
            self.slice = max(self.slice-1,0)

        self.display_image()
        return
    def mousePressEvent(self, event):
        if self.modified_pixmap is None:
            return  # No image loaded, do nothing

        # Get the position where the mouse was clicked
        mouse_pos = event.pos()

        # Convert mouse position to relative position on the pixmap
        #relative_pos = self.label.mapFromParent(mouse_pos)
        relative_pos = self.origImagePlaceholder.mapFromParent(mouse_pos)

        if not self.modified_pixmap.rect().contains(relative_pos):
            return  # Clicked outside the image

        if event.button() == Qt.LeftButton:
            # Left-click: Draw a big dot at the clicked position
            self.place_dot(relative_pos)
        elif event.button() == Qt.RightButton:
            # Right-click: Remove the dot (restore original pixel)
            self.remove_dot(relative_pos)

    def place_dot(self, pos: QPoint):
        # Draw a big dot on the image at the specified position
        dot_radius = 5
        painter = QPainter(self.modified_pixmap)
        painter.setBrush(QColor(255, 0, 0))  # Red color for the dot
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(pos - QPoint(dot_radius // 2, dot_radius // 2), dot_radius, dot_radius)
        painter.end()

        # Save the dot position and update the pixmap
        self.dots.append(pos)
        #self.label.setPixmap(self.modified_pixmap)
        self.origImagePlaceholder.setPixmap(self.modified_pixmap)

    def remove_dot(self, pos: QPoint):
        # Remove the dot by restoring the original pixel at the clicked position
        if not self.dots:
            return  # No dots placed, do nothing

        # Find the closest dot to the clicked position
        closest_dot = min(self.dots, key=lambda dot: (dot - pos).manhattanLength())

        # Check if the click is close enough to a dot to be considered a right-click on it
        if (closest_dot - pos).manhattanLength() < 20:
            self.dots.remove(closest_dot)  # Remove the dot from the list

            # Restore the original pixel by copying from the original image
            painter = QPainter(self.modified_pixmap)
            painter.drawPixmap(closest_dot - QPoint(15 // 2, 15 // 2),
                               self.original_pixmap.copy(closest_dot.x() - 15 // 2, closest_dot.y() - 15 // 2, 15, 15))
            painter.end()

            #self.label.setPixmap(self.modified_pixmap)  # Update the displayed image
            self.origImagePlaceholder.setPixmap(self.modified_pixmap)  # Update the displayed image

    def evaluatePicture(self):
        if self.original_pixmap == None:
            print("alert dialog: choose image")
            return

        mask = None
        alg = self.selectAlg.currentIndex()

        if alg == 0:
            mask = self.logic.executeAlg2(self.image[:,:,self.slice]).astype(np.uint8)
        elif alg == 1:
            self.logic.select5Seed(self.image[:,:, self.slice])
        elif alg == 3:
            pass


        if mask is not None:
            normImg = cv2.normalize(self.image[:, :, self.slice].astype(np.float32), None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)
            cvImage = cv2.cvtColor(normImg, cv2.COLOR_GRAY2RGB)

            h, w, _ = cvImage.shape

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                cv2.drawContours(cvImage, [c], -1, (255, 0, 0), thickness=2)

            qim = QImage(cvImage, w, h, cvImage.strides[0], QImage.Format_RGB888)
            self.display_image(QPixmap(qim))

        elif len(self.dots)>0:
            mask = np.zeros((self.image.shape[0] + 2, self.image.shape[1] + 2), dtype=np.uint8)
            print(mask.shape)
            print(self.image.shape)
            resImage = self.image.copy()
            for dot in self.dots:
                retval, resImage, mask, rect = cv2.floodFill(self.image[:,:,self.slice], mask, (dot.x(),dot.y()),(255,0,0),(30,30,30),(30,30,30), flags=4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8))

            print("retval", retval)
            cv2.imshow("mask", mask)

            print("rect", rect)


        return