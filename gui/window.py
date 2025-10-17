from PySide6 import QtWidgets
from PySide6.QtCore import QSize, Qt, QPoint
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget, QMenu, QFileDialog, QPushButton

import os

import cv2
import numpy as np
import nibabel as nib

from gui.control_window import ControlWindow
from gui.image_display_widget import ImageDisplayWidget
from logic.pipeline.pipelineContext import PipelineContext
from logic.pipeline.pipelineFactory import PipelineFactory


class MainWindow(QMainWindow):

    def __init__(self ,logic=None,debugLogic=None):
        super().__init__()

        self.debug = False
        self.pipeline = PipelineFactory.select5Seeds()
        self.ctx = PipelineContext()
        controls = getattr(self.pipeline, "controls", [])
        self.controlWidget = ControlWindow(self.ctx, controls, parent=self)

        self.controlAreaLayout = QVBoxLayout()
        self.controlAreaLayout.addWidget(self.controlWidget)

        self.setWindowTitle("Image Viewer")
        #self.setGeometry(100, 100, 800, 600)

        # Create a QLabel to display the image
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        # Create a button to open the file dialog
        self.button = QPushButton("Choose Image", self)
        self.button.clicked.connect(self.openFileDialog)

        self.selectAlg = QtWidgets.QComboBox()
        self.selectAlg.addItems(["select 5 seeds","divergence","divergence Blur", "Manual", "SplitMerge Gmm","Sliding Windows","Enhanced Divergence"])
        self.selectAlg.currentIndexChanged.connect(self.onAlgChange)

        self.selectType = QtWidgets.QComboBox()
        self.selectType.addItems(["Release","Debug"])
        self.selectType.currentIndexChanged.connect(self.onTypeChange)

        self.resultType = QtWidgets.QComboBox()
        self.resultType.addItems(["Masks","Contour"])
        self.resultType.currentIndexChanged.connect(self.onResultTypeChange)

        self.evaluateBtn = QPushButton("Kiértékelés")
        self.evaluateBtn.clicked.connect(self.evaluatePicture)



        # Set up the layout
        self.origImagePlaceholder = ImageDisplayWidget(enabledraw=True) #QLabel("bal")
        self.resultImagePlaceholder=ImageDisplayWidget() #QLabel("jobb")

        imageLayout = QtWidgets.QHBoxLayout()
        imageLayout.addWidget(self.origImagePlaceholder)
        imageLayout.addWidget(self.resultImagePlaceholder)
        imageLayout.addLayout(self.controlAreaLayout)

        selectorLayout = QtWidgets.QHBoxLayout()
        selectorLayout.addWidget(self.selectAlg)
        selectorLayout.addWidget(self.selectType)
        selectorLayout.addWidget(self.resultType)

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
        self.ctx = PipelineContext()#self.origImagePlaceholder.image, self.debug,self.origImagePlaceholder.seeds)
        if idx==0:
            self.pipeline = PipelineFactory.select5Seeds()
        elif idx==1:
            self.pipeline = PipelineFactory.divergenceSeeds()
        elif idx==2:
            self.pipeline = PipelineFactory.divergenceSeedsWithGaussianBlur()
        elif idx==3:
            self.pipeline = PipelineFactory.manualSeeds()
        elif idx == 4:
            self.pipeline = PipelineFactory.splitmergeGmm()
        elif idx == 5:
            self.pipeline = PipelineFactory.slidingWindows()
        elif idx==6:
            self.pipeline = PipelineFactory.enhancedDivergence()
        else:
            pass

        if getattr(self, "controlWidget", None) is not None:
            self.controlAreaLayout.removeWidget(self.controlWidget)
            self.controlWidget.deleteLater()
            self.controlWidget = None

        controls = getattr(self.pipeline, "controls", [])
        self.controlWidget = ControlWindow(self.ctx, controls, parent=self)
        self.controlAreaLayout.addWidget(self.controlWidget)

    def onTypeChange(self,idx):
        if idx==0:
            self.debug = False
        else:
            self.debug = True
        return
    def onResultTypeChange(self, idx):
        if idx==1:
            self.resultImagePlaceholder.displayType = "contour"
        else:
            self.resultImagePlaceholder.displayType = "mask"
        return
    def openFileDialog(self):
        # Open a file dialog to select an image file
        filePath, c = QFileDialog.getOpenFileName(self, "Choose Image", "",
                                                   "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif *.nii *.nii.gz)")

        if not filePath:  # If a file was selected
            return

        ext = os.path.splitext(filePath)[1].lower()
        self.slice = 0

        # ezen a ponton lehet rgb vagy szürkeárnyalatos

        if ext in ['.nii','.nii.gz']:
            self.image = self.loadNifti(filePath)
            self.slice = self.image.shape[2]//2
            self.origImagePlaceholder.images = self.image
            self.origImagePlaceholder.slice = self.image.shape[2] // 2

        else:
            self.image = np.moveaxis(np.array([cv2.imread(filePath,cv2.IMREAD_GRAYSCALE)]),0,-1)
            self.slice = 0
            self.origImagePlaceholder.images = self.image

        self.origImagePlaceholder.displayImage()


    def evaluatePicture(self):
        if self.origImagePlaceholder.images is None or len(self.origImagePlaceholder.images)==0:
            print("alert dialog: choose image")
            return

        ctx = PipelineContext(self.origImagePlaceholder.image, self.debug,self.origImagePlaceholder.seeds)

        self.ctx.data={'image':self.origImagePlaceholder.image, 'debug':self.debug, 'roi':self.origImagePlaceholder.image}

        self.ctx.params = {key: value for ctrl in self.controlWidget.controls for key, value in ctrl.ctx.params.items() }

        if self.origImagePlaceholder.seeds is not None:
            self.ctx.set('seeds',self.origImagePlaceholder.seeds)


        #mask = self.logic.executeAlg2(self.image[:,:,self.slice]).astype(np.uint8)
        masks = self.pipeline.run(self.ctx)
        mask = masks[0]
        masks = np.moveaxis(np.array(masks), 0, -1)

        self.resultImagePlaceholder.images = masks
        self.resultImagePlaceholder.originalImage = self.origImagePlaceholder.image
        self.resultImagePlaceholder.displayImage()

        '''if mask is not None:
            normImg = cv2.normalize(self.image[:, :, self.slice].astype(np.float32), None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)
            cvImage = cv2.cvtColor(normImg, cv2.COLOR_GRAY2RGB)

            h, w, _ = cvImage.shape

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                cv2.drawContours(cvImage, [c], -1, (255, 0, 0), thickness=2)

            qim = QImage(cvImage, w, h, cvImage.strides[0], QImage.Format_RGB888)
            #self.display_image(QPixmap(qim))

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

        '''
        return

    def loadNifti(self, path):
        img = nib.load(path)
        return img.get_fdata()