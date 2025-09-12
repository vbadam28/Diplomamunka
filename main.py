import sys

from PySide6.QtWidgets import QApplication
from gui.window import MainWindow
if __name__ == '__main__':
    app = QApplication(sys.argv)
    wd = MainWindow()
    wd.show()
    app.exec()

'''
    #LOAD IMG
    path = 'data_ni/BraTS20_Training_355_flair.nii'
    seg_path = 'data_ni/W39_1998.09.19_Segm.nii'
    import nibabel as nibb
    img  = nibb.load(path)
    nifti_img =  img.get_fdata()
    seg_img  = nibb.load(path)
    nifti_img_seg =  seg_img.get_fdata()

    layer = 100
    from matplotlib import pyplot as plt
    import cv2
    import numpy as np

    test = nifti_img[:,:,layer].copy()
    #IMPORT
    from logic.pipeline.pipelineFactory import PipelineFactory
    from logic.pipeline.pipelineContext import PipelineContext
    #INIT PIPELINE
    ctx = PipelineContext(test.copy(),False)
    pl = PipelineFactory.slidingWindows()

    masks = pl.run(ctx)

    #SHOW RESULT
    for m in masks:
        plt.figure()
        plt.imshow(m,cmap="gray")
        plt.show()

    #plt.imshow(b,cmap='gray')
    #a = pipeline.run(test)
    plt.figure()
    plt.title('mask 0')
    plt.imshow(masks[0], cmap="gray")
    plt.show()
'''