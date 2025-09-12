import sys

from PySide6.QtWidgets import QApplication
from gui.window import MainWindow
if __name__ == '__main__':
    app = QApplication(sys.argv)
    wd = MainWindow()
    wd.show()
    app.exec()

'''
    l= DebugLogic()
    path = '../content/data_ni/BraTS20_Training_355_flair.nii'
    nifti_img = l.loadNifti(path)
    seg_path = '../content/data_ni/W39_1998.09.19_Segm.nii'
    nifti_img_seg = l.loadNifti(seg_path)

    layer = 100
    wds = l.getWindows(nifti_img[:,:,layer],(7,7),(2,2))

    print(nifti_img[:,:,layer].shape)
    print(wds.shape)

    from matplotlib import pyplot as plt
    import cv2
    import numpy as np


    test = cv2.imread("../content/data_ni/test2.png",cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    test = cv2.normalize(cv2.resize(test,(240,240)),None, 0, 1, cv2.NORM_MINMAX)
    test = nifti_img[:,:,layer].copy()
    b = []#l.executeCustomAlg1(test.copy())#.astype(np.uint8))
    #pipeline = PipelineFactory.select5Seeds()
    #pipeline.options.debug=True
    from logic.pipeline.pipelineFactory import PipelineFactory
    from logic.pipeline.pipelineContext import PipelineContext
    from logic.preprocessing.sliding_window import SlidingWindow
    from logic.preprocessing.preprocessing_step import enhanceImage,PreprocessingStep
    from logic.seed_selector.window_seed_selector import WindowSeedSelector
    from logic.region_growing.opencv_region_growing import OpenCVRegionGrowing
    ctx = PipelineContext(test.copy(),False)
    pl = PipelineFactory.slidingWindows()
    a = pl.run(ctx)
    #a = pl.run(ctx)
    for m in a:
        plt.imshow(m,cmap="gray")
        plt.show()

    plt.subplot(1,2,1)
    plt.title('régi')
    #plt.imshow(b,cmap='gray')
    #a = pipeline.run(test)
    plt.subplot(1,2,2)
    plt.imshow(a[0], cmap="gray")
    plt.show()
'''
if False:
    path = '../content/data_ni/BraTS20_Training_355_flair.nii'
    nifti_img = loadNifti(path)
    seg_path = '../content/data_ni/W39_1998.09.19_Segm.nii'
    nifti_img_seg = loadNifti(seg_path)

    layer = 100


    testQuad=cv2.imread("test_quad.png", cv2.IMREAD_GRAYSCALE)
    testQuad[200:,:] = 0
    testQuad[:,:10] = 0
    testQuad = cv2.resize(testQuad,(240,240))
    testQuad = cv2.normalize(testQuad.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    testQuad2 = enhanceImage(testQuad)
    test = np.zeros_like(testQuad)
    test[140:180, 120:180] = 1
    testRegion = testQuad2.copy()
    testRegion[test==0] = 0

    #plt.imshow(testRegion, cmap="gray")
    #plt.show()

    QTree = QuadTree(nifti_img[:,:,layer])
    treeRes =  QTree.result

    #findSeedPoints(testRegion, testQuad)
    mask = findSeedPoints(treeRes, nifti_img[:,:,layer],nifti_img_seg[:,:,layer])



    #INNEN
    #select5Seed(nifti_img[:,:,layer],nifti_img_seg[:, :, layer])


    '''#showSlices([(nifti_img,'gray'),(nifti_img_seg,None)],100)
    #skullStripping(nifti_img[:,:,100])

    #test = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
    #skullStrippedImg = skullStripping(test)
    width  = skullStrippedImg.shape[0] if skullStrippedImg.shape[0]%8==0 else 256 #btw resize img kéne
    height = skullStrippedImg.shape[1] if skullStrippedImg.shape[1]%8==8 else 256

    copyImg = cv2.cvtColor(cv2.resize(cv2.normalize(skullStrippedImg,None,0,255, cv2.NORM_MINMAX).astype(np.uint8),(width,height)),cv2.COLOR_GRAY2BGR)
    #copyImg = cv2.cvtColor(skullStrippedImg,cv2.COLOR_GRAY2BGR)
    '''

    threshold = 20  # Adjust this value as needed

    # Segment the image
    #result = split_and_merge(nifti_img[:,:,layer], threshold)
    #cv2.imshow("a",result)

    #cv2.waitKey(0)

'''
    with h5py.File('../content/data/volume_355_slice_100.h5', 'r') as f:
        # Print the structure of the file
        print(f.keys())
        print(len(f))
        # Access a specific dataset
        image_data = f['image'][:]
        print("image",image_data.shape)
        mask_data = f['mask'][:]
        print("mask",mask_data.shape)



        cv2.imshow("IMAGE flair",cv2.normalize(image_data[ :, :, 0],None,0,255, cv2.NORM_MINMAX).astype(np.uint8))
        cv2.imshow("IMAGE t1",cv2.normalize(image_data[ :, :, 1],None,0,255, cv2.NORM_MINMAX).astype(np.uint8))
        cv2.imshow("IMAGE t1ce",cv2.normalize(image_data[ :, :, 2],None,0,255, cv2.NORM_MINMAX).astype(np.uint8))
        cv2.imshow("IMAGE t2",cv2.normalize(image_data[ :, :, 3],None,0,255, cv2.NORM_MINMAX).astype(np.uint8))
        cv2.imshow("MASK", cv2.normalize(mask_data,None,0,255, cv2.NORM_MINMAX).astype(np.uint8))


        cv2.waitKey(0)

        mask_data_norm = mask_data *255


        plt.figure(figsize=(12,8))

        plt.subplot(2,4,1)
        plt.imshow(image_data[:,:,0], cmap='gray')
        plt.title("Image Flair")
        plt.subplot(2,4,2)
        plt.imshow(image_data[:,:,1],cmap='gray')
        plt.title("Image t1")
        plt.subplot(2,4,3)
        plt.imshow(image_data[:,:,2],cmap='gray')
        plt.title("Image t1ec")
        plt.subplot(2,4,4)
        plt.imshow(image_data[:,:,3],cmap='gray')
        plt.title("Image t2")
        plt.subplot(2,4,5)
        plt.imshow(mask_data_norm)
        plt.title('Mask')

        plt.subplot(2, 4, 6)
        plt.imshow(mask_data[:, :, 0],cmap='gray')
        plt.title('Mask1')
        plt.subplot(2, 4, 7)
        plt.imshow(mask_data[:, :, 1],cmap='gray')
        plt.title('Mask2')
        plt.subplot(2, 4, 8)
        plt.imshow(mask_data[:, :, 2],cmap='gray')
        plt.title('Mask3')

        plt.show()

'''


