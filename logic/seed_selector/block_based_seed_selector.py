import cv2
import numpy as np
from logic.seed_selector.seed_selector import SeedSelector

class BlockBasedSeedSelector(SeedSelector):#for first method
    def __init__(self,blockSize=8, topK=5):
        self.blockSize=blockSize
        self.topK=topK
        self.debug = False

    def select(self,ctx):
        img = ctx.get('roi',ctx.get('image'))
        self.debug = ctx.get("debug",False)
        '''  Resize the Image''' '''im ← imresize(im, [256, 256]) '''
        resized_im = np.array(img).copy()  # cv2.resize(img,[256,256])

        ''' Iterate through each 8 × 8 block 
                Collect the mean of each block
                Collect the centers of each block
        '''
        width = resized_im.shape[0] if resized_im.shape[0] % 8 == 0 else 256  # btw resize img kéne
        height = resized_im.shape[1] if resized_im.shape[1] % 8 == 0 else 256
        if width == 256 or height == 256:
            resized_im = cv2.resize(img, [256, 256])
            width = 256
            height = 256

        mIs = []
        cBs = []
        for i in range(0, width, 8):
            for j in range(0, height, 8):
                block = resized_im[i:i + 8, j:j + 8]
                mIs.append(np.mean(block))
                cBs.append((j + 3, i + 3))

        ''' 10: Select top 5 blocks based on the intensity '''

        top5Indicies = np.argsort(mIs)[-5:][::-1]
        seeds = np.array(cBs)[top5Indicies]
        ctx.set('seeds',seeds)

        if self.debug:
            self.showSeeds(img,seeds)

        return ctx
    ''' for scoring the first Alg
    12: for m = 1 : 5 do
    13: ROIm= Region-growing(seedm)
    14: end for
    15: Compare each ROIm against GT using evaluation parameters for m = 1 : 5
    16: Select the best ROI as a final segmentation output.
    '''

    def showSeeds(self,image,seeds):
        from matplotlib import pyplot as plt
        coloredImage = cv2.cvtColor(
            cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2BGR)

        for seed in seeds:
            cv2.circle(coloredImage,center=seed,radius=4,color=(255,0,0))

        plt.figure()
        plt.title('seeds')
        plt.imshow(coloredImage)
        plt.show()