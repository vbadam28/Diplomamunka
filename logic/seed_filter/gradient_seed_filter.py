import cv2
import numpy as np

class GradientSeedFilter:

    def __init__(self):
        self.debug=False


    def filter(self,ctx):
        self.debug = ctx.get('debug')
        img = ctx.get('image')
        seeds = ctx.get('seeds')

        gradX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gradY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        # gradY = ndimage.sobel(img, axis=0)
        sobel = np.hypot(gradX, gradY)


        sobelBlur = cv2.GaussianBlur(sobel, (7, 7), 0)
        edgeMask = (sobel > sobelBlur * 1.5).astype(np.uint8)
        edgeMaskDil = cv2.dilate(edgeMask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))


        seedMask = np.zeros_like(img, dtype=np.uint8)
        for x, y in seeds:
            seedMask[y, x] = 1

        filtered = seedMask & (1 - edgeMaskDil)



        selectedSeeds = np.argwhere(filtered == 1)[:,::-1]


        if self.debug:
            self.showDebug(img,sobel,edgeMask,edgeMaskDil,selectedSeeds,seeds)

        ctx.set('seeds',selectedSeeds)

        return ctx

    def showDebug(self,img,sobel,edgeMask,edgeMaskDil,selectedSeeds,seeds):
        from matplotlib import pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 3, 1)
        plt.axis("off")
        plt.imshow(sobel, cmap="gray")
        plt.subplot(2, 3, 2)
        plt.imshow(edgeMask,cmap="gray")
        plt.axis("off")
        plt.title(f"sobel img {len(selectedSeeds)}/{len(seeds)}")
        plt.subplot(2, 3, 3)
        plt.axis("off")
        plt.imshow(edgeMaskDil,cmap="gray")


        if len(selectedSeeds) > 0:
            #coloredMask=np.zeros((img.shape[0],img.shape[1],3))
            coloredImg = cv2.cvtColor(cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U),cv2.COLOR_GRAY2RGB)
            for x,y in seeds:
                coloredImg[y,x] = (255,0,0)
            for x, y in selectedSeeds:
                coloredImg[y, x] = (0, 255, 0)

            plt.subplot(2, 3, 5)
            plt.axis("off")
            plt.imshow(coloredImg, cmap="gray")
            #plt.scatter(x=selectedSeeds[:, 0], y=selectedSeeds[:, 1], c="red")

        plt.show()