import cv2
import numpy as np

class SkullStripping:
    def __init__(self):
        self.debug=False

    def calculateOtsuThreshold(self,img):
        norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        thres, res = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Az Otsu csak normalizált
        orig_thres = (thres / 255) * (img.max() - img.min()) + img.min()
        return norm_img, thres, orig_thres

    def thresholdImage(self,img,orig_thres,norm_img,thres):
        img_c = img.copy()
        img_c[img_c < orig_thres] = 0
        norm_img_c = norm_img.copy()
        norm_img_c[norm_img_c < thres] = 0

        _, bw = cv2.threshold(img, thres, 255, cv2.THRESH_BINARY)
        return bw

    def diskStructuring(self, bw,radius=4):
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, se)
        return se,bw

    def selectLargestBinaryImg(self,bw):

        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw.astype(np.uint8), connectivity=8)
        largestLabel = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        largestComponent = np.zeros_like(bw)
        largestComponent[labels == largestLabel] = 255


        return largestComponent

    def fillHoles(self, bw, se, largest_blob_mask, start_point=(0,0)):
        mask = np.zeros((bw.shape[0] + 2, bw.shape[1] + 2), dtype=np.uint8)
        inputMask = bw.copy().astype(np.uint8)
        cv2.floodFill(inputMask, mask, start_point, 255)

        inputMaskInv = cv2.bitwise_not(inputMask)
        outImg = cv2.bitwise_or(bw.astype(np.uint8), inputMaskInv)


        return outImg

    def process(self, ctx):
        self.debug = ctx.get('debug',False)
        img = ctx.get('roi',ctx.get('image'))
        ''' Calculate Otsu’s Threshold '''
        norm_img, thres, orig_thres = self.calculateOtsuThreshold(img)

        ''' Threshold the image '''
        bw = self.thresholdImage(img,orig_thres,norm_img,thres)
        from matplotlib import pyplot as plt

        if self.debug:
            plt.figure(figsize=(12, 8))
            self.showStep(plt, 1, norm_img, "Original Norm")
            self.showStep(plt, 2, bw, "Otsu mask")

        ''' Open the binary image using a disk structuring Seed '''
        se, bw = self.diskStructuring(bw)
        if self.debug:
            self.showStep(plt, 3, bw, "Opening")

        '''5: Dilate the binary image'''
        bw = cv2.morphologyEx(bw, cv2.MORPH_DILATE, se)
        if self.debug:
            self.showStep(plt, 4, bw, "Dilate")

        '''6: Select the largest binary image'''
        largest_blob_mask = self.selectLargestBinaryImg(bw)
        if self.debug:
            self.showStep(plt, 5, largest_blob_mask, "largest blob")

        ''' 7: Close the binary image ''' '''     BW ← imclose(BW,se) --->?'''
        bw = cv2.morphologyEx(largest_blob_mask, cv2.MORPH_CLOSE, se)
        if self.debug:
            self.showStep(plt, 6, bw, "IMCLOSE")

        ''' 8: Fill holes on the binary image ''' '''BW ← imfill(BW,se)'''
        bw = self.fillHoles(bw, se, largest_blob_mask)
        if self.debug:
            self.showStep(plt, 7, bw, "IMFILL")

        ''' 9: Remove the skull '''   ''''stripped ← im(!BW) = 0'''
        norm_img[~bw.astype(bool)] = 0
        if self.debug:
            self.showStep(plt, 8, norm_img, "Stripped norm_img")


        img[~bw.astype(bool)] = 0
        norm_img[~bw.astype(bool)] = 0
        if self.debug:
            plt.show(block=True)
        ctx.set('roi',img)
        return ctx  # norm_img

    def showStep(self, plt, i, img, title):
        plt.subplot(2, 4, i)
        plt.imshow(img, cmap='gray')
        plt.title(title)
