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
        contours, _ = cv2.findContours(bw.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        largest_blob_mask = np.zeros(bw.shape, dtype=np.uint8)

        cv2.drawContours(largest_blob_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        return largest_blob_mask

    def fillHoles(self, bw, se, largest_blob_mask, start_point=(0,0)):
        mask = np.zeros((bw.shape[0] + 2, bw.shape[1] + 2), dtype=np.uint8)
        retval, resImage, bw, rect = cv2.floodFill(bw, mask, start_point, 255, 30, 30,
                                                   flags=4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (
                                                           255 << 8))

        bw = cv2.morphologyEx(largest_blob_mask, cv2.MORPH_CLOSE, se)
        return bw

    def process(self, ctx):
        self.debug = ctx.get('debug',False)
        img = ctx.get('roi',ctx.get('image'))
        ''' Calculate Otsu’s Threshold '''
        norm_img, thres, orig_thres = self.calculateOtsuThreshold(img)

        ''' Threshold the image '''
        bw = self.thresholdImage(img,orig_thres,norm_img,thres)
        if self.debug:

            from matplotlib import pyplot as plt

            plt.figure(figsize=(12, 8))

            plt.subplot(2, 4, 1)
            plt.imshow(norm_img, cmap='gray')
            plt.title("Original Norm")

            plt.subplot(2, 4, 2)
            plt.imshow(bw, cmap='gray')
            plt.title("Otsu mask")

        ''' Open the binary image using a disk structuring Seed '''
        se, bw = self.diskStructuring(bw)
        if self.debug:
            plt.subplot(2, 4, 3)
            plt.imshow(bw, cmap='binary')
            plt.title("Opening")
        '''5: Dilate the binary image'''
        bw = cv2.morphologyEx(bw, cv2.MORPH_DILATE, se)
        if self.debug:
            plt.subplot(2, 4, 4)
            plt.imshow(bw, cmap='binary')
            plt.title("Dilate")

        '''6: Select the largest binary image'''
        largest_blob_mask = self.selectLargestBinaryImg(bw)
        if self.debug:
            plt.subplot(2, 4, 5)
            plt.imshow(largest_blob_mask, cmap='binary')
            plt.title("largest_blob")
        ''' 7: Dilate the binary image ''' '''     BW ← imclose(BW,se) --->?'''
        bw = cv2.morphologyEx(largest_blob_mask, cv2.MORPH_CLOSE, se)
        if self.debug:
            plt.subplot(2, 4, 6)
            plt.imshow(bw, cmap='binary')
            plt.title("IMCLOSE")
        ''' 8: Fill holes on the binary image ''' '''BW ← imfill(BW,se)'''
        bw = self.fillHoles(bw, se, largest_blob_mask)
        if self.debug:
            plt.subplot(2, 4, 7)
            plt.imshow(bw, cmap='binary')
            plt.title("IMFILL")
        ''' 9: Remove the skull '''   ''''stripped ← im(!BW) = 0'''
        norm_img[~bw.astype(bool)] = 0
        if self.debug:
            plt.subplot(2, 4, 8)
            plt.imshow(norm_img, cmap='gray')
            plt.title("Stripped norm_img")

        img[~bw.astype(bool)] = 0
        norm_img[~bw.astype(bool)] = 0
        if self.debug:
            plt.show(block=True)
        ctx.set('roi',img)
        return ctx  # norm_img
