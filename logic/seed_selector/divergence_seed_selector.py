import cv2
import numpy as np

from logic.preprocessing.preprocessing_step import enhanceImage
from logic.seed_selector.seed_selector import SeedSelector

class DivergenceSeedSelector(SeedSelector):
    def __init__(self):
        self.debug=False

    def calcRoiHistogram(self,region):
        hist, bin_edges = np.histogram(region, bins=256, range=(0.0, 1.0))
        hist[0] = 0  # ha üres terület belelóg nem érdekel (divergence-nél is negatív lenne)
        return  hist, bin_edges

    def setOptimalThreshold(self,div,bin_edges):
        idx = np.argmax(div)
        tail = div[idx + 1:]

        nearest_zero = np.argmin(np.abs(tail))
        optimal_threshold_idx = nearest_zero + 1 + idx
        optimal_threshold = bin_edges[optimal_threshold_idx]

        return idx,tail,nearest_zero, optimal_threshold_idx,optimal_threshold
    def select(self,ctx):
        self.debug=ctx.get('debug',False)
        region = ctx.get('roi',ctx.get('image'))
        image = enhanceImage(ctx.get('image'))
        ctx.set('image',image)
        ''' 1. Calc Histogram '''
        hist, bin_edges = self.calcRoiHistogram(region)

        ''' 2. Calc Divergence '''
        div = np.gradient(hist)

        ''' 3. Set optimal threshold'''
        idx,tail, nearest_zero, optimal_threshold_idx,optimal_threshold = self.setOptimalThreshold(div,bin_edges)
        if self.debug:
            self.showOptThreshold(hist,bin_edges,div,idx,optimal_threshold_idx)
        ''' 4. Automatic seed selection'''
        mask = np.zeros_like(image)
        mask[region >= bin_edges[nearest_zero + 1 + idx]] = 1
        seeds = np.argwhere(mask == 1)  # (row,col ) (y,x)

        if self.debug:
            self.showSeeds(image, region,mask)

        ctx.set('optimal_threshold',optimal_threshold)
        ctx.set('seeds',seeds)
        return ctx

    def showOptThreshold(self,hist,bin_edges,div,idx,optimal_threshold_idx):
        from matplotlib import pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.plot(hist)
        plt.title("histogram")
        plt.xticks(np.arange(0, 256, 50), np.round(bin_edges[np.arange(0, 256, 50)], 2))
        plt.subplot(1, 2, 2)
        plt.plot(div)
        plt.xticks(np.arange(0, 256, 50), np.round(bin_edges[np.arange(0, 256, 50)], 2))
        plt.scatter(idx, div[idx], c="red")
        plt.scatter(optimal_threshold_idx, div[optimal_threshold_idx], c="green")
        plt.title(f"Divergence, Thres: ({bin_edges[optimal_threshold_idx]},{div[optimal_threshold_idx]})")

        plt.show()
    def showSeeds(self, image, region,mask):
        from matplotlib import pyplot as plt
        coloredImage = cv2.cvtColor(
            cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2BGR)
        regionCopy = region
        regionCopy[regionCopy != 0] = 1
        regionCopy = regionCopy.astype(np.uint8)
        contours, _ = cv2.findContours(regionCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            cv2.drawContours(coloredImage, [c], -1, (255, 0, 0), thickness=2)

        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.title("teljes kép")
        plt.imshow(coloredImage)
        plt.subplot(1, 2, 2)
        plt.title("seedpontok")
        plt.imshow(mask, cmap="binary")
        plt.show()