from logic.region_growing.region_growing import RegionGrowingStrategy
import numpy as np
import cv2


class AccOpenCVRegionGrowing(RegionGrowingStrategy):
    def __init__(self, newVal=1, loDiff=35,upDiff=35,flags=4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (
                        255 << 8)):
        self.loDiff=loDiff
        self.upDiff=upDiff
        self.newVal = newVal
        self.flags = flags
        self.debug = False

    def regionGrowing(self,img,seed):
        mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
        floodFill = img.copy()

        retval, resImage, mask, rect = cv2.floodFill(
            cv2.normalize(floodFill, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), mask, seed, self.newVal,
            self.loDiff,
            self.upDiff,
            flags=self.flags)
        return mask[1:-1,1:-1]


    def run(self, ctx):
        self.debug = ctx.get('debug',False)
        image = ctx.get('image')
        seeds = ctx.get('seeds')

        '''5,6,7 region growing'''
        accumulatedMask = np.zeros_like(image)
        for seed in seeds:
            if accumulatedMask[
                seed[0], seed[1]] == 1:  # Ha még nem tartozik hozzá egy segmented régióhoz se akkor kell csak számolni
                continue
            segmented = self.regionGrowing(image, seed)
            accumulatedMask[segmented == 1] = 1
            # accumulatedMask |= segmented
        if self.debug:
            from matplotlib import pyplot as plt
            plt.figure(figsize=(12, 8))

            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(accumulatedMask, cmap="binary")
            plt.title("Segmented version")
        return [accumulatedMask]
