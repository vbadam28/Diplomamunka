import cv2
import numpy as np
from logic.region_growing.region_growing import RegionGrowingStrategy

class OpenCVRegionGrowing(RegionGrowingStrategy):
    def __init__(self, newVal=255, loDiff=75,upDiff=75,flags=4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (
                        255 << 8)):
        self.loDiff=loDiff
        self.upDiff=upDiff
        self.newVal = newVal
        self.flags = flags
        self.debug = False

    #cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    def run(self, ctx): # top_k masj
        self.debug = ctx.get('debug',False)
        img=ctx.get('image')
        seeds=ctx.get('seeds')
        self.loadParams(ctx)
        width = 240
        height = 240

        masks = []

        seedImg = cv2.cvtColor(
            cv2.resize(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), (width, height)),
            cv2.COLOR_GRAY2BGR)

        for i, seed in enumerate(seeds):

            cv2.circle(seedImg, seed, 2, [0, 0, 255], 1)
            mask = np.zeros((seedImg.shape[0] + 2, seedImg.shape[1] + 2), dtype=np.uint8)
            floodFill = img.copy()
            mask = np.zeros((floodFill.shape[0] + 2, floodFill.shape[1] + 2), dtype=np.uint8)

            retval, resImage, mask, rect = cv2.floodFill(
                cv2.normalize(floodFill, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), mask, seed, self.newVal, self.loDiff,
                self.upDiff,
                flags=self.flags)
            '''retval, resImage, mask, rect = cv2.floodFill(cv2.normalize(floodFill.astype(np.float32), None, 0, 1.0, cv2.NORM_MINMAX), mask, seed, 1.0,  0.3,
                                                         0.3,
                                                         flags=4  | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (
                                                                     255 << 8))'''

            masks.append(mask[1:-1,1:-1])

        if self.debug:
            from matplotlib import pyplot as plt
            plt.figure(figsize=(12, 8))
            for i in range(len(masks)):
                plt.subplot(2, 3, i + 1)
                plt.imshow(masks[i], cmap='binary')
                plt.title(f"Mask seed:({seeds[i][0]},{seeds[i][1]})")
            cv2.imshow("RESULT", seedImg)
            plt.show()

        return masks

    def loadParams(self,ctx):
        self.loDiff = ctx.params.get("cvrg:lo_diff", self.loDiff)
        self.upDiff = ctx.params.get("cvrg:up_diff", self.upDiff)
        return